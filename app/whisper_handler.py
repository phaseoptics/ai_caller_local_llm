import os
import logging
import asyncio
import io
import wave
import time
from typing import Tuple, Dict, Optional

import numpy as np
from openai import AsyncOpenAI
from app.twilio_stream_handler import ready_chunks, transcription_queue, detected_phrases
from app.data_types import AudioChunk, PhraseObject
from app.conversation_manager import handle_phrase

# --------------------------------------------------------------------------------------
# CONFIG: Flip this to False to use the OpenAI Whisper API instead of local faster-whisper
# --------------------------------------------------------------------------------------
USE_LOCAL_WHISPER = True

# Local model defaults aimed at better accuracy on CPU (still reasonable latency)
LOCAL_MODEL_NAME = "small.en"          # try "medium.en" if you need even more accuracy (slower)
LOCAL_COMPUTE_TYPE = "int8_float32"    # higher fidelity than int8; faster than full float32
LOCAL_BEAM_SIZE = 5                    # beam search for accuracy; raise/lower to trade accuracy/latency
FORCE_LANGUAGE = "en"                  # skip detection for speed/stability
DOWNLOAD_ROOT = "/mnt/data/models/faster-whisper"

# Queue retained for future stitching logic (unchanged API surface)
stitch_ready_chunks: asyncio.Queue = asyncio.Queue()

# --- Logging Configuration ---
logger = logging.getLogger("whisper_handler")
logger.setLevel(logging.INFO if os.getenv("LOGGING_ENABLED", "false").lower() == "true" else logging.CRITICAL)

# --- OpenAI Whisper Client (for API path) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- faster-whisper globals (lazy init) ---
_faster_model = None

# --------------------------------------------------------------------------------------
# Utility: 8 kHz PCM16 (bytes) -> 16 kHz float32 numpy array in [-1, 1]
# Simple 2x linear upsample (fast, adequate for telephony)
# --------------------------------------------------------------------------------------
def pcm8k_bytes_to_float32_16k(audio_bytes: bytes) -> Tuple[np.ndarray, Dict[str, float]]:
    t0 = time.perf_counter()
    pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)

    # Normalize to float32 [-1, 1]
    x = pcm16.astype(np.float32) / 32768.0

    if x.size == 0:
        up = np.zeros(0, dtype=np.float32)
    else:
        up = np.empty(x.size * 2, dtype=np.float32)
        up[0::2] = x
        # Midpoints between samples (last sample repeats)
        if x.size > 1:
            up[1:-1:2] = (x[:-1] + x[1:]) * 0.5
            up[-1] = x[-1]
        else:
            up[1] = x[0]

    t1 = time.perf_counter()
    return up, {"upsample_ms": (t1 - t0) * 1000.0}

# --------------------------------------------------------------------------------------
# API path (fallback) — in-memory WAV, no disk I/O
# --------------------------------------------------------------------------------------
def _pcm_to_wav_bytesio(audio_bytes: bytes, sample_rate: int = 8000) -> io.BytesIO:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)        # 16-bit PCM
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    buf.seek(0)
    buf.name = "chunk.wav"  # type: ignore[attr-defined]
    return buf

async def _transcribe_api(audio_bytes: bytes) -> Tuple[str, Dict]:
    t0 = time.perf_counter()
    wav_buf = _pcm_to_wav_bytesio(audio_bytes, sample_rate=8000)
    t1 = time.perf_counter()

    transcript_text = ""
    whisper_ms = 0.0
    try:
        resp_start = time.perf_counter()
        response = await oai_client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buf,
            response_format="text"
        )
        whisper_ms = (time.perf_counter() - resp_start) * 1000.0
        transcript_text = str(response).strip()
    except Exception as e:
        logger.error(f"Transcription (API) failed: {e}")
        transcript_text = ""

    t2 = time.perf_counter()
    timings = {
        "path": "api",
        "wav_build_ms": (t1 - t0) * 1000.0,
        "whisper_ms": whisper_ms,
        "total_ms": (t2 - t0) * 1000.0,
    }
    return transcript_text, timings

# --------------------------------------------------------------------------------------
# faster-whisper local path (higher accuracy settings)
# --------------------------------------------------------------------------------------
def _lazy_load_faster_model():
    global _faster_model
    if _faster_model is None:
        from faster_whisper import WhisperModel
        os.makedirs(DOWNLOAD_ROOT, exist_ok=True)
        logger.info(f"Loading faster-whisper model='{LOCAL_MODEL_NAME}' device=cpu compute={LOCAL_COMPUTE_TYPE}")
        _faster_model = WhisperModel(
            LOCAL_MODEL_NAME,
            device="cpu",
            compute_type=LOCAL_COMPUTE_TYPE,
            download_root=DOWNLOAD_ROOT,
        )
        logger.info(f"Loaded faster-whisper model='{LOCAL_MODEL_NAME}' device=cpu compute={LOCAL_COMPUTE_TYPE}")
    return _faster_model

# Preload at import to avoid mid-call stall
if USE_LOCAL_WHISPER:
    _lazy_load_faster_model()

def _transcribe_local_sync(float16k: np.ndarray) -> str:
    model = _lazy_load_faster_model()
    segments, info = model.transcribe(
        float16k,
        beam_size=LOCAL_BEAM_SIZE,
        vad_filter=False,            # VAD already handled upstream
        language=FORCE_LANGUAGE,     # speed + stability
        without_timestamps=False,    # keep timestamps for latency analysis
    )
    return " ".join(seg.text.strip() for seg in segments if seg.text).strip()

async def _transcribe_local(audio_bytes: bytes, loop: Optional[asyncio.AbstractEventLoop] = None) -> Tuple[str, Dict]:
    t0 = time.perf_counter()
    float16k, up_timing = pcm8k_bytes_to_float32_16k(audio_bytes)
    t1 = time.perf_counter()

    if loop is None:
        loop = asyncio.get_running_loop()

    start_model = time.perf_counter()
    try:
        # Run the sync model in a threadpool to avoid blocking the event loop
        transcript_text = await loop.run_in_executor(None, _transcribe_local_sync, float16k)
    except Exception as e:
        logger.error(f"Transcription (local) failed: {e}")
        transcript_text = ""
    model_ms = (time.perf_counter() - start_model) * 1000.0

    t2 = time.perf_counter()
    timings = {
        "path": "local",
        "upsample_ms": up_timing["upsample_ms"],
        "whisper_ms": model_ms,
        "total_ms": (t2 - t0) * 1000.0,
    }
    return transcript_text, timings

# --------------------------------------------------------------------------------------
# Public entry used by the loop
# --------------------------------------------------------------------------------------
async def transcribe_audio(audio_bytes: bytes) -> Tuple[str, Dict]:
    if USE_LOCAL_WHISPER:
        return await _transcribe_local(audio_bytes)
    else:
        return await _transcribe_api(audio_bytes)

# --------------------------------------------------------------------------------------
# Main loop (unchanged logic, enhanced logs)
# --------------------------------------------------------------------------------------
async def whisper_transcription_loop():
    logger.info("Whisper transcription loop started.")
    processed_chunks = set()

    while True:
        try:
            chunk: AudioChunk = await transcription_queue.get()

            # Skip if already handled
            if chunk.transcription != "" or chunk.chunk_index in processed_chunks:
                continue

            # ---- Transcribe this chunk ----
            transcript, timings = await transcribe_audio(chunk.audio_bytes)

            if not transcript:
                logger.warning(f"Whisper returned empty transcript for chunk {chunk.chunk_index} (phrase {chunk.phrase_id}).")

            # Update the shared chunk object
            chunk.transcription = transcript
            processed_chunks.add(chunk.chunk_index)

            # Timing/latency diagnostics
            if timings.get("path") == "local":
                logger.info(
                    "WHISPER(local) | phrase=%s chunk=%d dur=%.3fs bytes=%d | upsample=%.1fms whisper=%.1fms total=%.1fms | text='%s'",
                    chunk.phrase_id,
                    chunk.chunk_index,
                    float(chunk.duration),
                    len(chunk.audio_bytes),
                    timings.get("upsample_ms", 0.0),
                    timings.get("whisper_ms", 0.0),
                    timings.get("total_ms", 0.0),
                    (transcript[:80] + "…") if len(transcript) > 80 else transcript
                )
            else:
                logger.info(
                    "WHISPER(api)   | phrase=%s chunk=%d dur=%.3fs bytes=%d | wav=%.1fms whisper=%.1fms total=%.1fms | text='%s'",
                    chunk.phrase_id,
                    chunk.chunk_index,
                    float(chunk.duration),
                    len(chunk.audio_bytes),
                    timings.get("wav_build_ms", 0.0),
                    timings.get("whisper_ms", 0.0),
                    timings.get("total_ms", 0.0),
                    (transcript[:80] + "…") if len(transcript) > 80 else transcript
                )

            await stitch_ready_chunks.put(chunk)

            # If the whole phrase has transcripts, kick off the LLM path once
            phrase: PhraseObject | None = detected_phrases.get(chunk.phrase_id)
            if phrase and phrase.is_complete() and not phrase.is_done:
                phrase.is_done = True
                asyncio.create_task(handle_phrase(phrase))

        except asyncio.CancelledError:
            logger.info("Whisper transcription loop cancelled.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in transcription loop: {e}")
