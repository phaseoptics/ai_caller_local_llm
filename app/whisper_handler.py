import os
import logging
import asyncio
import io
import wave
import time
from openai import AsyncOpenAI
from app.twilio_stream_handler import ready_chunks, transcription_queue, detected_phrases
from app.data_types import AudioChunk, PhraseObject
from app.conversation_manager import handle_phrase

# Queue retained for future stitching logic (unchanged API surface)
stitch_ready_chunks: asyncio.Queue = asyncio.Queue()

# --- Logging Configuration ---
logger = logging.getLogger("whisper_handler")
logger.setLevel(logging.INFO if os.getenv("LOGGING_ENABLED", "false").lower() == "true" else logging.CRITICAL)

# --- OpenAI Whisper Client ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def _pcm_to_wav_bytesio(audio_bytes: bytes, sample_rate: int = 8000) -> io.BytesIO:
    """
    Wrap raw PCM16 mono @ sample_rate Hz in a WAV header entirely in memory.
    Returns a BytesIO positioned at start, with a .name for the OpenAI client.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)        # 16-bit PCM
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    buf.seek(0)
    # Some SDK paths like having a filename
    buf.name = "chunk.wav"  # type: ignore[attr-defined]
    return buf


async def transcribe_audio(audio_bytes: bytes) -> tuple[str, dict]:
    """
    Build WAV in-memory and send to Whisper. Returns (transcript, timings).
    Timings dict includes:
      - wav_build_ms
      - whisper_ms
      - total_ms
    """
    t0 = time.perf_counter()
    wav_buf = _pcm_to_wav_bytesio(audio_bytes, sample_rate=8000)
    t1 = time.perf_counter()

    transcript_text = ""
    whisper_ms = 0.0
    try:
        resp_start = time.perf_counter()
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buf,
            response_format="text"
        )
        whisper_ms = (time.perf_counter() - resp_start) * 1000.0
        transcript_text = str(response).strip()
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        transcript_text = ""

    t2 = time.perf_counter()
    timings = {
        "wav_build_ms": (t1 - t0) * 1000.0,
        "whisper_ms": whisper_ms,
        "total_ms": (t2 - t0) * 1000.0,
    }
    return transcript_text, timings


async def whisper_transcription_loop():
    logger.info("Whisper transcription loop started.")
    processed_chunks = set()

    while True:
        try:
            chunk: AudioChunk = await transcription_queue.get()

            # Skip if already handled
            if chunk.transcription != "" or chunk.chunk_index in processed_chunks:
                continue

            # ---- Transcribe this chunk (in-memory, no disk I/O) ----
            transcript, timings = await transcribe_audio(chunk.audio_bytes)

            if not transcript:
                logger.warning(f"Whisper empty transcript for chunk {chunk.chunk_index} (phrase {chunk.phrase_id}).")

            # Update the shared chunk object
            chunk.transcription = transcript
            processed_chunks.add(chunk.chunk_index)

            # Timing/latency diagnostics
            # duration is already stored in the chunk (seconds)
            logger.info(
                "WHISPER | phrase=%s chunk=%d dur=%.3fs bytes=%d | build=%.1fms whisper=%.1fms total=%.1fms | text='%s'",
                chunk.phrase_id,
                chunk.chunk_index,
                float(chunk.duration),
                len(chunk.audio_bytes),
                timings["wav_build_ms"],
                timings["whisper_ms"],
                timings["total_ms"],
                (transcript[:80] + "…") if len(transcript) > 80 else transcript
            )

            await stitch_ready_chunks.put(chunk)

            # If the whole phrase has transcripts, kick off the LLM path once
            phrase: PhraseObject | None = detected_phrases.get(chunk.phrase_id)
            if phrase and phrase.is_complete() and not phrase.is_done:
                phrase.is_done = True
                logger.debug(f"Phrase {phrase.phrase_id} complete. Launching LLM task.")
                asyncio.create_task(handle_phrase(phrase))

        except asyncio.CancelledError:
            logger.info("Whisper transcription loop cancelled.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in transcription loop: {e}")
