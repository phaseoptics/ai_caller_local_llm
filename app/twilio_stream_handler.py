# app/twilio_stream_handler.py
import os
import logging
import asyncio
import base64
import audioop
import time
import json
from collections import deque
from dataclasses import dataclass
from typing import Optional, AsyncIterator, Union, Any

from quart import Quart, Response, websocket, request, jsonify
from dotenv import load_dotenv
from asyncio import Queue
import uuid

from app.outbound_call_handler import create_outbound_call
from app.data_types import AudioChunk, PhraseObject
from app.queues import llm_playback_queue, tts_requests_queue
from app.queues import start_assistant_playing, stop_assistant_playing, reset_playback_pause_accumulator
from app.elevenlabs_handler import encode_mp3_to_ulaw_frames, stream_tts_ulaw_frames

from app.config import GREETING_TEXT
from app.transcript_manager import append_transcript_line


# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
call_ended = asyncio.Event()
call_active = asyncio.Event()
ready_chunks: Queue = Queue()          # Queue of AudioChunks to persist at shutdown
transcription_queue: Queue = Queue()   # Queue of AudioChunks for Whisper

logger = logging.getLogger("twilio_handler")
logger.setLevel(logging.INFO if os.getenv("LOGGING_ENABLED", "false").lower() == "true" else logging.CRITICAL)

# Load environment variables
load_dotenv()
ELEVEN_STREAMING = os.getenv("ELEVEN_STREAMING", "false").lower() == "true"

try:
    PLAYBACK_CLEAR_MARGIN = float(os.getenv("PLAYBACK_CLEAR_MARGIN", "0.25"))
except Exception:
    PLAYBACK_CLEAR_MARGIN = 0.25

PLAYBACK_CLEAR_AFTER_END = os.getenv("PLAYBACK_CLEAR_AFTER_END", "true").lower() == "true"


# ---- Caller silence tracking ----
_last_speech_time: float = time.monotonic()

def get_last_speech_time() -> float:
    return _last_speech_time

def _mark_speech_now() -> None:
    global _last_speech_time
    _last_speech_time = time.monotonic()
    logger.info("_mark_speech_now: updated last_speech_time=%f", _last_speech_time)
    try:
        reset_playback_pause_accumulator()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Quart app
# -----------------------------------------------------------------------------
app = Quart(__name__)

# phrase_id -> PhraseObject
detected_phrases: dict[str, PhraseObject] = {}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ulaw_to_pcm(ulaw_bytes: bytes) -> bytes:
    return audioop.ulaw2lin(ulaw_bytes, 2)

def calculate_rms_ulaw(ulaw_bytes: bytes) -> float:
    try:
        return float(audioop.rms(audioop.ulaw2lin(ulaw_bytes, 2), 2))
    except Exception as e:
        logger.error("RMS calculation error: %s", e)
        return 0.0

async def _send_dummy_frame(stream_sid: str, send_lock: asyncio.Lock) -> None:
    dummy_frame = base64.b64encode(b"\xff" * 160).decode()
    async with send_lock:
        await websocket.send_json({"event": "media", "streamSid": stream_sid, "media": {"payload": dummy_frame}})
    logger.info("🟡 Sent dummy frame to hold stream open.")

async def _send_clear(stream_sid: str, send_lock: asyncio.Lock) -> None:
    try:
        async with send_lock:
            await websocket.send_json({"event": "clear", "streamSid": stream_sid})
        logger.info("🧹 Sent clear to flush caller buffered audio.")
    except Exception as e:
        logger.error("Failed sending clear: %s", e)

async def _send_media_frame(stream_sid: str, payload_b64: str, send_lock: asyncio.Lock) -> None:
    async with send_lock:
        await websocket.send_json({"event": "media", "streamSid": stream_sid, "media": {"payload": payload_b64}})


def _normalize_tts_item(item: Any) -> Optional[str]:
    # Expected enriched payload: {"text": "..."}
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        t = item.get("text")
        if isinstance(t, str) and t.strip():
            return t
    return None

def _normalize_mp3_item(item: Any) -> tuple[Optional[str], Optional[str]]:
    # Expected enriched payload: {"mp3_path": "...", "text": "..."}
    if item is None:
        return None, None
    if isinstance(item, str):
        return item, None
    if isinstance(item, dict):
        p = item.get("mp3_path")
        t = item.get("text")
        mp3_path = p if isinstance(p, str) and p.strip() else None
        text = t if isinstance(t, str) and t.strip() else None
        return mp3_path, text
    return None, None


# -----------------------------------------------------------------------------
# Player job model
# -----------------------------------------------------------------------------
@dataclass
class PlayerJob:
    kind: str                      # "mp3" or "tts"
    value: str                     # mp3 path or text
    generation: int                # invalidate on barge in
    transcript_text: Optional[str] = None  # write only on normal completion


# -----------------------------------------------------------------------------
# Disk writer
# -----------------------------------------------------------------------------
async def write_all_chunks_to_disk(queue: asyncio.Queue, output_dir: str) -> None:
    import wave
    os.makedirs(output_dir, exist_ok=True)
    idx = 0
    while not queue.empty():
        chunk = await queue.get()
        out_path = os.path.join(output_dir, f"{chunk.phrase_id}__chunk_{chunk.chunk_index}.wav")
        with wave.open(out_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(8000)
            wav_file.writeframes(chunk.audio_bytes)
        logger.info("Wrote chunk to %s", out_path)
        idx += 1
    logger.info("Finished writing %d chunks to disk.", idx)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/voice", methods=["POST"])
async def voice_webhook():
    logger.info("Received /voice webhook from Twilio.")
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://youngbullserver.com/ai_caller/stream" />
  </Connect>
</Response>"""
    return Response(twiml, mimetype="text/xml")


@app.route("/call_mom", methods=["POST"])
async def call_mom():
    token_expected = os.getenv("CALL_TRIGGER_TOKEN", "").strip()
    if token_expected:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {token_expected}":
            return jsonify({"ok": False, "error": "Unauthorized"}), 401

    mom_number = os.getenv("MOM_PHONE_NUMBER", "").strip()
    public_base_url = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")

    if not mom_number:
        return jsonify({"ok": False, "error": "Missing MOM_PHONE_NUMBER"}), 500
    if not public_base_url:
        return jsonify({"ok": False, "error": "Missing PUBLIC_BASE_URL"}), 500

    twiml_url = f"{public_base_url}/voice"

    try:
        result = create_outbound_call(
            to_number=mom_number,
            twiml_url=twiml_url,
        )
        return jsonify(
            {
                "ok": True,
                "call_sid": result.get("sid"),
                "status": result.get("status"),
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -----------------------------------------------------------------------------
# True barge in WebSocket handler
# -----------------------------------------------------------------------------
@app.websocket("/stream")
async def media_stream():
    logger.info("WebSocket connection established with Twilio.")

    # VAD / chunking parameters
    try:
        MIN_SPEECH_RMS_THRESHOLD = float(os.getenv("TWILIO_MIN_SPEECH_RMS_THRESHOLD", "750"))
    except Exception:
        MIN_SPEECH_RMS_THRESHOLD = 750.0

    CHUNK_SILENCE_DURATION_SECONDS = float(os.getenv("CHUNK_SILENCE_DURATION_SECONDS", "0.55"))
    DONE_SPEAKING_SILENCE_DURATION_SECONDS = float(os.getenv("DONE_SPEAKING_SILENCE_DURATION_SECONDS", "1.2"))
    MINCHUNK_DURATION_SECONDS = float(os.getenv("MINCHUNK_DURATION_SECONDS", "1.0"))
    MAXCHUNK_DURATION_SECONDS = float(os.getenv("MAXCHUNK_DURATION_SECONDS", "10.0"))
    LEAD_IN_DURATION_SECONDS = float(os.getenv("LEAD_IN_DURATION_SECONDS", "0.25"))

    FRAME_DURATION_SECONDS = 0.02
    silence_chunk_limit = int(CHUNK_SILENCE_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    done_speaking_chunk_limit = int(DONE_SPEAKING_SILENCE_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    max_chunk_chunks = int(MAXCHUNK_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    lead_in_bytes = int(8000 * 2 * LEAD_IN_DURATION_SECONDS)

    # Barge in parameters
    try:
        BARGE_IN_MULTIPLIER = float(os.getenv("BARGE_IN_MULTIPLIER", "1.25"))
    except Exception:
        BARGE_IN_MULTIPLIER = 1.25

    try:
        BARGE_IN_CONSEC_FRAMES = int(os.getenv("BARGE_IN_CONSEC_FRAMES", "2"))
    except Exception:
        BARGE_IN_CONSEC_FRAMES = 2

    BARGE_IN_THRESHOLD = MIN_SPEECH_RMS_THRESHOLD * BARGE_IN_MULTIPLIER

    # Shared state
    send_lock = asyncio.Lock()
    barge_in_event = asyncio.Event()
    playback_active = asyncio.Event()
    shutdown_event = asyncio.Event()

    stream_sid: Optional[str] = None

    player_queue: asyncio.Queue[PlayerJob] = asyncio.Queue()
    generation = 0

    # VAD state
    phrase_id = str(uuid.uuid4())
    chunk_index = 0
    active_pcm = bytearray()
    pre_chunk_pcm_buffer = deque(maxlen=lead_in_bytes)
    silence_counter = 0
    phrase_silence_counter = 0
    in_chunk = False
    has_spoken = False
    first_media_logged = False

    # barge in detection state
    barge_in_hits = 0

    def _reset_vad_state_for_new_phrase() -> None:
        nonlocal phrase_id, silence_counter, phrase_silence_counter, in_chunk, has_spoken, active_pcm, pre_chunk_pcm_buffer
        phrase_id = str(uuid.uuid4())
        silence_counter = 0
        phrase_silence_counter = 0
        in_chunk = False
        has_spoken = False
        active_pcm = bytearray()
        pre_chunk_pcm_buffer.clear()

    async def _drain_external_assistant_queues() -> None:
        drained_tts = 0
        drained_mp3 = 0
        try:
            while not tts_requests_queue.empty():
                _ = await tts_requests_queue.get()
                drained_tts += 1
        except Exception:
            pass
        try:
            while not llm_playback_queue.empty():
                _ = await llm_playback_queue.get()
                drained_mp3 += 1
        except Exception:
            pass
        if drained_tts or drained_mp3:
            logger.info("Discarded queued assistant output: tts=%d mp3=%d", drained_tts, drained_mp3)

    async def _drain_player_queue() -> None:
        drained = 0
        try:
            while not player_queue.empty():
                _ = player_queue.get_nowait()
                drained += 1
        except Exception:
            pass
        if drained:
            logger.info("Discarded internal player jobs: %d", drained)

    async def _player_stream_frames(frames: Union[list[str], AsyncIterator[str]], job_generation: int) -> bool:
        nonlocal stream_sid
        if not stream_sid:
            return False

        interval = 0.02
        base = time.monotonic()
        i = 0

        def _should_stop() -> bool:
            if shutdown_event.is_set():
                return True
            if barge_in_event.is_set():
                return True
            if job_generation != generation:
                return True
            return False

        if isinstance(frames, list):
            for frame in frames:
                if _should_stop():
                    return False
                i += 1
                await _send_media_frame(stream_sid, frame, send_lock)
                target = base + i * interval
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            return True

        async for frame in frames:
            if _should_stop():
                return False
            i += 1
            await _send_media_frame(stream_sid, frame, send_lock)
            target = base + i * interval
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        return True

    async def player_loop() -> None:
        nonlocal generation
        while not shutdown_event.is_set():
            try:
                job: PlayerJob = await player_queue.get()
            except asyncio.CancelledError:
                return
            except Exception:
                continue

            if job.generation != generation:
                continue
            if not stream_sid:
                continue

            barge_in_event.clear()
            playback_active.set()
            spoken_text = job.transcript_text

            try:
                start_assistant_playing()
            except Exception:
                pass

            completed = False
            try:
                if job.kind == "mp3":
                    if not os.path.exists(job.value):
                        logger.error("Playback mp3 not found: %s", job.value)
                        completed = True
                    else:
                        frames = encode_mp3_to_ulaw_frames(job.value)
                        logger.info("📢 Playback mp3: %s frames=%d", os.path.basename(job.value), len(frames))
                        completed = await _player_stream_frames(frames, job.generation)

                elif job.kind == "tts":
                    text = job.value
                    logger.info("🔊 Playback TTS streaming chars=%d", len(text))
                    completed = await _player_stream_frames(stream_tts_ulaw_frames(text), job.generation)

                else:
                    logger.error("Unknown player job kind: %s", job.kind)
                    completed = True

            finally:
                playback_active.clear()
                try:
                    stop_assistant_playing()
                except Exception:
                    pass

            # Interrupted path: only write partial transcript
            if (not completed) and barge_in_event.is_set() and stream_sid:
                if spoken_text:
                    append_transcript_line("Assistant", f"{spoken_text} [interrupted]")
                logger.info("BARGE IN: stopping playback now, sending one clear, discarding assistant output")
                generation += 1
                await _drain_player_queue()
                await _drain_external_assistant_queues()
                await _send_clear(stream_sid, send_lock)
                continue

            # Normal completion: now write assistant transcript line
            if completed and job.transcript_text:
                append_transcript_line("Assistant", job.transcript_text)

            if completed and stream_sid and PLAYBACK_CLEAR_AFTER_END:
                if PLAYBACK_CLEAR_MARGIN > 0:
                    await asyncio.sleep(PLAYBACK_CLEAR_MARGIN)
                await _send_clear(stream_sid, send_lock)

    async def receiver_loop() -> None:
        nonlocal stream_sid
        nonlocal phrase_id, chunk_index, active_pcm, pre_chunk_pcm_buffer
        nonlocal silence_counter, phrase_silence_counter, in_chunk, has_spoken, first_media_logged
        nonlocal barge_in_hits

        while not shutdown_event.is_set():
            if call_ended.is_set():
                shutdown_event.set()
                try:
                    await websocket.close(code=1000)
                except Exception:
                    pass
                return

            raw = await websocket.receive()
            if raw is None:
                logger.warning("WebSocket closed by Twilio.")
                shutdown_event.set()
                return

            try:
                event = json.loads(raw)
            except Exception:
                continue

            event_type = event.get("event")

            if event_type == "start":
                logger.info("Twilio media stream started.")
                stream_sid = event.get("start", {}).get("streamSid", "placeholder")
                call_active.set()

                _mark_speech_now()
                await _send_dummy_frame(stream_sid, send_lock)

                # Greeting as a player job, transcript only after normal completion
                try:
                    greeting_path = "app/audio_static/greeting.mp3"
                    if os.path.exists(greeting_path):
                        await player_queue.put(
                            PlayerJob(kind="mp3", value=greeting_path, generation=generation, transcript_text=GREETING_TEXT)
                        )
                    else:
                        logger.warning("No greeting.mp3 found, skipping greeting playback.")
                except Exception as e:
                    logger.error("Failed to enqueue greeting: %s", e)

            elif event_type == "stop":
                logger.info("Twilio media stream stopped.")
                call_active.clear()
                call_ended.set()
                shutdown_event.set()
                return

            elif event_type == "media":
                payload = event.get("media", {}).get("payload")
                if not payload:
                    continue

                ulaw = base64.b64decode(payload)
                rms = calculate_rms_ulaw(ulaw)
                pcm = ulaw_to_pcm(ulaw)

                if not first_media_logged:
                    logger.info("First media frame RMS=%.1f (threshold=%.1f)", rms, MIN_SPEECH_RMS_THRESHOLD)
                    _mark_speech_now()
                    first_media_logged = True

                # Move assistant outputs from external queues into the internal player queue.
                # We now expect enriched payloads so transcript can be written only on completion.
                try:
                    while ELEVEN_STREAMING and not tts_requests_queue.empty():
                        item = await tts_requests_queue.get()
                        text = _normalize_tts_item(item)
                        if text:
                            await player_queue.put(
                                PlayerJob(kind="tts", value=text, generation=generation, transcript_text=text)
                            )

                    while (not ELEVEN_STREAMING) and (not llm_playback_queue.empty()):
                        item = await llm_playback_queue.get()
                        mp3_path, text = _normalize_mp3_item(item)
                        if mp3_path:
                            await player_queue.put(
                                PlayerJob(kind="mp3", value=mp3_path, generation=generation, transcript_text=text)
                            )
                except Exception:
                    pass

                # Barge in detection
                if playback_active.is_set():
                    if rms >= BARGE_IN_THRESHOLD:
                        barge_in_hits += 1
                    else:
                        barge_in_hits = 0

                    if barge_in_hits >= max(1, BARGE_IN_CONSEC_FRAMES):
                        if not barge_in_event.is_set():
                            logger.info(
                                "BARGE IN DETECTED rms=%.1f threshold=%.1f hits=%d",
                                rms,
                                BARGE_IN_THRESHOLD,
                                barge_in_hits,
                            )
                        barge_in_event.set()
                        barge_in_hits = 0
                        _reset_vad_state_for_new_phrase()

                if rms >= MIN_SPEECH_RMS_THRESHOLD:
                    _mark_speech_now()

                # VAD and phrase segmentation
                pre_chunk_pcm_buffer.extend(pcm)

                if not in_chunk and rms >= MIN_SPEECH_RMS_THRESHOLD:
                    if has_spoken and phrase_silence_counter >= done_speaking_chunk_limit:
                        logger.info("Long silence detected. Generating new phrase ID.")
                        phrase_id = str(uuid.uuid4())
                    has_spoken = True
                    in_chunk = True
                    silence_counter = 0
                    phrase_silence_counter = 0
                    active_pcm.extend(pre_chunk_pcm_buffer)
                    pre_chunk_pcm_buffer.clear()
                    logger.info("AudioChunk formation initiated at frame=%d phrase_id=%s", chunk_index, phrase_id)

                if in_chunk:
                    active_pcm.extend(pcm)

                    if rms < MIN_SPEECH_RMS_THRESHOLD:
                        silence_counter += 1
                        phrase_silence_counter += 1
                    else:
                        silence_counter = 0
                        phrase_silence_counter = 0

                    # Hard cap chunk length placeholder (kept minimal, unchanged behavior)
                    if (chunk_index % max(1, max_chunk_chunks)) == 0 and len(active_pcm) > 0:
                        pass

                    if silence_counter >= silence_chunk_limit:
                        if len(active_pcm) >= int(MINCHUNK_DURATION_SECONDS * 8000 * 2):
                            chunk_obj = AudioChunk(
                                phrase_id=phrase_id,
                                chunk_index=chunk_index,
                                audio_bytes=bytes(active_pcm),
                                rms=rms,
                                timestamp=chunk_index * FRAME_DURATION_SECONDS,
                                duration=len(active_pcm) / (8000 * 2),
                                transcription="",
                                capture_state="listening",
                            )
                            await ready_chunks.put(chunk_obj)
                            await transcription_queue.put(chunk_obj)

                            if phrase_id not in detected_phrases:
                                detected_phrases[phrase_id] = PhraseObject(phrase_id=phrase_id, chunks=[])
                            detected_phrases[phrase_id].chunks.append(chunk_obj)

                            logger.info("AudioChunk completed at frame=%d phrase_id=%s", chunk_index, phrase_id)

                        active_pcm = bytearray()
                        silence_counter = 0
                        in_chunk = False

                else:
                    if rms < MIN_SPEECH_RMS_THRESHOLD:
                        phrase_silence_counter += 1
                        if phrase_silence_counter == done_speaking_chunk_limit:
                            if has_spoken:
                                logger.info("Detected end of phrase based on long silence.")
                                has_spoken = False
                                phrase_id = str(uuid.uuid4())

                chunk_index += 1

            else:
                continue

    player_task = asyncio.create_task(player_loop())
    receiver_task = asyncio.create_task(receiver_loop())

    try:
        await receiver_task
    finally:
        shutdown_event.set()
        if not player_task.done():
            player_task.cancel()
            try:
                await player_task
            except Exception:
                pass
        call_active.clear()
        call_ended.set()
