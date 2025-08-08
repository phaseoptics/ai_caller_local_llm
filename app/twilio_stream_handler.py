import os
import logging
import asyncio
import base64
import audioop
import wave
import json
from io import BytesIO
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from quart import Quart, request, Response, websocket
from dotenv import load_dotenv
from asyncio import Queue
import uuid

from app.conversation_manager import handle_phrase
from app.data_types import AudioChunk, PhraseObject
from app.queues import llm_playback_queue
from app.elevenlabs_handler import encode_mp3_to_ulaw_frames

# Global Variables and Events
call_ended = asyncio.Event()
ready_chunks: Queue = Queue()   # Queue of AudioChunks to persist at shutdown
transcription_queue: Queue = Queue()  # Queue of AudioChunks for Whisper

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("twilio_handler")
logger.setLevel(logging.INFO if os.getenv("LOGGING_ENABLED", "false").lower() == "true" else logging.CRITICAL)

# Create Quart app
app = Quart(__name__)

# Mapping of phrase_id → PhraseObject
detected_phrases: dict[str, PhraseObject] = {}

async def write_all_chunks_to_disk(queue: asyncio.Queue, output_dir: str):
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
        logger.info(f"Wrote chunk to {out_path}")
        idx += 1
    logger.info(f"Finished writing {idx} chunks to disk.")

# --- Webhook Route ---
@app.route("/voice", methods=["POST"])
async def voice_webhook():
    logger.info("Received /voice webhook from Twilio.")
    twiml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Response>
  <Connect>
    <Stream url=\"wss://youngbullserver.com/ai_caller/stream\" />
  </Connect>
</Response>"""
    return Response(twiml, mimetype="text/xml")

# --- Audio Utilities ---
def ulaw_to_pcm(ulaw_bytes: bytes) -> bytes:
    return audioop.ulaw2lin(ulaw_bytes, 2)

def calculate_rms(chunk: bytes) -> float:
    try:
        return audioop.rms(audioop.ulaw2lin(chunk, 2), 2)
    except Exception as e:
        logger.error(f"RMS calculation error: {e}")
        return 0.0

async def _send_dummy_frame(stream_sid: str):
    dummy_frame = base64.b64encode(b"\xff" * 160).decode()
    await websocket.send_json({
        "event": "media",
        "streamSid": stream_sid,
        "media": {"payload": dummy_frame}
    })
    logger.info("🟡 Sent dummy frame to hold stream open.")

async def _send_clear(stream_sid: str):
    try:
        await websocket.send_json({"event": "clear", "streamSid": stream_sid})
        logger.info("🧹 Sent clear to flush caller buffered audio.")
    except Exception as e:
        logger.error(f"Failed sending clear: {e}")

async def _stream_mp3_path(stream_sid: str, mp3_path: str):
    """Blocking send inside media_stream coroutine. Reads MP3, encodes to μ-law 8kHz, streams at 20ms pace."""
    frames = encode_mp3_to_ulaw_frames(mp3_path)
    logger.info(f"📢 Streaming {os.path.basename(mp3_path)}: {len(frames)} frames")
    for frame in frames:
        await websocket.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": frame}
        })
        await asyncio.sleep(0.02)  # 20ms pacing
    logger.info(f"✅ Finished streaming {os.path.basename(mp3_path)}")

# --- WebSocket Route ---
@app.websocket("/stream")
async def media_stream():
    logger.info("WebSocket connection established with Twilio.")
    # Parameters
    MIN_SPEECH_RMS_THRESHOLD = 600.0
    CHUNK_SILENCE_DURATION_SECONDS = 0.3
    DONE_SPEAKING_SILENCE_DURATION_SECONDS = 0.6
    MINCHUNK_DURATION_SECONDS = 0.4
    MAXCHUNK_DURATION_SECONDS = 10.0
    LEAD_IN_DURATION_SECONDS = 0.2

    # Derived
    FRAME_DURATION_SECONDS = 0.02
    silence_chunk_limit = int(CHUNK_SILENCE_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    done_speaking_chunk_limit = int(DONE_SPEAKING_SILENCE_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    min_chunk_chunks = int(MINCHUNK_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    max_chunk_chunks = int(MAXCHUNK_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    lead_in_bytes = int(8000 * 2 * LEAD_IN_DURATION_SECONDS)

    phrase_id = str(uuid.uuid4())
    chunk_index = 0
    active_pcm = bytearray()
    pre_chunk_pcm_buffer = deque(maxlen=lead_in_bytes)
    silence_counter = 0
    phrase_silence_counter = 0
    in_chunk = False
    has_spoken = False
    stream_sid = None

    try:
        while True:
            msg = await websocket.receive()
            if msg is None:
                logger.warning("WebSocket closed by Twilio.")
                break

            event = json.loads(msg)
            event_type = event.get("event")

            if event_type == "start":
                logger.info("Twilio media stream started.")
                stream_sid = event.get("start", {}).get("streamSid", "placeholder")

                # Prime immediately
                await _send_dummy_frame(stream_sid)

                # Play greeting then clear
                try:
                    greeting_path = "app/audio_static/greeting.mp3"
                    if os.path.exists(greeting_path):
                        await _stream_mp3_path(stream_sid, greeting_path)
                        await _send_clear(stream_sid)
                    else:
                        logger.warning("No greeting.mp3 found, skipping greeting playback.")
                except Exception as e:
                    logger.error(f"Failed to stream greeting: {e}")

            elif event_type == "stop":
                logger.info("Twilio media stream stopped.")
                call_ended.set()
                break

            elif event_type == "media":
                # Receive caller audio
                payload = event["media"].get("payload")
                if not payload:
                    # Still use this opportunity to drain playback queue if any
                    if stream_sid:
                        while not llm_playback_queue.empty():
                            mp3_path = await llm_playback_queue.get()
                            await _stream_mp3_path(stream_sid, mp3_path)
                            await _send_clear(stream_sid)
                    continue

                chunk = base64.b64decode(payload)
                rms = calculate_rms(chunk)
                pcm = ulaw_to_pcm(chunk)

                # Always buffer a small lead-in ring to avoid clipping
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
                    logger.info(f"AudioChunk formation initiated in frame #{chunk_index} for phrase_id={phrase_id}")

                if in_chunk:
                    active_pcm.extend(pcm)

                    if rms < MIN_SPEECH_RMS_THRESHOLD:
                        silence_counter += 1
                        phrase_silence_counter += 1
                    else:
                        silence_counter = 0
                        phrase_silence_counter = 0

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
                                capture_state="listening"
                            )
                            await ready_chunks.put(chunk_obj)
                            await transcription_queue.put(chunk_obj)

                            if phrase_id not in detected_phrases:
                                detected_phrases[phrase_id] = PhraseObject(phrase_id=phrase_id, chunks=[])
                            detected_phrases[phrase_id].chunks.append(chunk_obj)

                            logger.info(f"AudioChunk completed at frame #{chunk_index} for phrase_id={phrase_id}")
                            active_pcm = bytearray()
                            silence_counter = 0
                            in_chunk = False

                elif rms < MIN_SPEECH_RMS_THRESHOLD:
                    phrase_silence_counter += 1
                    if phrase_silence_counter == done_speaking_chunk_limit:
                        if has_spoken:
                            logger.info("Detected end of phrase based on long silence.")
                            has_spoken = False
                            phrase_id = str(uuid.uuid4())

                chunk_index += 1

                # Drain any queued assistant audio right after handling this inbound frame
                if stream_sid:
                    while not llm_playback_queue.empty():
                        mp3_path = await llm_playback_queue.get()
                        await _stream_mp3_path(stream_sid, mp3_path)
                        await _send_clear(stream_sid)

            # Other Twilio events are ignored, but we still opportunistically drain queue
            if stream_sid:
                while not llm_playback_queue.empty():
                    mp3_path = await llm_playback_queue.get()
                    await _stream_mp3_path(stream_sid, mp3_path)
                    await _send_clear(stream_sid)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    finally:
        # make sure the main loop cam exit even if Twilio never sent stop
        call_ended.set()
