import os
import logging
import asyncio
import base64
import audioop
import time
import json
from collections import deque
from quart import Quart, Response, websocket
from dotenv import load_dotenv
from asyncio import Queue
import uuid

from app.data_types import AudioChunk, PhraseObject
from app.queues import llm_playback_queue, tts_requests_queue
from app.elevenlabs_handler import encode_mp3_to_ulaw_frames, stream_tts_ulaw_frames

# Global Variables and Events
call_ended = asyncio.Event()
ready_chunks: Queue = Queue()          # Queue of AudioChunks to persist at shutdown
transcription_queue: Queue = Queue()   # Queue of AudioChunks for Whisper

# Load environment variables
load_dotenv()
ELEVEN_STREAMING = os.getenv("ELEVEN_STREAMING", "false").lower() == "true"
STORE_ALL_RESPONSE_AUDIO = os.getenv("STORE_ALL_RESPONSE_AUDIO", "false").lower() == "true"

# Configure logging
logger = logging.getLogger("twilio_handler")
logger.setLevel(logging.INFO if os.getenv("LOGGING_ENABLED", "false").lower() == "true" else logging.CRITICAL)

# Create Quart app
app = Quart(__name__)

# Mapping of phrase_id → PhraseObject (shared with whisper_handler)
detected_phrases: dict[str, PhraseObject] = {}

async def _stream_ulaw_frames(stream_sid: str, frames: list[str]):
    base = time.monotonic()
    interval = 0.02
    for i, frame in enumerate(frames, 1):
        await websocket.send_json({"event": "media", "streamSid": stream_sid, "media": {"payload": frame}})
        target = base + i * interval
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

async def _stream_ulaw_frame_iter(stream_sid: str, aiter):
    base = time.monotonic()
    interval = 0.02
    i = 0
    async for frame in aiter:
        i += 1
        await websocket.send_json({"event": "media", "streamSid": stream_sid, "media": {"payload": frame}})
        target = base + i * interval
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

async def write_all_chunks_to_disk(queue: asyncio.Queue, output_dir: str):
    import wave, os
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

def ulaw_to_pcm(ulaw_bytes: bytes) -> bytes:
    return audioop.ulaw2lin(ulaw_bytes, 2)

def calculate_rms(chunk: bytes) -> float:
    try:
        return audioop.rms(audioop.ulaw2lin(chunk, 2), 2)
    except Exception:
        return 0.0

async def _send_dummy_frame(stream_sid: str):
    dummy_frame = base64.b64encode(b"\xff" * 160).decode()
    await websocket.send_json({"event": "media", "streamSid": stream_sid, "media": {"payload": dummy_frame}})

async def _send_clear(stream_sid: str):
    try:
        await websocket.send_json({"event": "clear", "streamSid": stream_sid})
    except Exception as e:
        logger.error(f"Failed sending clear: {e}")

async def _stream_mp3_path(stream_sid: str, mp3_path: str):
    frames = encode_mp3_to_ulaw_frames(mp3_path)
    await _stream_ulaw_frames(stream_sid, frames)

@app.websocket("/stream")
async def media_stream():
    logger.info("WebSocket connection established with Twilio.")
    # VAD / chunking parameters
    MIN_SPEECH_RMS_THRESHOLD = 750.0
    CHUNK_SILENCE_DURATION_SECONDS = 0.3
    DONE_SPEAKING_SILENCE_DURATION_SECONDS = 0.5
    MINCHUNK_DURATION_SECONDS = 0.5
    MAXCHUNK_DURATION_SECONDS = 10.0
    LEAD_IN_DURATION_SECONDS = 0.2

    FRAME_DURATION_SECONDS = 0.02
    silence_chunk_limit = int(CHUNK_SILENCE_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    done_speaking_chunk_limit = int(DONE_SPEAKING_SILENCE_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    min_chunk_len_bytes = int(MINCHUNK_DURATION_SECONDS * 8000 * 2)
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

    # NEW: gate VAD during assistant playback
    listening = True

    try:
        while True:
            msg = await websocket.receive()
            if msg is None:
                break

            event = json.loads(msg)
            event_type = event.get("event")

            if event_type == "start":
                stream_sid = event.get("start", {}).get("streamSid", "placeholder")
                await _send_dummy_frame(stream_sid)

                # Optional greeting playback (file path)
                try:
                    greeting_path = "app/audio_static/greeting.mp3"
                    if os.path.exists(greeting_path):
                        listening = False
                        await _stream_mp3_path(stream_sid, greeting_path)
                        await _send_clear(stream_sid)
                        listening = True
                except Exception as e:
                    logger.error(f"Failed to stream greeting: {e}")

            elif event_type == "stop":
                call_ended.set()
                break

            elif event_type == "media":
                # 1) Handle inbound caller audio (VAD / chunking)
                payload = event["media"].get("payload")
                if payload:
                    ulaw = base64.b64decode(payload)

                    if listening:
                        rms = calculate_rms(ulaw)
                        pcm = ulaw_to_pcm(ulaw)
                        pre_chunk_pcm_buffer.extend(pcm)

                        if not in_chunk and rms >= MIN_SPEECH_RMS_THRESHOLD:
                            if has_spoken and phrase_silence_counter >= done_speaking_chunk_limit:
                                phrase_id = str(uuid.uuid4())
                            has_spoken = True
                            in_chunk = True
                            silence_counter = 0
                            phrase_silence_counter = 0
                            active_pcm.extend(pre_chunk_pcm_buffer)
                            pre_chunk_pcm_buffer.clear()

                        if in_chunk:
                            active_pcm.extend(pcm)

                            if rms < MIN_SPEECH_RMS_THRESHOLD:
                                silence_counter += 1
                                phrase_silence_counter += 1
                            else:
                                silence_counter = 0
                                phrase_silence_counter = 0

                            if silence_counter >= silence_chunk_limit:
                                if len(active_pcm) >= min_chunk_len_bytes:
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

                                    # Only store raw audio if explicitly enabled
                                    if STORE_ALL_RESPONSE_AUDIO:
                                        await ready_chunks.put(chunk_obj)

                                    await transcription_queue.put(chunk_obj)

                                    if phrase_id not in detected_phrases:
                                        detected_phrases[phrase_id] = PhraseObject(phrase_id=phrase_id, chunks=[])
                                    detected_phrases[phrase_id].chunks.append(chunk_obj)

                                    active_pcm = bytearray()
                                    silence_counter = 0
                                    in_chunk = False

                        elif rms < MIN_SPEECH_RMS_THRESHOLD:
                            phrase_silence_counter += 1
                            if phrase_silence_counter == done_speaking_chunk_limit:
                                if has_spoken:
                                    has_spoken = False
                                    phrase_id = str(uuid.uuid4())

                    # advance frame index regardless (keeps timestamps monotonic)
                    chunk_index += 1

                # 2) After each inbound frame, drain any assistant audio
                if stream_sid:
                    # Streaming path (B)
                    while ELEVEN_STREAMING and not tts_requests_queue.empty():
                        text = await tts_requests_queue.get()
                        listening = False
                        await _stream_ulaw_frame_iter(stream_sid, stream_tts_ulaw_frames(text))
                        await _send_clear(stream_sid)
                        listening = True

                    # Fallback/file path
                    while not ELEVEN_STREAMING and not llm_playback_queue.empty():
                        mp3_path = await llm_playback_queue.get()
                        listening = False
                        await _stream_mp3_path(stream_sid, mp3_path)
                        await _send_clear(stream_sid)
                        listening = True

            # Opportunistic drains outside media events
            if stream_sid:
                while ELEVEN_STREAMING and not tts_requests_queue.empty():
                    text = await tts_requests_queue.get()
                    listening = False
                    await _stream_ulaw_frame_iter(stream_sid, stream_tts_ulaw_frames(text))
                    await _send_clear(stream_sid)
                    listening = True

                while not ELEVEN_STREAMING and not llm_playback_queue.empty():
                    mp3_path = await llm_playback_queue.get()
                    listening = False
                    await _stream_mp3_path(stream_sid, mp3_path)
                    await _send_clear(stream_sid)
                    listening = True

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    finally:
        call_ended.set()
