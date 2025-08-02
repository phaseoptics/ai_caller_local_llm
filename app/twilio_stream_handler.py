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

# Global Variables and Events

call_ended = asyncio.Event()
ready_chunks: Queue = Queue() # Queue of AudioChunks
transcription_queue : Queue = Queue() # Another Queue of AudioChunks

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
    # Drain the queue completely
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

# --- WebSocket Route ---
@app.websocket("/stream")
async def media_stream():
    logger.info("WebSocket connection established with Twilio.")
    # Parameters
    MIN_SPEECH_RMS_THRESHOLD = 400.0
    CHUNK_SILENCE_DURATION_SECONDS = 0.3 #0.5
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

            elif event_type == "stop":
                logger.info("Twilio media stream stopped.")
                call_ended.set()
                break

            elif event_type == "media":
                payload = event["media"].get("payload")
                if not payload:
                    continue

                chunk = base64.b64decode(payload)
                rms = calculate_rms(chunk)
                pcm = ulaw_to_pcm(chunk)

                if rms > 50:
                    logger.info(f"Frame #{chunk_index}: RMS={rms:.2f}")

                # Always buffer to pre-chunk ring buffer
                pre_chunk_pcm_buffer.extend(pcm)

                if not in_chunk and rms >= MIN_SPEECH_RMS_THRESHOLD:
                    if has_spoken and phrase_silence_counter >= done_speaking_chunk_limit:
                        logger.info(f"Long silence detected. Generating new phrase ID.")
                        phrase_id = str(uuid.uuid4())
                    has_spoken = True
                    in_chunk = True
                    silence_counter = 0
                    phrase_silence_counter = 0
                    active_pcm.extend(pre_chunk_pcm_buffer) # Adds a bit of lead in to stop clipping edges to imrpove donwstream STT.
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
                            chunk = AudioChunk(
                                phrase_id=phrase_id,
                                chunk_index=chunk_index,
                                audio_bytes=bytes(active_pcm),
                                rms=rms,
                                timestamp=chunk_index * FRAME_DURATION_SECONDS,
                                duration=len(active_pcm) / (8000 * 2),
                                transcription="",
                                capture_state="listening"
                            )

                            await ready_chunks.put(chunk)
                            await transcription_queue.put(chunk)

                            # Add chunk to PhraseObject
                            if phrase_id not in detected_phrases:
                                detected_phrases[phrase_id] = PhraseObject(phrase_id=phrase_id, chunks=[])
                            detected_phrases[phrase_id].chunks.append(chunk)

                            logger.info(f"Created AudioChunk #{chunk_index} with phrase_id={phrase_id} and added to queues.")
                            
                            # Reset
                            active_pcm = bytearray()
                            silence_counter = 0
                            in_chunk = False

                # Check for end of phrase outside of speech
                elif rms < MIN_SPEECH_RMS_THRESHOLD:
                    phrase_silence_counter += 1
                    if phrase_silence_counter == done_speaking_chunk_limit:
                        if has_spoken:
                            logger.info("Detected end of phrase based on long period of silence.")
                            has_spoken = False
                            phrase_id = str(uuid.uuid4())

                chunk_index += 1

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
