import os
import logging
import threading
import asyncio
import base64
import audioop
from typing import AsyncIterator, Dict

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from pydub import AudioSegment, effects

logger = logging.getLogger("elevenlabs_handler")
logger.setLevel(logging.INFO)
# Prevent messages from also being handled by the root logger (avoids duplicate output)
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
else:
    # Ensure any pre-existing handlers will also emit INFO and have a formatter
    for h in logger.handlers:
        h.setLevel(logging.INFO)
        if h.formatter is None:
            h.setFormatter(formatter)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("elevenlabs").setLevel(logging.WARNING)

load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
try:
    ELEVENLABS_SPEED = float(os.getenv("ELEVENLABS_SPEED", "0.90"))
except ValueError:
    ELEVENLABS_SPEED = 0.90

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def synthesize_speech_to_mp3(text: str, output_path: str) -> bool:
    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL,
            text=text,
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.55,
                similarity_boost=0.70,
                speed=ELEVENLABS_SPEED,
            ),
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        logger.info(f"✅ ElevenLabs MP3 written to {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ ElevenLabs synthesis failed: {e}")
        return False

def encode_mp3_to_ulaw_frames(mp3_path: str) -> list[str]:
    logger.info(f"Encoding MP3 to μ-law base64 frames: {mp3_path}")
    seg = AudioSegment.from_mp3(mp3_path).set_channels(1)
    try:
        seg = seg.low_pass_filter(3400).high_pass_filter(120)
    except Exception:
        pass
    seg = seg.set_frame_rate(8000).set_sample_width(2)
    seg = effects.normalize(seg, headroom=3.0)
    seg = effects.compress_dynamic_range(seg, threshold=-18.0, ratio=2.0, attack=5, release=50)
    seg = seg.fade_in(8).fade_out(8)
    pad = AudioSegment.silent(duration=20, frame_rate=8000)
    seg = pad + seg + pad
    frame_ms = 20
    remainder = len(seg) % frame_ms
    if remainder:
        seg += AudioSegment.silent(duration=(frame_ms - remainder), frame_rate=8000)
    pcm16 = seg.raw_data
    mulaw = audioop.lin2ulaw(pcm16, 2)
    frame_size = 160
    return [base64.b64encode(mulaw[i:i+frame_size]).decode("utf-8") for i in range(0, len(mulaw), frame_size)]

def generate_static_prompt_mp3s(output_dir: str = "app/audio_static") -> Dict[str, bool]:
    """
    Pre-generate static MP3 prompts used during calls:
      - greeting.mp3 (existing)
      - reminder.mp3 ("Hello? Are you still there?")
      - goodbye.mp3 ("Goodbye")

    Returns a dict mapping filename -> bool (synthesis success).
    """
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, bool] = {}

    greeting_path = os.path.join(output_dir, "greeting.mp3")
    reminder_path = os.path.join(output_dir, "reminder.mp3")
    goodbye_path = os.path.join(output_dir, "goodbye.mp3")

    greeting_text = "Hellooo! How can I help you today?"
    reminder_text = "Hello? Are you still there?"
    goodbye_text = "Goodbye"

    logger.info("generate_static_prompt_mp3s: starting generation (model=%s, speed=%s, voice=%s)", ELEVENLABS_MODEL, ELEVENLABS_SPEED, ELEVENLABS_VOICE_ID)
    
    results[os.path.basename(greeting_path)] = synthesize_speech_to_mp3(
        greeting_text, greeting_path
    )

    results[os.path.basename(reminder_path)] = synthesize_speech_to_mp3(
        reminder_text, reminder_path
    )

    results[os.path.basename(goodbye_path)] = synthesize_speech_to_mp3(
        goodbye_text, goodbye_path
    )

    return results

async def stream_tts_ulaw_frames(text: str) -> AsyncIterator[str]:
    """
    Yield base64 μ-law frames (160 bytes = 20ms @ 8 kHz) from ElevenLabs in real time.
    Head is preserved under pressure (drop-newest policy).
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=512)
    SENTINEL = object()

    def _enqueue_bytes_from_thread(data: bytes):
        try:
            # Preserve the head: if full, skip this burst (drop newest) to avoid chopping the beginning
            if q.full():
                return
            q.put_nowait(data)
        except Exception as e:
            # Avoid scheduling Queue.put_nowait directly; keep all operations inside this function
            logger.debug(f"enqueue skip/warn: {e}")

    def _producer():
        try:
            stream = client.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                model_id=ELEVENLABS_MODEL,
                text=text,
                output_format="ulaw_8000",
                voice_settings=VoiceSettings(
                    stability=0.55,
                    similarity_boost=0.70,
                    speed=ELEVENLABS_SPEED,
                ),
            )
            for chunk in stream:
                if chunk:
                    loop.call_soon_threadsafe(_enqueue_bytes_from_thread, chunk)
        except Exception as e:
            logger.error(f"❌ ElevenLabs streaming failed: {e}")
        finally:
            loop.call_soon_threadsafe(q.put_nowait, SENTINEL)

    threading.Thread(target=_producer, daemon=True).start()

    buf = bytearray()
    while True:
        item = await q.get()
        if item is SENTINEL:
            break
        buf.extend(item)
        while len(buf) >= 160:
            frame = bytes(buf[:160]); del buf[:160]
            yield base64.b64encode(frame).decode("utf-8")

    if buf:
        pad = 160 - len(buf)
        yield base64.b64encode(bytes(buf + b"\xff" * pad)).decode("utf-8")
