# app/elevenlabs_handler.py
import os
import logging
import threading
import asyncio
import base64
import audioop
from typing import AsyncIterator

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from pydub import AudioSegment, effects  # normalize/compress

# --- Logging ---
logger = logging.getLogger("elevenlabs_handler")
logger.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("elevenlabs").setLevel(logging.WARNING)

# --- Env ---
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
try:
    ELEVENLABS_SPEED = float(os.getenv("ELEVENLABS_SPEED", "0.90"))  # calmer pacing
except ValueError:
    ELEVENLABS_SPEED = 0.90

# --- Client ---
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ================================
# A) File-based synthesis (fallback)
# ================================
def synthesize_speech_to_mp3(text: str, output_path: str) -> bool:
    """
    ElevenLabs TTS → MP3 on disk.
    """
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
    """
    MP3 → (filter, normalize, compress, fade, pad) → 8 kHz mono → μ-law → base64 20 ms frames.
    """
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

    frame_size = 160  # 20 ms @ 8 kHz μ-law
    return [
        base64.b64encode(mulaw[i:i+frame_size]).decode("utf-8")
        for i in range(0, len(mulaw), frame_size)
    ]

def generate_initial_greeting_mp3(output_path: str = "app/audio_static/greeting.mp3") -> bool:
    """
    Generate/refresh the greeting used at call start (file-based path).
    """
    greeting_text = "Hello. How can I help you today?"
    logger.info(
        f"🎙️ Generating initial greeting MP3 (model={ELEVENLABS_MODEL}, speed={ELEVENLABS_SPEED}): {greeting_text}"
    )
    return synthesize_speech_to_mp3(greeting_text, output_path)

# =========================================
# B) True streaming (μ-law 8 kHz, no files)
# =========================================
async def stream_tts_ulaw_frames(text: str) -> AsyncIterator[str]:
    """
    Yield base64 μ-law frames (160 bytes = 20ms @ 8 kHz) from ElevenLabs in real time.
    """
    loop = asyncio.get_running_loop()

    # Larger burst tolerance; we'll still enforce bounded memory.
    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=512)
    SENTINEL = object()

    def _enqueue_bytes_from_thread(data: bytes):
        """
        Runs on the event loop thread (scheduled via call_soon_threadsafe).
        If q is full, drop the oldest item to keep latency low, then enqueue new data.
        """
        try:
            if q.full():
                try:
                    _ = q.get_nowait()  # drop oldest to avoid stall
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(data)
        except Exception as e:
            # Swallow any rare race conditions; better to skip a packet than crash.
            logger.warning(f"⚠️ enqueue warning: {e}")

    def _producer():
        try:
            stream = client.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                model_id=ELEVENLABS_MODEL,
                text=text,
                output_format="ulaw_8000",  # telephony-ready μ-law @ 8 kHz
                voice_settings=VoiceSettings(
                    stability=0.55,
                    similarity_boost=0.70,
                    speed=ELEVENLABS_SPEED,
                ),
            )
            for chunk in stream:
                if chunk:
                    # hand off to event loop thread safely
                    loop.call_soon_threadsafe(_enqueue_bytes_from_thread, chunk)
        except Exception as e:
            logger.error(f"❌ ElevenLabs streaming failed: {e}")
        finally:
            loop.call_soon_threadsafe(q.put_nowait, SENTINEL)

    threading.Thread(target=_producer, daemon=True).start()

    # Frame to exact 160-byte packets and yield as base64
    buf = bytearray()
    while True:
        item = await q.get()
        if item is SENTINEL:
            break
        buf.extend(item)
        while len(buf) >= 160:
            frame = bytes(buf[:160]); del buf[:160]
            yield base64.b64encode(frame).decode("utf-8")

    # tail: pad to a full frame so Twilio doesn't click on shutdown
    if buf:
        pad = 160 - len(buf)
        yield base64.b64encode(bytes(buf + b"\xff" * pad)).decode("utf-8")
