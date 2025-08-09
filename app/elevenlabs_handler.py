import os
import logging
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import base64
import audioop
from pydub import AudioSegment, effects  # normalize/compress

logger = logging.getLogger("elevenlabs_handler")
logger.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("elevenlabs").setLevel(logging.WARNING)

load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# Low-latency model; keep configurable via .env
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
try:
    ELEVENLABS_SPEED = float(os.getenv("ELEVENLABS_SPEED", "0.90"))  # slower = calmer
except ValueError:
    ELEVENLABS_SPEED = 0.90

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def synthesize_speech_to_mp3(text: str, output_path: str) -> bool:
    """
    ElevenLabs TTS → MP3 on disk.
    NOTE: 'speed' must be set inside voice_settings (not as a top-level arg).
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
                # use_speaker_boost left default (True)
            ),
            # apply_text_normalization left at service defaults; cannot be forced "on" for flash_v2_5
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        logger.info(
            "✅ ElevenLabs MP3 written to %s (model=%s, speed=%.2f)",
            output_path, ELEVENLABS_MODEL, ELEVENLABS_SPEED
        )
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
    greeting_text = "Hello. How can I help you today?"
    logger.info(f"🎙️ Generating initial greeting MP3 (model={ELEVENLABS_MODEL}, speed={ELEVENLABS_SPEED}): {greeting_text}")
    return synthesize_speech_to_mp3(greeting_text, output_path)
