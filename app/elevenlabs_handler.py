import os
import logging
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import base64
import audioop
from pydub import AudioSegment

# --- Logging ---
logger = logging.getLogger("elevenlabs_handler")
logger.setLevel(logging.INFO)

# Suppress noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("elevenlabs").setLevel(logging.WARNING)

# --- Load environment ---
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# --- Init client ---
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def synthesize_speech_to_mp3(text: str, output_path: str) -> bool:
    """
    Converts text to speech and saves it as an MP3 to `output_path`.
    Returns True if successful, False otherwise.
    """
    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id="eleven_multilingual_v2",
            text=text,
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
            output_format="mp3_44100_128"
        )

        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        logger.info(f"✅ ElevenLabs MP3 written to {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ ElevenLabs synthesis failed: {e}")
        return False

def encode_mp3_to_ulaw_frames(mp3_path: str) -> list[str]:
    """Convert MP3 to μ-law 8kHz mono and return a list of base64-encoded 20ms frames."""
    logger.info(f"Encoding MP3 to μ-law base64 frames: {mp3_path}")
    wav = AudioSegment.from_mp3(mp3_path).set_channels(1).set_frame_rate(8000)
    pcm = wav.raw_data
    mulaw = audioop.lin2ulaw(pcm, wav.sample_width)
    frames = [mulaw[i:i+160] for i in range(0, len(mulaw), 160)]
    b64_frames = [base64.b64encode(frame).decode("utf-8") for frame in frames]
    logger.info(f"Encoded {len(b64_frames)} frames.")
    return b64_frames

def generate_initial_greeting_mp3(output_path: str = "app/audio_temp/greeting.mp3", overwrite: bool = False) -> bool:
    """
    Generate the greeting "Hello! How can I be of assistance?" and save to MP3.
    Skips generation if file exists unless overwrite=True.
    """
    greeting_text = "Hello! How can I be of assistance?"

    if os.path.exists(output_path) and not overwrite:
        logger.info(f"📁 Greeting MP3 already exists at {output_path}. Skipping regeneration.")
        return True

    logger.info(f"🎙️ Generating initial greeting MP3: {greeting_text}")
    return synthesize_speech_to_mp3(greeting_text, output_path)
