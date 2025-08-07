import os
import logging
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("elevenlabs_tester")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("elevenlabs").setLevel(logging.WARNING)

# --- Load .env ---
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# --- Initialize client ---
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def synthesize_speech(text: str, output_path: str, return_bytes: bool = False) -> str | bytes | None:
    """
    Convert text to speech using ElevenLabs and either save to file or return bytes.
    """
    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id="eleven_multilingual_v2",
            text=text,
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
            output_format="mp3_44100_128"
        )

        if return_bytes:
            audio_bytes = b''.join(chunk for chunk in audio_stream)
            logger.info("✅ Speech generated and returned as bytes")
            return audio_bytes

        # Save to file
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        logger.info(f"✅ Speech saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"❌ Error in synthesize_speech: {e}")
        return None

# --- Test block ---
if __name__ == "__main__":
    SAMPLE_TEXT = "Hi Cody. This is a test of the ElevenLabs voice system using the new client."
    OUTPUT_FILE = "test_output.mp3"
    synthesize_speech(SAMPLE_TEXT, OUTPUT_FILE)
