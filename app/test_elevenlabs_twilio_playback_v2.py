import os
import asyncio
import logging
from quart import Quart, Response, send_from_directory
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings
from hypercorn.asyncio import serve
from hypercorn.config import Config

# --- Load Environment ---
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
TEXT = "Hi Cody. This is a test using the audio_temp directory with proper routing."
MODEL_ID = "eleven_multilingual_v2"
PORT = 5000

# --- Paths ---
AUDIO_TEMP_DIR = "app/audio_temp"
MP3_FILENAME = "tts_response.mp3"
MP3_PATH = os.path.join(AUDIO_TEMP_DIR, MP3_FILENAME)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_audio_temp_playback")

# --- Quart App ---
app = Quart(__name__)
os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)

# --- ElevenLabs Client ---
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def synthesize_to_mp3_file(text: str, output_path: str):
    logger.info("🎤 Synthesizing speech with ElevenLabs...")
    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            text=text,
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
            output_format="mp3_44100_128"
        )
        mp3_bytes = b"".join(chunk for chunk in audio_stream)
        logger.info(f"✅ ElevenLabs synthesis complete. Length: {len(mp3_bytes)} bytes")

        with open(output_path, "wb") as f:
            f.write(mp3_bytes)
        logger.info(f"✅ MP3 saved to: {output_path}")

    except Exception as e:
        logger.error(f"❌ ElevenLabs synthesis failed: {e}")
        raise

# --- Serve MP3 from /audio_temp ---
@app.route("/audio_temp/<path:filename>")
async def audio_temp(filename):
    return await send_from_directory(AUDIO_TEMP_DIR, filename)

# --- Webhook for Twilio <Play> ---
@app.route("/voice", methods=["POST"])
async def voice_webhook():
    logger.info("📞 Received /voice webhook from Twilio.")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Play>https://youngbullserver.com/audio_temp/{MP3_FILENAME}</Play>
</Response>"""
    return Response(twiml, status=200, content_type="text/xml")

# --- App Runner ---
async def main():
    synthesize_to_mp3_file(TEXT, MP3_PATH)

    config = Config()
    config.bind = [f"0.0.0.0:{PORT}"]
    config.use_reloader = False

    logger.info(f"🚀 Starting test server on port {PORT}...")
    await serve(app, config)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Shutdown requested.")
