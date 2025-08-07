import os
import asyncio
import logging
import base64
import audioop
import json
from io import BytesIO
from quart import Quart, websocket, Response
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings
from pydub import AudioSegment
from hypercorn.asyncio import serve
from hypercorn.config import Config

# --- Environment ---
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
TEXT = "Hi Cody. This is a test stream from ElevenLabs using μ-law frames."

# --- Quart App ---
app = Quart(__name__)
logger = logging.getLogger("stream_test")
logging.basicConfig(level=logging.INFO)

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def synthesize_to_ulaw_frames(text: str) -> list[str]:
    logger.info("🎤 Synthesizing with ElevenLabs...")
    stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        text=text,
        voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
        output_format="mp3_44100_128"
    )
    mp3 = b"".join(chunk for chunk in stream)
    pcm = AudioSegment.from_file(BytesIO(mp3), format="mp3")
    pcm = pcm.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    raw = pcm.raw_data
    ulaw = audioop.lin2ulaw(raw, 2)

    frame_size = 160
    frames = [base64.b64encode(ulaw[i:i + frame_size]).decode()
              for i in range(0, len(ulaw), frame_size)]

    logger.info(f"✅ Encoded {len(frames)} μ-law frames.")
    return frames

@app.route("/voice", methods=["POST"])
async def voice_webhook():
    logger.info("📞 Received /voice webhook from Twilio.")
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://youngbullserver.com/ai_caller/stream" />
  </Connect>
</Response>"""
    return Response(twiml, status=200, content_type="text/xml")

@app.websocket("/stream")
async def stream():
    logger.info("🔌 WebSocket connection established.")
    dummy = base64.b64encode(b"\xff" * 160).decode()
    await websocket.send_json({
        "event": "media",
        "streamSid": "placeholder",
        "media": {"payload": dummy}
    })
    logger.info("🟡 Sent dummy frame to prevent hangup.")

    try:
        while True:
            msg = await websocket.receive()
            event = json.loads(msg)
            if event["event"] == "start":
                stream_sid = event["start"]["streamSid"]
                logger.info(f"📡 streamSid: {stream_sid}")
                break
    except Exception as e:
        logger.error(f"❌ Failed during handshake: {e}")
        return

    try:
        frames = synthesize_to_ulaw_frames(TEXT)
        logger.info(f"🎧 Streaming {len(frames)} audio frames...")
        for frame in frames:
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": frame}
            })
            await asyncio.sleep(0.02)
        logger.info("✅ Finished streaming audio.")
    except Exception as e:
        logger.error(f"❌ Streaming error: {e}")

async def main():
    config = Config()
    config.bind = ["0.0.0.0:5000"]
    config.use_reloader = False
    logger.info("🚀 Launching Hypercorn server on port 5000...")
    await serve(app, config)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Shutdown requested.")
