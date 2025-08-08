import logging
import sys
import asyncio
import os
from app.twilio_stream_handler import app, call_ended, ready_chunks, write_all_chunks_to_disk
from app.whisper_handler import whisper_transcription_loop
from app.elevenlabs_handler import generate_initial_greeting_mp3
from hypercorn.asyncio import serve
from hypercorn.config import Config

# --- Logging Setup (force early and only once) ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Only escalate OpenAI message if WARNING 
logging.getLogger("openai").setLevel(logging.WARNING)

# Silence Hypercorn's default logging duplication
logging.getLogger("hypercorn.error").propagate = False
logging.getLogger("hypercorn.access").propagate = False
logger = logging.getLogger("main")

# --- Async Entry Point ---
async def run_server():
    config = Config()
    config.bind = ["0.0.0.0:5000"]
    config.use_reloader = False
    config.loglevel = "WARNING"  # Suppress Hypercorn info messages

    print("Starting Quart app via Hypercorn on port 5000...")
    sys.stdout.flush()

    server_task = asyncio.create_task(serve(app, config))

    print("Starting Whisper transcription handler...")
    sys.stdout.flush()
    whisper_task = asyncio.create_task(whisper_transcription_loop())

    print("Waiting for Twilio call to complete...")
    sys.stdout.flush()
    await call_ended.wait()

    print("Call ended. Proceeding to shutdown and save audio chunks.")
    server_task.cancel()
    whisper_task.cancel()

    try:
        await server_task
    except asyncio.CancelledError:
        print("Server shut down cleanly.")

    try:
        await whisper_task
    except asyncio.CancelledError:
        print("Whisper handler shut down cleanly.")

    BUFFER_DIR = 'app/audio_temp'
    print(f"Writing {ready_chunks.qsize()} ready AudioChunks to disk in {BUFFER_DIR}...")
    await write_all_chunks_to_disk(ready_chunks, BUFFER_DIR)
    print("All chunks written. Exiting.")

# --- Startup Hook ---
if __name__ == "__main__":
    try:
        # Generate ElevenLabs greeting MP3 (if not already cached)
        generate_initial_greeting_mp3("app/audio_temp/greeting.mp3")

        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("Shutdown requested by user (Ctrl+C). Exiting cleanly...")
