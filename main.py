import logging
import sys
import asyncio
import os

from app.twilio_stream_handler import app, call_ended, ready_chunks, write_all_chunks_to_disk
from app.whisper_handler import whisper_transcription_loop
from app.elevenlabs_handler import generate_initial_greeting_mp3
from hypercorn.asyncio import serve
from hypercorn.config import Config

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("hypercorn.error").propagate = False
logging.getLogger("hypercorn.access").propagate = False

logger = logging.getLogger("main")

async def run_server():
    """
    Starts Hypercorn for the Quart app and the Whisper background task.
    Waits for the call_ended event, but uses a timeout as a safety net.
    On exit, cancels tasks and, if enabled, writes buffered audio chunks to disk.
    """
    # New flag: only persist raw caller chunks if explicitly enabled
    STORE_ALL_RESPONSE_AUDIO = os.getenv("STORE_ALL_RESPONSE_AUDIO", "false").lower() == "true"

    # Prepare Hypercorn
    config = Config()
    config.bind = ["0.0.0.0:5000"]
    config.use_reloader = False
    config.loglevel = "WARNING"

    print("Starting Quart app via Hypercorn on port 5000...")
    sys.stdout.flush()
    server_task = asyncio.create_task(serve(app, config))

    print("Starting Whisper transcription handler...")
    sys.stdout.flush()
    whisper_task = asyncio.create_task(whisper_transcription_loop())

    print("Waiting for Twilio call to complete...")
    sys.stdout.flush()
    TIMEOUT_SECONDS = 120

    try:
        await asyncio.wait_for(call_ended.wait(), timeout=TIMEOUT_SECONDS)
        print("Call end signal received.")
    except asyncio.TimeoutError:
        print(f"No call end signal within {TIMEOUT_SECONDS} seconds. Forcing shutdown...")

    # Begin shutdown sequence
    print("Shutting down tasks...")
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

    # Conditionally persist buffered chunks
    if STORE_ALL_RESPONSE_AUDIO:
        BUFFER_DIR = "app/audio_temp"
        print(f"Writing {ready_chunks.qsize()} ready AudioChunks to disk in {BUFFER_DIR}...")
        await write_all_chunks_to_disk(ready_chunks, BUFFER_DIR)
        print("All chunks written.")
    else:
        # Drain queue without saving (free memory)
        drained = 0
        while not ready_chunks.empty():
            _ = await ready_chunks.get()
            drained += 1
        print(f"Skipped writing caller chunks (STORE_ALL_RESPONSE_AUDIO=false). Drained {drained} buffered chunks.")

    print("Exiting.")

if __name__ == "__main__":
    try:
        generate_initial_greeting_mp3("app/audio_static/greeting.mp3")
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("Shutdown requested by user. Exiting cleanly...")
