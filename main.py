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
# Remove any preexisting handlers so we control formatting
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Keep OpenAI logs quieter unless warning
logging.getLogger("openai").setLevel(logging.WARNING)

# Avoid duplicate Hypercorn logs
logging.getLogger("hypercorn.error").propagate = False
logging.getLogger("hypercorn.access").propagate = False

logger = logging.getLogger("main")


async def run_server():
    """
    Starts Hypercorn for the Quart app and the Whisper background task.
    Waits for the call_ended event, but uses a timeout as a safety net.
    On exit, cancels tasks and writes any buffered audio chunks to disk.
    """
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

    # Wait for call end with a timeout so we never hang forever
    # If Twilio fails to send stop, twilio_stream_handler sets call_ended in finally
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

    # Await server task
    try:
        await server_task
    except asyncio.CancelledError:
        print("Server shut down cleanly.")

    # Await whisper task
    try:
        await whisper_task
    except asyncio.CancelledError:
        print("Whisper handler shut down cleanly.")

    # Persist any buffered chunks
    BUFFER_DIR = "app/audio_temp"
    print(f"Writing {ready_chunks.qsize()} ready AudioChunks to disk in {BUFFER_DIR}...")
    await write_all_chunks_to_disk(ready_chunks, BUFFER_DIR)
    print("All chunks written. Exiting.")


if __name__ == "__main__":
    try:
        # Ensure greeting exists each run so voice and text changes are reflected
        generate_initial_greeting_mp3("app/audio_static/greeting.mp3")

        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("Shutdown requested by user. Exiting cleanly...")
