import logging
import sys
import asyncio
import os
import time

from app.twilio_stream_handler import app, call_ended, ready_chunks, write_all_chunks_to_disk, get_last_speech_time
from app.whisper_handler import whisper_transcription_loop
from app.elevenlabs_handler import generate_static_prompt_mp3s
from app.queues import llm_playback_queue, get_playback_pause_since_reset
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
    Waits for call end OR for the caller to be silent for MAX_SILENCE_SECONDS.
    On exit, cancels tasks and, if enabled, writes buffered audio chunks to disk.
    """
    # Only persist raw caller chunks if explicitly enabled
    STORE_ALL_RESPONSE_AUDIO = os.getenv("STORE_ALL_RESPONSE_AUDIO", "false").lower() == "true"

    # Max allowed caller silence (seconds)
    try:
        MAX_SILENCE_SECONDS = float(os.getenv("MAX_SILENCE_SECONDS", "30"))
    except ValueError:
        MAX_SILENCE_SECONDS = 60.0

    # Prepare Hypercorn
    config = Config()
    config.bind = ["0.0.0.0:5000"]
    config.use_reloader = False
    config.loglevel = "WARNING"

    logger.info("Starting Quart app via Hypercorn on port 5000...")
    server_task = asyncio.create_task(serve(app, config))

    logger.info("Starting Whisper transcription handler...")
    whisper_task = asyncio.create_task(whisper_transcription_loop())

    logger.info("Waiting for Twilio call to complete or silence timeout...")

    # Watchdog loop: exit if call ends OR caller has been silent for MAX_SILENCE_SECONDS
    try:
        poll_interval = 0.5  # seconds
        # Track last reminder effective-silence so we can repeat reminders every REMINDER_SECONDS
        last_reminder_effective_silence = 0.0
        goodbye_enqueued = False
        REMINDER_SECONDS = 10.0

        while True:
            if call_ended.is_set():
                logger.info("Call end signal received.")
                break

            raw_silent_for = time.monotonic() - get_last_speech_time()
            pause = get_playback_pause_since_reset()
            # Effective silent time excludes time when assistant was playing
            silent_for = max(0.0, raw_silent_for - pause)

            # Reset reminder timer if caller just spoke (raw_silent_for small)
            if raw_silent_for < 0.25:
                last_reminder_effective_silence = 0.0

            # If caller silent for REMINDER_SECONDS, play a polite reminder (repeatable)
            if silent_for - last_reminder_effective_silence >= REMINDER_SECONDS:
                reminder_path = "app/audio_static/reminder.mp3"
                if os.path.exists(reminder_path):
                    logger.info("Silent for %.1fs: enqueueing reminder.", silent_for)
                    await llm_playback_queue.put(reminder_path)
                    # record the effective_silence at which we played the reminder
                    last_reminder_effective_silence = silent_for

            # At MAX_SILENCE_SECONDS, enqueue goodbye once and then trigger clean shutdown (same behavior as before)
            if not goodbye_enqueued and silent_for >= MAX_SILENCE_SECONDS:
                goodbye_path = "app/audio_static/goodbye.mp3"
                if os.path.exists(goodbye_path):
                    logger.info("Silent for %.1fs: enqueueing goodbye and waiting for playback to finish...", silent_for)
                    await llm_playback_queue.put(goodbye_path)
                    # Determine goodbye.mp3 duration and wait exactly that long (plus small margin)
                    try:
                        from app.queues import wait_for_playback_completion
                        from pydub import AudioSegment
                        seg = AudioSegment.from_mp3(goodbye_path)
                        duration_s = len(seg) / 1000.0
                        margin = 0.5
                        ok = await wait_for_playback_completion(timeout=duration_s + margin)
                        if not ok:
                            logger.warning("Goodbye playback did not complete within expected duration; proceeding with shutdown.")
                    except Exception:
                        logger.exception("Error while waiting for goodbye playback duration; falling back to default wait")
                        try:
                            from app.queues import wait_for_playback_completion
                            ok = await wait_for_playback_completion(timeout=10.0)
                            if not ok:
                                logger.warning("Timed out waiting for goodbye playback; proceeding with shutdown.")
                        except Exception:
                            logger.exception("Fallback wait also failed")
                # Signal shutdown so the same clean shutdown path runs as when the caller hangs up
                call_ended.set()
                goodbye_enqueued = True
                break

            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        pass

    # Begin shutdown sequence
    logger.info("Shutting down tasks...")
    server_task.cancel()
    whisper_task.cancel()

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Server shut down cleanly.")

    try:
        await whisper_task
    except asyncio.CancelledError:
        logger.info("Whisper handler shut down cleanly.")

    # Conditionally persist buffered chunks
    if STORE_ALL_RESPONSE_AUDIO:
        BUFFER_DIR = "app/audio_temp"
        logger.info("Writing %d ready AudioChunks to disk in %s...", ready_chunks.qsize(), BUFFER_DIR)
        await write_all_chunks_to_disk(ready_chunks, BUFFER_DIR)
        logger.info("All chunks written.")
    else:
        # Drain queue without saving (free memory)
        drained = 0
        while not ready_chunks.empty():
            _ = await ready_chunks.get()
            drained += 1
        logger.info("Skipped writing caller chunks (STORE_ALL_RESPONSE_AUDIO=false). Drained %d buffered chunks.", drained)

    logger.info("Exiting.")

if __name__ == "__main__":
    try:
        # Pre-generate greeting/reminder/goodbye synchronously before starting the async server
        try:
            generate_static_prompt_mp3s()
        except Exception:
            logger.exception("Failed to pre-generate static prompt mp3s")

        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user. Exiting cleanly...")
