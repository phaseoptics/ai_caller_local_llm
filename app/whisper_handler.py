import os
import logging
import asyncio
import wave
from tempfile import NamedTemporaryFile
from openai import AsyncOpenAI
from app.twilio_stream_handler import ready_chunks, transcription_queue, AudioChunk

from app.twilio_stream_handler import detected_phrases, PhraseObject
from app.conversation_manager import handle_phrase


# This queue will hold fully transcribed chunks ready for stitching
stitch_ready_chunks: asyncio.Queue = asyncio.Queue()

# --- Logging Configuration ---
logger = logging.getLogger("whisper_handler")
logger.setLevel(logging.INFO if os.getenv("LOGGING_ENABLED", "false").lower() == "true" else logging.CRITICAL)

# --- OpenAI Client Initialization ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Save audio bytes to a properly formatted WAV file ---
def write_chunk_to_temp_wav(audio_bytes: bytes) -> str:
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            with wave.open(tmp, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(8000)
                wav_file.writeframes(audio_bytes)
            return tmp.name
    except Exception as e:
        logger.error(f"Failed to write chunk to WAV: {e}")
        return ""

# --- Transcribe audio using Whisper API ---
async def transcribe_audio(audio_bytes: bytes) -> str:
    try:
        tmp_path = write_chunk_to_temp_wav(audio_bytes)
        if not tmp_path:
            return ""

        with open(tmp_path, "rb") as audio_file:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            return str(response).strip()

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""

# --- Background Task: Continuously transcribe chunks ---
async def whisper_transcription_loop(thread_id: str):
    logger.info("Whisper transcription loop started.")
    processed_chunks = set()

    while True:
        try:
            chunk: AudioChunk = await transcription_queue.get()

            if chunk.transcription != "" or chunk.chunk_index in processed_chunks:
                continue

            transcript = await transcribe_audio(chunk.audio_bytes)

            if not transcript:
                logger.warning(f"Whisper returned an empty transcript for chunk {chunk.chunk_index}.")

            chunk.transcription = transcript
            processed_chunks.add(chunk.chunk_index)

            logger.info(f"Transcript for chunk {chunk.chunk_index} in phrase {chunk.phrase_id}:\n{transcript}")
            await stitch_ready_chunks.put(chunk)

            phrase = detected_phrases.get(chunk.phrase_id)

            # temporary debug logging remove if okay
            #logger.info(f"Checking if phrase {chunk.phrase_id} is complete. Chunks: {len(phrase.chunks)}")
            #for c in phrase.chunks:
            #    logger.info(f"  Chunk {c.chunk_index} transcript: {repr(c.transcription)}")

            if phrase and phrase.is_complete() and not phrase.is_done:
                phrase.is_done = True  # mark it done so we don’t reprocess
                logger.info(f"Phrase {phrase.phrase_id} is complete. GPT task launched.")
                logger.info(f"[DEBUG] Phrase text: '{phrase.phrase_text()}'")
                asyncio.create_task(handle_phrase(phrase, thread_id))
        except asyncio.CancelledError:
            logger.info("Whisper transcription loop cancelled.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in transcription loop: {e}")