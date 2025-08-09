import os
from uuid import uuid4
from datetime import datetime

from app.llm_local_handler import generate_llm_response, initialize_conversation
from app.data_types import PhraseObject
from app.elevenlabs_handler import synthesize_speech_to_mp3
from app.queues import llm_playback_queue, tts_requests_queue

# Persistent message history for the current call session
message_history = initialize_conversation()
ELEVEN_STREAMING = os.getenv("ELEVEN_STREAMING", "false").lower() == "true"

MAX_TURNS = 2  # keep only the last 2 (user,assistant) pairs

def _trim_history(hist: list, max_turns: int = MAX_TURNS) -> None:
    if not hist:
        return
    # keep system message + last N*2 role messages
    system = hist[0:1]
    tail = hist[-(max_turns * 2):] if len(hist) > 1 else []
    hist.clear()
    hist.extend(system + tail)

async def handle_phrase(phrase: PhraseObject) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] >>> Caller: {phrase.phrase_text()}", flush=True)

    start = datetime.now()
    response, updated_history = generate_llm_response(phrase.phrase_text(), message_history)
    duration = (datetime.now() - start).total_seconds()

    print(f"⏱ LLM response time: {duration:.2f} seconds", flush=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] <<< LLM: {response}", flush=True)

    # Update the shared message history with trimming
    message_history.clear()
    message_history.extend(updated_history)
    _trim_history(message_history, MAX_TURNS)

    if ELEVEN_STREAMING:
        await tts_requests_queue.put(response)
    else:
        out_dir = "app/audio_temp"
        os.makedirs(out_dir, exist_ok=True)
        mp3_path = os.path.join(out_dir, f"llm_response__{uuid4().hex}.mp3")
        ok = synthesize_speech_to_mp3(response, mp3_path)
        if ok:
            await llm_playback_queue.put(mp3_path)
        else:
            print("⚠️ ElevenLabs synthesis failed, skipping playback enqueue.", flush=True)

    return response
