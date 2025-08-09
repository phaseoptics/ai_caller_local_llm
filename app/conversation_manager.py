from app.llm_local_handler import generate_llm_response, initialize_conversation
from app.data_types import PhraseObject
from app.elevenlabs_handler import synthesize_speech_to_mp3
from app.queues import llm_playback_queue
from datetime import datetime
import os

# Persistent message history for the current call session
message_history = initialize_conversation()

async def handle_phrase(phrase: PhraseObject) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] >>> Caller: {phrase.phrase_text()}", flush=True)

    start = datetime.now()
    response, updated_history = generate_llm_response(phrase.phrase_text(), message_history)
    duration = (datetime.now() - start).total_seconds()

    print(f"⏱ LLM response time: {duration:.2f} seconds", flush=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] <<< LLM: {response}", flush=True)

    # Update the shared message history
    message_history.clear()
    message_history.extend(updated_history)

    # Synthesize and enqueue for playback (WebSocket sends happen only inside media_stream)
    out_dir = "app/audio_temp"
    os.makedirs(out_dir, exist_ok=True)
    mp3_path = os.path.join(out_dir, "llm_response.mp3")

    ok = synthesize_speech_to_mp3(response, mp3_path)
    if ok:
        await llm_playback_queue.put(mp3_path)
    else:
        print("⚠️ ElevenLabs synthesis failed, skipping playback enqueue.", flush=True)

    return response
