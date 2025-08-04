from app.llm_local_handler import generate_llm_response, initialize_conversation
from app.data_types import PhraseObject
from datetime import datetime

# Persistent message history for the current call session
message_history = initialize_conversation()

async def handle_phrase(phrase: PhraseObject) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] >>> Caller: {phrase.phrase_text()}", flush=True)
    #print(f"[{timestamp}] >>> Sending to local LLM...", flush=True)

    start = datetime.now()
    response, updated_history = generate_llm_response(phrase.phrase_text(), message_history)
    duration = (datetime.now() - start).total_seconds()

    print(f"⏱ LLM response time: {duration:.2f} seconds", flush=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] <<< LLM: {response}", flush=True)

    # Update the shared message history
    message_history.clear()
    message_history.extend(updated_history)

    return response
