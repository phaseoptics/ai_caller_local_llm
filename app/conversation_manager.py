from app.gpt_assistant_handler import init_conversation, send_user_message
from app.twilio_stream_handler import PhraseObject
from datetime import datetime

# Thread memory initialized once
thread_id = None

async def handle_phrase(phrase: PhraseObject, thread_id: str) -> str:

    #print(f"[DEBUG] Phrase text: {repr(phrase.phrase_text())}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] >>> Caller: {phrase.phrase_text()}", flush=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >>> Sending to GPT: '{phrase.phrase_text()}'", flush=True)
    start = datetime.now()
    response = await send_user_message(thread_id, phrase.phrase_text())
    duration = (datetime.now() - start).total_seconds()
    print(f"⏱ GPT response time: {duration:.2f} seconds", flush=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] <<< GPT: {response}", flush=True)

    return response
