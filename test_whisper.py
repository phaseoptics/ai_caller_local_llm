import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Missing OPENAI_API_KEY in environment.")
    exit(1)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def transcribe_file():
    file_path = "app/audio_static/ae12edf1-acf0-4261-92de-4dd429bebfbd__chunk_382.wav"
    try:
        with open(file_path, "rb") as f:
            print("Sending file to Whisper API...")
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json"
            )
            print("Raw response from OpenAI:\n", response)
            print("\nTranscript Text:\n", response.text)

    except Exception as e:
        import traceback
        print("Error during transcription:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(transcribe_file())
