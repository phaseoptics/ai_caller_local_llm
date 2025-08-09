from asyncio import Queue

# Queue of MP3 file paths to be played back to the caller (file-based fallback).
llm_playback_queue: Queue[str] = Queue()

# Queue of TEXT responses to synthesize via ElevenLabs streaming (true stream-through).
tts_requests_queue: Queue[str] = Queue()
