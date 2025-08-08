from asyncio import Queue

# Queue of MP3 file paths to be played back to the caller.
llm_playback_queue: Queue[str] = Queue()
