from asyncio import Queue

# Queue of MP3 file paths to be played back to the caller (file-based fallback).
llm_playback_queue: Queue[str] = Queue()

# Queue of TEXT responses to synthesize via ElevenLabs streaming (true stream-through).
tts_requests_queue: Queue[str] = Queue()

# Assistant playback tracking (used to pause the silence timer while assistant is speaking)
# We track total playback time since the last caller speech. When the caller speaks we reset
# the accumulator so silence is measured only during caller silence, excluding assistant replies.
import time

assistant_playing: bool = False
_playback_pause_accumulator: float = 0.0
_playback_pause_start: float = 0.0

def start_assistant_playing() -> None:
	global assistant_playing, _playback_pause_start
	if not assistant_playing:
		assistant_playing = True
		_playback_pause_start = time.monotonic()

def stop_assistant_playing() -> None:
	global assistant_playing, _playback_pause_accumulator, _playback_pause_start
	if assistant_playing:
		_playback_pause_accumulator += time.monotonic() - _playback_pause_start
		assistant_playing = False
		_playback_pause_start = 0.0

def reset_playback_pause_accumulator() -> None:
	"""Called when caller speech is detected to reset the playback pause counter."""
	global _playback_pause_accumulator, _playback_pause_start
	_playback_pause_accumulator = 0.0
	_playback_pause_start = time.monotonic() if assistant_playing else 0.0

def get_playback_pause_since_reset() -> float:
	"""Return total assistant playback time since last reset (includes current playing).
	This value should be subtracted from raw silent time to get effective caller silence.
	"""
	total = _playback_pause_accumulator
	if assistant_playing and _playback_pause_start:
		total += time.monotonic() - _playback_pause_start
	return total

async def wait_for_playback_completion(timeout: float | None = 30.0) -> bool:
	"""Await until no assistant playback is active and the llm_playback_queue is empty.

	Returns True if playback completed within timeout, False on timeout.
	"""
	import asyncio, time
	start = time.monotonic()
	while True:
		# If nothing is playing and the file queue is empty, we're done
		try:
			queue_empty = llm_playback_queue.empty()
		except Exception:
			queue_empty = True

		if not assistant_playing and queue_empty:
			return True

		if timeout is not None and (time.monotonic() - start) > timeout:
			return False

		await asyncio.sleep(0.05)
