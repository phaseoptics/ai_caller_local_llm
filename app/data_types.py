from dataclasses import dataclass

@dataclass
class AudioChunk:
    phrase_id: str
    chunk_index: int
    audio_bytes: bytes
    rms: float
    timestamp: float  # seconds since start of stream
    duration: float   # len(audio_bytes) / (8000 * 2)
    transcription: str = ""
    capture_state: str = ""  # "listening" or "speaking"
    is_transcribed: bool = False  # NEW: set True once Whisper finishes (even if transcript is empty)

@dataclass
class PhraseObject:
    phrase_id: str
    chunks: list[AudioChunk]
    is_done: bool = False

    def is_complete(self) -> bool:
        # NEW: completion is based on "was processed", not "has non-empty text"
        return all(chunk.is_transcribed for chunk in self.chunks)

    def phrase_text(self) -> str:
        sorted_chunks = sorted(self.chunks, key=lambda c: c.chunk_index)
        # Preserve existing behavior: only join non-empty transcripts
        return " ".join(
            chunk.transcription.strip()
            for chunk in sorted_chunks
            if chunk.transcription
        ).strip()
