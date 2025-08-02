from dataclasses import dataclass

@dataclass
class AudioChunk:
    phrase_id: str
    chunk_index: int
    audio_bytes: bytes
    rms: float
    timestamp: float  # seconds since start of stream
    duration: float  # should become len(audio_chunk.audio_bytes) / (8000 * 2)
    transcription: str = ""
    capture_state: str = ""  # "listening or "speaking"

@dataclass
class PhraseObject:
    phrase_id: str
    chunks: list[AudioChunk]
    is_done: bool = False

    def is_complete(self) -> bool:
        return all(chunk.transcription for chunk in self.chunks)

    def phrase_text(self) -> str:
        sorted_chunks = sorted(self.chunks, key=lambda c: c.chunk_index)
        return " ".join(chunk.transcription.strip() for chunk in sorted_chunks if chunk.transcription).strip()
