# app/transcript_manager.py
import os
from datetime import datetime
from typing import List

# In memory transcript lines for the current run
_transcript_lines: List[str] = []


def append_transcript_line(role: str, text: str) -> None:
    """
    Append one line to the transcript buffer.
    role: "Caller" or "Assistant" (or any label you want)
    text: the utterance text
    """
    if text is None:
        return
    t = str(text).strip()
    if not t:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _transcript_lines.append(f"[{ts}] {role}: {t}")


def write_transcript(output_path: str) -> bool:
    """
    Overwrite output_path with the transcript for this run.
    Returns True on success, False on failure.
    """
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for line in _transcript_lines:
                f.write(line + "\n")

        return True
    except Exception:
        return False
