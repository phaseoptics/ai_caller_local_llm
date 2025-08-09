# app/llm_local_handler.py
import re
import requests
import logging

# Configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:1b"
MAX_TOKENS = 120          # hard cap to discourage rambling
TEMPERATURE = 0.2         # steadier, less listy tone
TOP_P = 0.9
REPEAT_PENALTY = 1.05

logger = logging.getLogger("llm_local_handler")
logger.setLevel(logging.INFO)

SYSTEM_INSTRUCTIONS = (
    "You are a friendly and speak in a natural, conversational tone. "
    "Replies must be three sentences or fewer. "
    "Do not use 'e.g.', lists, bullets, numbering, emoji, slang, or symbols like '*' or '-'. "
    "Write one short paragraph only."
)

PER_TURN_GUARDRAIL = (
    "RULES: Answer like natural speech. Maximum three sentences. "
    "No lists, bullets, numbering, or 'e.g.'. No symbols like '*' or '-'. "
    "One short paragraph only."
)

def _postprocess_speech_style(text: str) -> str:
    """
    Make outputs speech-like and concise without hard stops.
    - Strip common list markers at line starts
    - Remove decorative symbols
    - Flatten newlines and normalize whitespace
    - Limit to three sentences
    """
    if not text:
        return text

    # Remove common list markers
    text = re.sub(r"(?m)^\s*(?:[*\-•]+|\d+\.)\s+", "", text)

    # Remove decorative symbols and avoid 'e.g.'
    text = text.replace("•", " ")
    text = text.replace("*", " ")
    text = text.replace("—", " ")
    text = text.replace("-", " ")
    text = text.replace("`", " ")
    text = text.replace("e.g.", "for example")

    # Collapse newlines and extra spaces
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    # Enforce at most three sentences
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) > 3:
        text = " ".join(parts[:3]).strip()

    return text

def initialize_conversation() -> list:
    return [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTIONS,
        }
    ]

def generate_llm_response(prompt: str, message_history: list) -> tuple[str, list]:
    logger.debug(f"[Gemma] User prompt:\n{prompt}")

    composed_prompt = f"{PER_TURN_GUARDRAIL}\n\nUser: {prompt}"
    message_history.append({"role": "user", "content": composed_prompt})

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": message_history,
                "stream": False,
                "options": {
                    "num_predict": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "repeat_penalty": REPEAT_PENALTY,
                },
            },
            timeout=15,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"].strip()
        reply = _postprocess_speech_style(raw)

        logger.debug(f"[Gemma] Assistant raw reply:\n{raw}")
        logger.debug(f"[Gemma] Assistant postprocessed reply:\n{reply}")

        message_history.append({"role": "assistant", "content": reply})
        return reply, message_history

    except Exception as e:
        logger.error(f"[Gemma] Error calling local model: {e}")
        return "[Error generating response]", message_history
