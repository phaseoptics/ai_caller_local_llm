import requests
import logging

# Configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:1b"
MAX_TOKENS = 80
TEMPERATURE = 0.7

# Logger setup
logger = logging.getLogger("llm_local_handler")
logger.setLevel(logging.INFO)

# Initialize a new conversation with a system prompt
def initialize_conversation() -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are a calm, intelligent, human-like assistant."
                "Never use emoji, slang, or decorative formatting."
                "Do not respond in lists."
                "Speak clearly and naturally like a helpful human."
                "Keep responses brief."
            )
        }
    ]

# Send a user prompt and return the assistant's reply and updated history
def generate_llm_response(prompt: str, message_history: list) -> tuple[str, list]:
    logger.debug(f"[Gemma] User prompt:\n{prompt}")
    message_history.append({"role": "user", "content": prompt})

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
                },
            },
            timeout=10,
        )
        response.raise_for_status()
        reply = response.json()["message"]["content"].strip()
        logger.debug(f"[Gemma] Assistant reply:\n{reply}")
        message_history.append({"role": "assistant", "content": reply})
        return reply, message_history

    except Exception as e:
        logger.error(f"[Gemma] Error calling local model: {e}")
        return "[Error generating response]", message_history
