import requests
import logging

# Configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:1b"
MAX_TOKENS = 120
TEMPERATURE = 0.4

logger = logging.getLogger("llm_local_handler")
logger.setLevel(logging.INFO)

def initialize_conversation() -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are a friendly assistant that speaks in natural, conversational tone."
                "Replies must be three sentences or fewer."
                "Do not use 'e.g.', lists, or bullet points." 
                "Do not use symbols like '*' or '-'."
            )
        }
    ]

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
            timeout=30,  # was 10
        )
        response.raise_for_status()
        reply = response.json()["message"]["content"].strip()
        logger.debug(f"[Gemma] Assistant reply:\n{reply}")
        message_history.append({"role": "assistant", "content": reply})
        return reply, message_history
    except Exception as e:
        logger.error(f"[Gemma] Error calling local model: {e}")
        return "[Error generating response]", message_history
