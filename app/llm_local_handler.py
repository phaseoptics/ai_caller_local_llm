import requests
import logging

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"
MAX_TOKENS = 80
TEMPERATURE = 0.7

logger = logging.getLogger("gpt_handler")
logger.setLevel(logging.INFO)

def generate_llm_response(prompt: str) -> str:
    logger.info(f"[GPT] Sending prompt to Gemma:\n{prompt}")
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                },
            },
            timeout=10,
        )
        response.raise_for_status()
        reply = response.json()["response"].strip()
        logger.info(f"[GPT] Response from Gemma:\n{reply}")
        return reply
    except Exception as e:
        logger.error(f"[GPT] Error calling local model: {e}")
        return "[Error generating response]"
