# app/llm_local_handler.py
import re
import requests
import logging
import os
import time
from typing import Optional
try:
    from dotenv import load_dotenv
    # Load environment from .env if present
    load_dotenv()
except Exception:
    logger = logging.getLogger("llm_local_handler")
    logger.debug("python-dotenv not available; skipping .env load. Make sure env vars are exported if running without .env support.")

# Configuration and environment-driven switches
USE_OPEN_WEBUI = os.getenv("USE_OPEN_WEBUI", "true").lower() != "false"
OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "http://127.0.0.1:3000")
# Backwards-compatible env names: prefer newly-provided OPEN_WEBUI_KEY or OPEN_WEBUI_JWT_TOKEN
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")
OPEN_WEBUI_KEY = os.getenv("OPEN_WEBUI_KEY")
OPEN_WEBUI_JWT_TOKEN = os.getenv("OPEN_WEBUI_JWT_TOKEN")
# Choose token priority: JWT token > OPEN_WEBUI_KEY > OPENWEBUI_API_KEY
if OPEN_WEBUI_JWT_TOKEN:
    OPENWEBUI_API_KEY = OPEN_WEBUI_JWT_TOKEN
elif OPEN_WEBUI_KEY:
    OPENWEBUI_API_KEY = OPEN_WEBUI_KEY

# Backwards-compatible Ollama URL (used when USE_OPEN_WEBUI is false)
OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:1b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.05"))

logger = logging.getLogger("llm_local_handler")
logger.setLevel(logging.INFO)

# Track whether we've logged the chosen URL once
_logged_url: Optional[str] = None

SYSTEM_INSTRUCTIONS = (
    "You are friendly and speak in a natural, conversational tone. "
    "Replies must be three sentences or fewer. "
    "Do not use 'e.g.', lists, bullets, numbering, emoji, slang, or symbols like '*' or '-'. "
    "Write one short paragraph only."
)

# Per-turn guardrail is merged into the single system message so the backend
# always receives a consistent system prompt at messages[0]. This avoids
# duplicating system/guardrail text on every user message and keeps history
# tidy when we send the full `message_history` to Open WebUI (app-side memory).
PER_TURN_GUARDRAIL = (
    "RULES: Answer using natural speech. Maximum three sentences. "
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
    # Combine the persistent system instructions with the per-turn guardrail
    # into a single system message. This message must remain at index 0 of
    # `message_history` so we always send the same system context to Open WebUI.
    return [
        {
            "role": "system",
            "content": f"{SYSTEM_INSTRUCTIONS}\n\n{PER_TURN_GUARDRAIL}",
        }
    ]

def generate_llm_response(prompt: str, message_history: list) -> tuple[str, list]:
    """
    Generate a response using either Open WebUI (preferred) or Ollama.

    Contract/behavior:
    - `message_history` is a mutable list where the first element MUST be the
      system message (initialize_conversation provides it).
    - This function will append a single `{'role':'user','content': prompt}`
      entry, send the full `message_history` to the backend, then append the
      assistant response to `message_history` before returning.
    """

    logger.debug("[Gemma] User prompt:\n%s", prompt)

    # Defensive: ensure message_history exists and begins with the system message.
    if not message_history or not isinstance(message_history, list):
        message_history = initialize_conversation()
    elif not (isinstance(message_history[0], dict) and message_history[0].get("role") == "system"):
        # Prepend a system message if the history does not have one.
        message_history.insert(0, {"role": "system", "content": f"{SYSTEM_INSTRUCTIONS}\n\n{PER_TURN_GUARDRAIL}"})

    # Append only the raw user prompt to the history. We keep the system message
    # separate and do not re-insert guardrails per-user-message; the system
    # prompt already contains the guardrail.
    message_history.append({"role": "user", "content": prompt})

    # Decide which backend to call
    if USE_OPEN_WEBUI:
        # Open WebUI expects the OpenAI-style chat completions at /api/chat/completions
        url = OPENWEBUI_BASE_URL.rstrip("/") + "/api/chat/completions"
    else:
        url = OLLAMA_URL

    # Log the chosen URL once
    global _logged_url
    if _logged_url is None:
        logger.info("LLM backend URL: %s (USE_OPEN_WEBUI=%s)", url, USE_OPEN_WEBUI)
        _logged_url = url

    # Prepare the request payload depending on backend
    if USE_OPEN_WEBUI:
        payload = {
            "model": MODEL_NAME,
            "messages": message_history,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if OPENWEBUI_API_KEY:
            headers["Authorization"] = f"Bearer {OPENWEBUI_API_KEY}"

        timeout = 10
        # Simple retry on 5xx with exponential backoff (200ms, then 600ms)
        backoffs = [0.2, 0.6]
        last_exc: Optional[Exception] = None
        for attempt, backoff in enumerate([None] + backoffs, start=1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                if 500 <= resp.status_code < 600 and attempt <= len(backoffs) + 1:
                    logger.warning("OpenWebUI returned %d on attempt %d, retrying after %.3f s", resp.status_code, attempt, backoff or 0.0)
                    last_exc = RuntimeError(f"HTTP {resp.status_code}")
                    if backoff:
                        time.sleep(backoff)
                    continue
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"].strip()
                reply = _postprocess_speech_style(raw)

                logger.debug("[Gemma] Assistant raw reply:\n%s", raw)
                logger.debug("[Gemma] Assistant postprocessed reply:\n%s", reply)

                message_history.append({"role": "assistant", "content": reply})
                # Return a shallow copy of the history so callers that clear/extend
                # the original list don't inadvertently erase the stored history.
                return reply, list(message_history)
            except Exception as e:
                last_exc = e
                # If this was a 5xx and retries exhausted, fall through to error handling
                if attempt <= len(backoffs):
                    # already slept above on 5xx; for other exceptions, sleep now
                    time.sleep(backoff or 0)
                    continue
                logger.error("[Gemma] Error calling OpenWebUI: %s", e)
                return "[Error generating response]", list(message_history)

    else:
        # Ollama-compatible legacy path (kept for A/B testing)
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
                timeout=10,
            )
            response.raise_for_status()
            raw = response.json()["message"]["content"].strip()
            reply = _postprocess_speech_style(raw)

            logger.debug("[Gemma] Assistant raw reply:\n%s", raw)
            logger.debug("[Gemma] Assistant postprocessed reply:\n%s", reply)

            message_history.append({"role": "assistant", "content": reply})
            return reply, list(message_history)

        except Exception as e:
            logger.error("[Gemma] Error calling Ollama: %s", e)
            return "[Error generating response]", list(message_history)


if __name__ == "__main__":
    # Two-turn sanity check to validate app-side message history is preserved
    # across consecutive calls to generate_llm_response(). This demonstrates
    # that we send the full `message_history` to Open WebUI and retain context.
    hist = initialize_conversation()
    print("Initial history:", hist)

    print("\nTurn 1: teach a fact to the model")
    resp1, hist = generate_llm_response("Remember that my favorite color is ultramarine.", hist)
    print("Turn 1 response:", resp1)

    print("\nTurn 2: ask a follow-up that requires memory")
    resp2, hist = generate_llm_response("What color did I say was my favorite?", hist)
    print("Turn 2 response:", resp2)
