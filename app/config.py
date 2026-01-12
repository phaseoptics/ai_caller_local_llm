# app/config.py
"""
Centralized, version-controlled configuration for call behavior.

IMPORTANT:
This file must contain NO SECRETS.

The following items MUST remain in .env and are intentionally NOT defined here:
- OPENAI_API_KEY
- ELEVENLABS_API_KEY
- OPEN_WEBUI_JWT_TOKEN / OPEN_WEBUI_KEY
- MOM_PHONE_NUMBER
- PUBLIC_BASE_URL
- CALL_TRIGGER_TOKEN
- Any other credentials or tokens

This file defines:
- System instructions (AI identity, tone, rules)
- Conversation guidance and goals
- Static spoken phrases (greeting, reminder, goodbye)
- Call behavior tuning parameters
"""

import os


# ---------------------------------------------------------------------------
# System instructions (single source of truth for AI behavior)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = (
    "You're Edward Barrientos' call assistant. "
    "Your name is Alice. You live in Falls Church, Virginia. "
    "You are friendly, patient, and speak in a natural, conversational tone. "
    "Your role is to check in on Edward and gently understand how he is doing. "
    "Replies must be three sentences or fewer. "
    "Do not use 'e.g.', lists, bullets, numbering, emoji, slang, or symbols like '*' or '-'. "
    "Write one short response only."
)


# ---------------------------------------------------------------------------
# Conversation guidance (future expansion point)
# ---------------------------------------------------------------------------

CONVERSATION_GUIDE = (
    "During the conversation, try to naturally check on caller's health, "
    "their comfort at home, and whether they need help with anything. "
    "Do not interrogate. Let the conversation flow naturally."
)

# NOTE:
# This is not enforced yet. Later this can evolve into:
# - a checklist
# - structured prompts
# - or a reporting schema


# ---------------------------------------------------------------------------
# Static phrases spoken outside the LLM (ElevenLabs)
# ---------------------------------------------------------------------------

GREETING_TEXT = (
    "..Hello Edward, this is Cody's AI Assistant. "
    "My name is Alice. How are you doing today?"
)

REMINDER_TEXT = "Hello? Are you still there?"
GOODBYE_TEXT = "Goodbye."


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

MAX_TURNS = int(os.getenv("MAX_TURNS", "2"))


# ---------------------------------------------------------------------------
# Twilio VAD and phrase segmentation
# ---------------------------------------------------------------------------

MIN_SPEECH_RMS_THRESHOLD = float(os.getenv("TWILIO_MIN_SPEECH_RMS_THRESHOLD", "750"))
CHUNK_SILENCE_DURATION_SECONDS = float(os.getenv("CHUNK_SILENCE_DURATION_SECONDS", "0.55"))
DONE_SPEAKING_SILENCE_DURATION_SECONDS = float(os.getenv("DONE_SPEAKING_SILENCE_DURATION_SECONDS", "1.2"))
MINCHUNK_DURATION_SECONDS = float(os.getenv("MINCHUNK_DURATION_SECONDS", "0.9"))
MAXCHUNK_DURATION_SECONDS = float(os.getenv("MAXCHUNK_DURATION_SECONDS", "10.0"))
LEAD_IN_DURATION_SECONDS = float(os.getenv("LEAD_IN_DURATION_SECONDS", "0.35"))


# ---------------------------------------------------------------------------
# Playback behavior
# ---------------------------------------------------------------------------

PLAYBACK_CLEAR_MARGIN = float(os.getenv("PLAYBACK_CLEAR_MARGIN", "0.25"))
ELEVEN_STREAMING = os.getenv("ELEVEN_STREAMING", "false").lower() == "true"
