---

# AI Caller Local LLM

This project implements a **real-time AI Caller Assistant** that answers incoming phone calls using a **local LLM** (Gemma 3:1B) for conversation, **Whisper** for speech-to-text, and optionally **ElevenLabs** for text-to-speech playback.

It is designed to run on your **home server** with low latency and complete control over conversation flow.

---

## Operation Commands

Activate the virtual environment in the first terminal.

**Terminal 1**
```bash
source venv/bin/activate
```

Expose the application port using ngrok in a second terminal.

**Terminal 2**
```bash
ngrok http 5000 --domain=cody-ai.ngrok.app
```

Start the Caller Assistant in the first terminal.

**Terminal 1**
```bash
python main.py
```

The Assistant is now running and ready to receive or place calls.

To trigger an outgoing call, activate the virtual environment in a third terminal.

**Terminal 3**
```bash
source venv/bin/activate
```

Trigger the outbound call.

**Terminal 3**
```bash
CALL_TRIGGER_TOKEN="$(grep '^CALL_TRIGGER_TOKEN=' .env | cut -d= -f2-)" && \
curl -i -sS -X POST \
  -H "Authorization: Bearer ${CALL_TRIGGER_TOKEN}" \
  https://cody-ai.ngrok.app/call_mom
```

**Note:** The `.env` file must be present and loaded in any terminal used to run or trigger the Assistant.

---

## Overview of Operation

1. **Incoming call via Twilio**
   - Twilio routes call audio to the `/voice` webhook.
   - Twilio then opens a **WebSocket** to `/stream` to send μ-law 8 kHz 20 ms audio frames and receive real-time audio back.

2. **Voice Activity Detection (VAD)**
   - Incoming audio is chunked into `AudioChunk` objects when RMS loudness exceeds a threshold.
   - Each chunk belongs to a `PhraseObject` representing one spoken phrase.

3. **Transcription**
   - Completed phrases are sent to **Whisper** (local faster-whisper or OpenAI API) for transcription.

4. **LLM Response**
   - The transcription is sent to the **local LLM** (`gemma3:1b` via Ollama).
   - A **system prompt** and **per-turn guardrail** enforce natural, ≤ 3-sentence, no-lists replies.

5. **Speech Playback**
   - The LLM’s response is converted to μ-law 8 kHz frames (via ElevenLabs streaming or MP3 fallback) and sent back over the same Twilio WebSocket.

6. **Silence Watchdog**
   - If the caller is silent for `MAX_SILENCE_SECONDS` (default 30s), the call ends automatically.

---

## Repository Structure

### `app/main.py`
- **Purpose:** Entry point. Starts Hypercorn ASGI server for Quart, launches Whisper transcription loop, and runs the max-silence watchdog.
- **Key Parameters:**
  - `MAX_SILENCE_SECONDS` — seconds of caller silence before ending call (default 30).
  - `STORE_ALL_RESPONSE_AUDIO` — `"true"` to save all assistant audio responses to `app/audio_temp`.

### `app/twilio_stream_handler.py`
- **Purpose:** Handles Twilio WebSocket `/stream` for incoming/outgoing audio.
- **Key Parameters:**
  - `MIN_SPEECH_RMS_THRESHOLD` — RMS level above which caller speech is detected.
  - `CHUNK_SILENCE_DURATION_SECONDS` — short silence within phrase before splitting.
  - `DONE_SPEAKING_SILENCE_DURATION_SECONDS` — long silence marking phrase end.
  - `MINCHUNK_DURATION_SECONDS` — minimum chunk length.
  - `MAXCHUNK_DURATION_SECONDS` — maximum chunk length.
  - `LEAD_IN_DURATION_SECONDS` — pre-roll before speech.
  - `ELEVEN_STREAMING` — `"true"` for live ElevenLabs streaming, `"false"` for MP3 file playback.

### `app/whisper_handler.py`
- **Purpose:** Transcribes audio chunks via faster-whisper or OpenAI Whisper API.
- **Key Parameters:**
  - `USE_LOCAL_WHISPER` — `True` for faster-whisper, `False` for API.
  - `LOCAL_MODEL_NAME` — e.g., `"small.en"`.
  - `LOCAL_COMPUTE_TYPE` — e.g., `"int8_float32"`.
  - `LOCAL_BEAM_SIZE` — higher for better accuracy.
  - `FORCE_LANGUAGE` — e.g., `"en"`.

### `app/llm_local_handler.py`
- **Purpose:** Sends transcription to local LLM via Ollama API, enforces conversation style.
- **Key Parameters:**
  - `OLLAMA_URL`, `MODEL_NAME`, `MAX_TOKENS`, `TEMPERATURE`.
  - Guardrails enforce ≤ 3 sentences, natural tone, no lists.

### `app/conversation_manager.py`
- **Purpose:** Maintains conversation history and enqueues playback.
- **Key Parameters:**
  - `ELEVEN_STREAMING` — controls streaming vs MP3 playback.

### `app/elevenlabs_handler.py`
- **Purpose:** ElevenLabs TTS streaming and file-based synthesis.
- **Key Parameters:**
  - `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL`, `ELEVENLABS_SPEED`.

### `app/data_types.py`
- **AudioChunk:** Holds audio bytes and metadata.
- **PhraseObject:** Holds multiple `AudioChunk`s and their transcripts.

### `app/queues.py`
- `llm_playback_queue` — MP3 paths for playback.
- `tts_requests_queue` — text for live TTS streaming.

---

## Environment Variables

Example `.env`:

```env
MAX_SILENCE_SECONDS=30
STORE_ALL_RESPONSE_AUDIO=false

# LLM
OLLAMA_URL=http://localhost:11434/api/chat
MODEL_NAME=gemma3:1b
MAX_TOKENS=120
TEMPERATURE=0.4

# ElevenLabs
ELEVENLABS_API_KEY=your_api_key
ELEVENLABS_VOICE_ID=your_voice_id
ELEVENLABS_MODEL=eleven_flash_v2_5
ELEVENLABS_SPEED=0.90
ELEVEN_STREAMING=true

# Whisper
USE_LOCAL_WHISPER=true
LOCAL_MODEL_NAME=small.en
LOCAL_COMPUTE_TYPE=int8_float32
LOCAL_BEAM_SIZE=5
FORCE_LANGUAGE=en
OPENAI_API_KEY=your_openai_key_if_needed


---

Deploying with Twilio + Caddy (HTTPS & WebSockets)

This app needs two public endpoints:

1. HTTPS webhook for Twilio Voice: POST /voice


2. Secure WebSocket for Twilio Media Streams: wss://<your-domain>/ai_caller/stream



Twilio Setup

A Call Comes In: Webhook → https://YOUR_DOMAIN/voice

The /voice route returns TwiML that tells Twilio to open a WebSocket to /ai_caller/stream.


Caddyfile Example

YOUR_DOMAIN.com {
  encode zstd gzip

  @ws path /ai_caller/stream
  handle @ws {
    reverse_proxy 127.0.0.1:5000 {
      transport http {
        read_buffer  8192
        write_buffer 8192
        read_timeout  600s
        write_timeout 600s
        idle_timeout  600s
      }
    }
  }

  handle {
    reverse_proxy 127.0.0.1:5000
  }

  redir https://YOUR_DOMAIN.com{uri} 301
}

:80 {
  respond "Use HTTPS" 308
}

Reload Caddy:

sudo systemctl reload caddy


---

WebSocket Rules

1. Only write to the WebSocket inside the /stream coroutine — use queues to send from background tasks.


2. Playback must be streamed — μ-law 8 kHz, 20 ms frames, base64 encoded.


3. Send {"event": "clear"} after playback to flush caller-buffered audio.


4. Pace frames at 20 ms for smooth playback.




---

JSON Format for Outbound Audio

Media frame:

{
  "event": "media",
  "streamSid": "YOUR_STREAM_SID",
  "media": {
    "payload": "BASE64_ULAW_160_BYTES"
  }
}

Clear command:

{
  "event": "clear",
  "streamSid": "YOUR_STREAM_SID"
}


---

Running the App

pip install -r requirements.txt
python main.py

Point Twilio to your domain as shown above.

---
