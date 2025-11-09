# AI Caller Local LLM

This project implements a real time **AI Caller Assistant** that answers incoming phone calls using a **local LLM** for conversation, **Whisper** for speech to text, and **ElevenLabs** for speech playback.  
It is designed to run on a home server with low latency and complete control over the audio and conversation flow.

---

## Purpose

This project began as a deeply personal effort.  
My mother has dementia, and I wanted to create an AI caller that could reach out to her with **infinite patience and kindness** — a gentle, dependable companion able to check in, offer reminders, or simply keep her company when family members could not be there.

The goal of the **Youngbull AI Caller** is to demonstrate that artificial intelligence, when used locally and designed carefully, can extend compassion through technology.  
It is not about replacing people. It is about **amplifying human presence** when loved ones cannot always be on the line.

---

## Overview of Operation

1. **Incoming call via Twilio**  
   Twilio routes the call to the `/voice` webhook.  
   Twilio then opens a **WebSocket** to `/ai_caller/stream` that sends μ-law 8 kHz 20 ms audio frames and receives playback frames in real time.

2. **Voice Activity Detection (VAD)**  
   Incoming frames are decoded, their RMS loudness is measured, and they are assembled into higher level speech units (`AudioChunk`s and `PhraseObject`s).

3. **Transcription**  
   When a phrase is complete, its audio is sent to **Whisper** (local Faster-Whisper or OpenAI Whisper API) for transcription.

4. **LLM Response**  
   The transcribed text is passed to the **local LLM** through **Ollama** (default: Gemma 3:1B).  
   Guardrails ensure short, natural, ≤ 3-sentence replies.

5. **Speech Playback**  
   The LLM response is converted into μ-law 8 kHz frames by **ElevenLabs** (streaming or MP3 fallback) and sent back through the same WebSocket for immediate playback.

6. **Silence Watchdog**  
   If the caller is silent longer than `MAX_SILENCE_SECONDS`, the call ends automatically.

---

## Voice Activity Detection (VAD) Design

This project defines its own VAD logic rather than using Twilio’s 20 ms frames as speech units.

**AudioChunk**  
* A contiguous segment of decoded PCM16 audio at 8 kHz that represents detected speech.  
* Created when the RMS level exceeds `MIN_SPEECH_RMS_THRESHOLD`.  
* Continues to grow until a brief silence exceeds `CHUNK_SILENCE_DURATION_SECONDS` or its maximum duration is reached.  
* Each chunk stores:  
  `phrase_id`, `chunk_index`, `rms`, `timestamp`, `duration`, `capture_state`, and later the `transcription`.  
* After Whisper finishes, the chunk is marked `is_transcribed=True` even if empty, ensuring phrase completion logic stays deterministic.

**PhraseObject**  
* A list of related `AudioChunk`s forming a single coherent spoken phrase.  
* When silence exceeds `DONE_SPEAKING_SILENCE_DURATION_SECONDS`, the phrase closes.  
* The system concatenates non-empty chunk transcripts and sends the phrase to the LLM.  
* A phrase is considered complete when **all** its chunks are transcribed.

**Silence handling**  
* If total silence exceeds `MAX_SILENCE_SECONDS`, the system ends the call.  

This structure produces natural turn-taking and low latency while preventing the WebSocket from being accessed outside its coroutine.

---

## Repository Structure

| File | Description |
|------|--------------|
| `app/main.py` | Entry point. Starts the Quart + Hypercorn ASGI server, initializes queues, and supervises the main event loop. |
| `app/twilio_stream_handler.py` | WebSocket endpoint for Twilio Media Streams. Handles frame decoding, RMS measurement, VAD segmentation, and outbound μ-law frame playback. All writing to the socket happens within this coroutine. |
| `app/data_types.py` | Defines `AudioChunk` and `PhraseObject`. Each chunk stores raw audio bytes, metadata, and its Whisper transcription. Phrases group chunks and determine when a caller has finished speaking. |
| `app/whisper_handler.py` | Handles transcription via local Faster-Whisper or OpenAI Whisper. Converts 8 kHz PCM to 16 kHz float arrays, runs the model, and updates the corresponding `AudioChunk`. |
| `app/llm_local_handler.py` | Communicates with the local Ollama API. Enforces guardrails on responses (≤ 3 sentences, no lists). Returns concise replies for smooth conversational pacing. |
| `app/elevenlabs_handler.py` | Handles text-to-speech through ElevenLabs. Supports streaming μ-law frames or file-based MP3 synthesis. Manages pacing and playback formatting. |
| `app/conversation_manager.py` | Maintains conversation history, coordinates between the Whisper transcription loop and LLM response generation, and manages playback queues. |
| `app/queues.py` | Central asynchronous queues connecting modules (`transcription_queue`, `tts_requests_queue`, and `llm_playback_queue`). |

---

## Environment Variables

Create a `.env` file in the repository root:

```env
# General
MAX_SILENCE_SECONDS=30
STORE_ALL_RESPONSE_AUDIO=false

# Local LLM through Ollama
OLLAMA_URL=http://localhost:11434/api/chat
MODEL_NAME=gemma3:1b
MAX_TOKENS=120
TEMPERATURE=0.6

# ElevenLabs
ELEVENLABS_API_KEY=your_api_key
ELEVENLABS_VOICE_ID=your_voice_id
ELEVENLABS_MODEL=eleven_flash_v2_5
ELEVENLABS_SPEED=0.9
ELEVEN_STREAMING=true

# Whisper
USE_LOCAL_WHISPER=true
LOCAL_MODEL_NAME=small.en
LOCAL_COMPUTE_TYPE=int8_float32
LOCAL_BEAM_SIZE=5
FORCE_LANGUAGE=en
OPENAI_API_KEY=your_openai_key_if_needed
```

---

## Deployment Overview

**Endpoints required by Twilio**

* HTTPS webhook for Voice: `POST /voice`  
* Secure WebSocket for Media Streams: `wss://YOUR_DOMAIN/ai_caller/stream`

**Typical flow**

1. Twilio calls `https://YOUR_DOMAIN/voice`.  
2. The `/voice` route returns TwiML instructing Twilio to open the WebSocket to `/ai_caller/stream`.  
3. Twilio streams 20 ms μ-law frames in both directions for live conversation.

---

## WebSocket Rules

* Only write to the socket inside the `/stream` coroutine.  
* Use queues for background tasks that need to send audio.  
* Stream playback at one 20 ms frame per iteration.  
* Send `{ "event": "clear" }` after playback to flush Twilio’s audio buffer.

Outbound frame example:

```json
{
  "event": "media",
  "streamSid": "YOUR_STREAM_SID",
  "media": {
    "payload": "BASE64_ULAW_160_BYTES"
  }
}
```

Clear command:

```json
{ "event": "clear", "streamSid": "YOUR_STREAM_SID" }
```

---

## Running the App

```bash
pip install -r requirements.txt
python main.py
```

Expose your server securely (for example with Caddy or ngrok) and point Twilio to your public HTTPS domain.

---

## License

This project is released under the **Youngbull AI Caller License (BY-NC-SA)**.

* ✅ Attribution required: cite “Youngbull AI Caller – https://github.com/phaseoptics/ai_caller_local_llm”.  
* 🚫 Non-commercial use only: may be used, studied, and modified freely for personal, academic, or research purposes.  
* 🔄 Share-alike: derivative works must remain open under the same license.  
* 💼 Commercial use requires a separate license.

See the full terms in the [LICENSE](LICENSE) file.  
Commercial inquiries should be opened as a GitHub issue:  
https://github.com/phaseoptics/ai_caller_local_llm/issues  

© 2025 Cody Youngbull

