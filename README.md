# AI Caller Local LLM

A real time AI Caller Assistant for home servers. It answers calls through Twilio Media Streams, transcribes speech with Whisper, reasons with a local LLM through Ollama, and speaks back with ElevenLabs. Designed for low latency, privacy, and full control of the conversation loop.

---

## Purpose

This project began as a personal effort.  
My mother has dementia, and I wanted to create an AI caller that would reach out with infinite patience and kindness, check in, offer reminders, and provide simple companionship when family could not be there.  

The goal is to show that careful, local, privacy respecting AI can extend compassion through technology. It is not about replacing people. It is about amplifying human presence when loved ones cannot always be on the line.

---

## Overview of Operation

1. **Incoming call through Twilio**
   * Twilio hits the `/voice` webhook.
   * Twilio then opens a secure WebSocket to `/ai_caller/stream` and sends μ law 8 kHz 20 ms frames as base64.

2. **Voice Activity Detection**
   * Caller audio is assembled into higher level speech units called AudioChunks and Phrases (defined below), not just raw Twilio frames.
   * Completed phrases are sent to Whisper for transcription.

3. **Transcription**
   * Whisper runs locally through faster whisper or remotely through OpenAI Whisper, based on configuration.

4. **Local LLM response**
   * Transcripts are sent to a local LLM through Ollama. The default model is Gemma 3:1B with concise, natural guardrails.

5. **Speech back to the caller**
   * ElevenLabs turns the LLM text into μ law 8 kHz frames for immediate playback over the same WebSocket, or falls back to MP3 files that are converted to frames.

6. **Silence watchdog**
   * If the caller stays silent beyond a timeout, the call ends cleanly.

---

## Voice Activity Detection design

This project uses custom speech units that are distinct from Twilio’s transport frames.

**AudioChunk**  
* A contiguous segment of decoded PCM16 audio at 8 kHz that represents detected speech.  
* Created when RMS exceeds `MIN_SPEECH_RMS_THRESHOLD`. Grows while speech continues, splits on short silences, and is bounded by min and max chunk duration settings.  
* Stores metadata: `phrase_id`, `chunk_index`, `rms`, `timestamp`, `duration`, `capture_state`.  
* Whisper writes the transcript back into the same object, setting `transcription` and `is_transcribed=True` even if the text is empty.

**PhraseObject**  
* An ordered list of AudioChunks that together form one coherent spoken phrase.  
* When a longer silence reaches `DONE_SPEAKING_SILENCE_DURATION_SECONDS`, the phrase is closed.  
* The phrase text is the concatenation of all non empty chunk transcripts, then it is sent to the LLM.  
* A phrase is considered complete when every chunk has been processed by Whisper, not only when text is non empty.

**Why this matters**  
* Natural segmentation for human like turn taking.  
* Lower latency than full file boundaries.  
* Clear async coordination through queues so that the WebSocket is only written from the handler coroutine.

---

## Repository structure

The following files reflect the working modules you uploaded.

* `app/data_types.py`  
  Defines `AudioChunk` and `PhraseObject`. Each chunk includes raw audio bytes, timing, RMS, capture state, and holds its own Whisper result through the `transcription` field. `PhraseObject.is_complete()` returns true when all chunks have been processed by Whisper.

* `app/queues.py`  
  Central asyncio queues.  
  `llm_playback_queue` carries MP3 file paths for fallback playback.  
  `tts_requests_queue` carries plain text for true streaming TTS back to the handler.

* `app/whisper_handler.py`  
  Transcription pipeline. Can run faster whisper locally or call the OpenAI Whisper API.  
  Includes 8 kHz PCM to 16 kHz float upsampling for local models.  
  Runs a background `whisper_transcription_loop()` that reads chunks from the transcription queue, transcribes them in place, and triggers phrase handling once all chunks are processed.

* `app/llm_local_handler.py`  
  Minimal, focused chat call to local Ollama.  
  Defaults: `MODEL_NAME="gemma3:1b"`, concise replies, no lists, natural tone.  
  Returns the assistant reply and updated history for stateful conversation.

* `app/elevenlabs_handler.py`  
  Two paths.  
  True streaming: yields μ law 8 kHz base64 frames directly in an async iterator.  
  Fallback: synth to MP3, normalize and compress, then convert to μ law frames for smooth 20 ms pacing.  
  Includes an optional greeting generator.

* `app/conversation_manager.py`  
  Holds the per call message history.  
  On phrase completion, calls the local LLM, logs timing, and either enqueues text for streaming TTS or writes an MP3 for fallback playback.

> Note  
> Your WebSocket stream handler and the main app entry point are not shown above. In this design, the WebSocket coroutine is responsible for receiving Twilio frames, doing VAD assembly, and sending audio back, while all blocking work is offloaded via queues or executors. The WebSocket object is never shared across tasks.

---

## Environment variables

Copy and adapt as `.env` in the repo root.

```env
# Core
MAX_SILENCE_SECONDS=30
STORE_ALL_RESPONSE_AUDIO=false

# Local LLM through Ollama
OLLAMA_URL=http://localhost:11434/api/chat
MODEL_NAME=gemma3:1b
MAX_TOKENS=120
TEMPERATURE=0.7

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
