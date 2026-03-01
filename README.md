# Live Speech Assistant (Streamlit + Sarvam AI)

This project bootstraps a fully local-first Streamlit assistant that listens to live microphone audio, sends it to Sarvam AI for speech-to-text (STT), routes the transcribed text through a lightweight dialogue brain, then synthesizes the reply back to audio with Sarvam's text-to-speech (TTS) service. The UI keeps a running chat timeline per browser session and doubles as a playground for experimenting with prompt styles, voice selections, and deployment to Streamlit Community Cloud.

## Features
- **Voice in / voice out** powered by `streamlit-webrtc` plus Sarvam STT+TTS APIs.
- **Session-scoped chat history** with automatic assistant replies and audio playback.
- **Modular client wrappers** (`sarvam_client.py`, `assistant_logic.py`, `audio_utils.py`) so you can swap providers or add caching without touching the UI.
- **Environment-first configuration**: keep secrets outside source via `.env` locally or Streamlit secrets in production.
- **Deployment ready**: includes requirements, config, and documentation for Streamlit Cloud.

## Quickstart
1. **Create & activate a virtual environment** (example shown for Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Set environment variables** (either via `.env` or your shell):
   ```env
   SARVAM_API_KEY=your_api_key_here
   SARVAM_CHAT_MODEL=bhashini-chat-lite        # optional, defaults in code
   SARVAM_TTS_VOICE=general-male               # optional voice preset
   ```
4. **Run Streamlit**:
   ```powershell
   streamlit run app.py
   ```
5. **Allow microphone access** when prompted, click *"Capture & Respond"* to process the most recent utterance, and/or use the chat input for typed prompts.

## File / Module Overview
| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, WebRTC plumbing, session state orchestration. |
| `sarvam_client.py` | Thin wrapper around Sarvam AI REST endpoints for STT, TTS, chat completions. |
| `assistant_logic.py` | Conversation state manager + fallback heuristics when LLM calls fail. |
| `audio_utils.py` | Helper functions for converting WebRTC frames to WAV bytes, normalizing signals, etc. |
| `requirements.txt` | Python dependencies. |
| `.streamlit/config.toml` | UI theming + caching defaults for Streamlit. |
| `README.md` | This document. |

## Configuration
| Variable | Description |
| --- | --- |
| `SARVAM_API_KEY` | **Required.** Secret key from Sarvam AI dashboard. |
| `SARVAM_BASE_URL` | Override if Sarvam provides a region-specific endpoint. Defaults to `https://api.sarvam.ai/v1/`. |
| `SARVAM_CHAT_MODEL` | Chat model slug for assistant responses. |
| `SARVAM_TTS_VOICE` | Voice preset for synthesized speech. |
| `SARVAM_STT_LANGUAGE` | Two-letter ISO code for preferred transcription language (e.g., `en`, `hi`). |

Store these locally in a `.env` file (ignored by Git) or configure Streamlit Cloud secrets (`.streamlit/secrets.toml`).

## Deploying to Streamlit Cloud
1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app pointing at `app.py`.
3. Add the secrets (same keys as above) under **App Settings → Secrets** in TOML form:
   ```toml
   SARVAM_API_KEY = "sk-..."
   SARVAM_CHAT_MODEL = "bhashini-chat-lite"
   SARVAM_TTS_VOICE = "general-female"
   ```
4. Set **Advanced settings → Enable WebRTC** (needed for microphone capture).
5. Deploy. Streamlit Cloud will install packages from `requirements.txt` automatically.

## Extensibility Ideas
- Real-time incremental transcription by draining frames on a background thread.
- Add wake-word detection or voice activity detection before shipping audio to Sarvam.
- Multi-voice playback or language auto-detection.
- Persist chat transcripts to a database (e.g., Supabase) or export per session.

## Troubleshooting
- **No audio captured**: ensure HTTPS + a valid certificate (Streamlit Cloud handles this) and that the browser granted microphone permissions.
- **API errors**: check Streamlit logs for the HTTP response. Most helper methods raise `SarvamAPIError` with the original body.
- **Latency**: experiment with shorter utterances, smaller sample windows, or chunk streaming.

Happy building! 🎙️
