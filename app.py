from __future__ import annotations

import io
import os
import queue
import wave
from typing import Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
import numpy as np
from streamlit_webrtc import AudioProcessorBase, RTCConfiguration, WebRtcMode, webrtc_streamer

from assistant_logic import AssistantBrain
from audio_utils import frames_to_wav_bytes, trim_silence
from sarvam_client import SarvamAPIError, SarvamClient

load_dotenv()

st.set_page_config(page_title="Live Speech Assistant", page_icon="🎙️", layout="wide")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class FrameBuffer(AudioProcessorBase):
    """Collect audio frames coming from streamlit-webrtc."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.sample_rate: Optional[int] = None

    def recv_audio(self, frame):  # type: ignore[override]
        audio = frame.to_ndarray().copy()
        self.sample_rate = frame.sample_rate
        self._queue.put(audio)
        return frame

    def flush(self) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        while not self._queue.empty():
            frames.append(self._queue.get())
        return frames


def ensure_state() -> None:
    if "chat_timeline" not in st.session_state:
        st.session_state.chat_timeline: List[Dict[str, object]] = []
    if "sarvam_client" not in st.session_state:
        try:
            st.session_state.sarvam_client = SarvamClient()
        except ValueError:
            st.error("Missing SARVAM_API_KEY. Create a .env file or set the environment variable before running.")
            st.stop()
    if "assistant" not in st.session_state:
        st.session_state.assistant = AssistantBrain(st.session_state.sarvam_client)
    if "voice_only_mode" not in st.session_state:
        st.session_state.voice_only_mode = False


def append_message(
    role: str,
    content: str,
    *,
    audio_bytes: Optional[bytes] = None,
    meta: Optional[str] = None,
    hide_when_voice_only: bool = False,
) -> None:
    st.session_state.chat_timeline.append(
        {
            "role": role,
            "content": content,
            "audio_bytes": audio_bytes,
            "meta": meta,
            "hide_when_voice_only": hide_when_voice_only,
        }
    )


def process_user_turn(transcript: str, source: str) -> None:
    assistant: AssistantBrain = st.session_state.assistant
    sarvam_client: SarvamClient = st.session_state.sarvam_client
    voice_only_mode: bool = st.session_state.voice_only_mode
    hide_text = voice_only_mode and source == "microphone"

    append_message("user", transcript, meta=f"Source: {source}", hide_when_voice_only=hide_text)
    reply = assistant.reply(transcript)

    tts_voice = os.getenv("SARVAM_TTS_VOICE")
    try:
        audio_bytes = sarvam_client.synthesize_speech(reply, voice=tts_voice)
    except SarvamAPIError as exc:
        st.warning(f"TTS failed: {exc}")
        audio_bytes = None

    append_message(
        "assistant",
        reply,
        audio_bytes=audio_bytes,
        meta=f"Voice: {tts_voice or 'default'}",
        hide_when_voice_only=hide_text,
    )


def transcribe_and_process(
    wav_bytes: bytes,
    sample_rate: int,
    language_selection: str,
    *,
    source: str,
) -> None:
    sarvam_client: SarvamClient = st.session_state.sarvam_client
    language_param = None if language_selection == "auto" else language_selection
    try:
        transcript = sarvam_client.transcribe_audio(
            wav_bytes,
            sample_rate=sample_rate,
            language=language_param,
        )
    except SarvamAPIError as exc:
        st.error(f"Sarvam API error: {exc}")
        return
    except ValueError as exc:
        st.error(str(exc))
        return
    process_user_turn(transcript, source=source)


def extract_wav_from_audio_input(audio_buffer) -> Tuple[bytes, int]:  # type: ignore[override]
    if audio_buffer is None:
        raise ValueError("No audio buffer provided.")
    if hasattr(audio_buffer, "to_wav_bytes"):
        wav_bytes = audio_buffer.to_wav_bytes()
        sample_rate = getattr(audio_buffer, "sample_rate", 16000)
        return wav_bytes, int(sample_rate or 16000)
    if hasattr(audio_buffer, "getvalue"):
        data = audio_buffer.getvalue()
    elif hasattr(audio_buffer, "read"):
        data = audio_buffer.read()
    else:
        raise ValueError("Unsupported audio buffer type returned by Streamlit.")
    return data, _infer_wav_sample_rate(data)


def _infer_wav_sample_rate(wav_bytes: bytes) -> int:
    try:
        with wave.open(io.BytesIO(wav_bytes)) as wav_file:
            return wav_file.getframerate()
    except wave.Error:
        return 16000


ensure_state()
st.title("🎙️ Live Speech AI Assistant")
st.caption("Record speech, transcribe via Sarvam AI, auto-reply, and hear the response back.")

col_mic, col_actions = st.columns([2, 1])
with col_mic:
    st.subheader("Microphone Stream")
    ctx = webrtc_streamer(
        key="sarvam-audio",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=FrameBuffer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"audio": True, "video": False},
    )

with col_actions:
    st.subheader("Controls")
    st.write("Capture a short utterance, then click below to process it.")
    use_vad = st.checkbox("Trim silence before sending", value=True)
    language = st.selectbox("STT language", options=["auto", "en", "hi", "ta"], index=0)
    voice_only_toggle = st.toggle(
        "Speech-to-speech mode",
        value=st.session_state.voice_only_mode,
        help="Hide transcripts and disable text input so the exchange stays fully audio.",
    )
    if voice_only_toggle != st.session_state.voice_only_mode:
        st.session_state.voice_only_mode = voice_only_toggle
        st.experimental_rerun()
    if st.session_state.voice_only_mode:
        st.caption("Voice-only mode active: typed chat is disabled and transcripts stay hidden.")

    capture_disabled = not ctx or not ctx.state.playing or ctx.audio_processor is None
    if st.button("Capture & Respond", disabled=capture_disabled):
        processor: FrameBuffer = ctx.audio_processor  # type: ignore[assignment]
        frames = processor.flush()
        if not frames:
            st.warning("No audio captured yet. Keep speaking and try again.")
        else:
            if use_vad:
                frames = trim_silence(frames)
            if not frames:
                st.warning("Captured audio was silent. Try speaking closer to the mic or disable trimming.")
            else:
                sample_rate = processor.sample_rate or 16000
                try:
                    wav_bytes = frames_to_wav_bytes(frames, sample_rate)
                    transcribe_and_process(wav_bytes, sample_rate, language, source="microphone")
                except ValueError as exc:
                    st.error(str(exc))

    st.divider()
    st.markdown("#### Alternate input options")
    recorded_audio = st.audio_input(
        "Record a quick clip",
        help="Browser-native recorder. After capturing, click the button below to send it.",
    )
    if st.button("Respond to recording", disabled=recorded_audio is None):
        try:
            wav_bytes, sample_rate = extract_wav_from_audio_input(recorded_audio)
            transcribe_and_process(wav_bytes, sample_rate, language, source="recorder")
        except ValueError as exc:
            st.error(str(exc))

    uploaded_wav = st.file_uploader(
        "Upload a WAV clip",
        type=["wav"],
        help="Use this if WebRTC or the recorder is unavailable.",
    )
    if st.button("Respond to upload", disabled=uploaded_wav is None):
        if uploaded_wav:
            wav_bytes = uploaded_wav.getvalue()
            sample_rate = _infer_wav_sample_rate(wav_bytes)
            transcribe_and_process(wav_bytes, sample_rate, language, source="upload")

chat_disabled = st.session_state.voice_only_mode
if user_text := st.chat_input("Or type a message", disabled=chat_disabled):
    process_user_turn(user_text, source="typed")

col_history, col_settings = st.columns([3, 1])
with col_history:
    st.subheader("Conversation")
    if st.session_state.voice_only_mode:
        st.info("Speech-to-speech mode hides message text. Use the audio players to review turns.")
    for message in st.session_state.chat_timeline:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            hide_text = st.session_state.voice_only_mode and message.get("hide_when_voice_only")
            if hide_text:
                st.caption("Speech turn hidden in voice-only mode.")
            else:
                st.markdown(message["content"])
            if message.get("meta"):
                st.caption(message["meta"])
            audio_bytes = message.get("audio_bytes")
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")

with col_settings:
    st.subheader("Session")
    if st.button("Clear conversation"):
        st.session_state.chat_timeline = []
        st.session_state.assistant.clear_history()
        st.experimental_rerun()

    st.markdown("**Environment diagnostics**")
    st.write(
        {
            "SARVAM_CHAT_MODEL": os.getenv("SARVAM_CHAT_MODEL", "bhashini-chat-lite"),
            "SARVAM_TTS_VOICE": os.getenv("SARVAM_TTS_VOICE", "default"),
            "Language": language if "language" in locals() else os.getenv("SARVAM_STT_LANGUAGE", "auto"),
        }
    )
