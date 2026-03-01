"""Sarvam AI SDK helper utilities."""

from __future__ import annotations

import base64
import os
from typing import Dict, Optional, Sequence

from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError as SarvamAPIError
from sarvamai.environment import SarvamAIEnvironment


_LANGUAGE_SHORT_TO_BCP: Dict[str, str] = {
    "auto": "unknown",
    "unknown": "unknown",
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "pa": "pa-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "od": "od-IN",
    "or": "od-IN",
    "as": "as-IN",
    "ur": "ur-IN",
    "ne": "ne-IN",
    "kok": "kok-IN",
    "ks": "ks-IN",
    "sd": "sd-IN",
    "sa": "sa-IN",
    "sat": "sat-IN",
    "mni": "mni-IN",
    "brx": "brx-IN",
    "mai": "mai-IN",
    "doi": "doi-IN",
}

_VALID_AUDIO_CODECS = {"wav", "mp3", "linear16", "mulaw", "alaw", "opus", "flac", "aac"}


class SarvamClient:
    """Thin wrapper around the official Sarvam SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("Missing SARVAM_API_KEY. Set it in your environment or .env file.")

        environment = self._resolve_environment(base_url or os.getenv("SARVAM_BASE_URL"))
        self.sdk = SarvamAI(
            api_subscription_key=self.api_key,
            environment=environment,
            timeout=timeout_seconds,
        )

    @staticmethod
    def _resolve_environment(custom_base: Optional[str]) -> SarvamAIEnvironment:
        if not custom_base:
            return SarvamAIEnvironment.PRODUCTION
        sanitized = custom_base.rstrip("/")
        return SarvamAIEnvironment(base=sanitized, production="wss://api.sarvam.ai")

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        language: Optional[str] = None,
        response_format: str = "text",
    ) -> str:
        """Send audio to Sarvam STT and return transcript text."""

        stt_model = os.getenv("SARVAM_STT_MODEL", "saarika:v2.5")
        stt_mode = self._resolve_stt_mode(stt_model)
        language_code = self._resolve_language_code(language)

        response = self.sdk.speech_to_text.transcribe(
            file=("audio.wav", audio_bytes, "audio/wav"),
            model=stt_model,
            mode=stt_mode,
            language_code=language_code,
            input_audio_codec="wav",
        )
        return response.transcript

    def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        speaking_rate: float = 1.0,
        audio_format: str = "wav",
    ) -> bytes:
        """Convert assistant text to speech using Sarvam bulbul models."""

        language_code = os.getenv("SARVAM_TTS_LANGUAGE", "en-IN")
        model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
        codec = self._resolve_audio_codec(audio_format)
        sample_rate = self._read_int_env("SARVAM_TTS_SAMPLE_RATE", 24000)
        temperature = self._read_float_env("SARVAM_TTS_TEMPERATURE", 0.6)
        pace = max(0.3, min(3.0, speaking_rate))

        response = self.sdk.text_to_speech.convert(
            text=text,
            target_language_code=language_code,
            speaker=voice or os.getenv("SARVAM_TTS_VOICE"),
            pace=pace,
            model=model,
            speech_sample_rate=sample_rate,
            output_audio_codec=codec,
            temperature=temperature,
        )

        if not response.audios:
            raise SarvamAPIError(body="Sarvam TTS returned no audio payload.")

        return base64.b64decode(response.audios[0])

    def chat_completion(
        self,
        messages: Sequence[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.4,
    ) -> str:
        """Call Sarvam chat completion endpoint with a message list."""

        response = self.sdk.chat.completions(
            messages=list(messages),
            temperature=temperature,
        )
        if not response.choices:
            raise SarvamAPIError(body="Sarvam chat returned no choices.")
        return response.choices[0].message.content

    @staticmethod
    def _resolve_stt_mode(model_name: str) -> Optional[str]:
        requested_mode = os.getenv("SARVAM_STT_MODE")
        if not requested_mode:
            return None
        if "saaras" not in (model_name or ""):
            return None
        return requested_mode

    @staticmethod
    def _resolve_language_code(language: Optional[str]) -> Optional[str]:
        if not language:
            language = os.getenv("SARVAM_STT_LANGUAGE", "unknown")
        normalized = (language or "unknown").strip()
        if not normalized:
            return "unknown"
        normalized_lower = normalized.lower()
        return _LANGUAGE_SHORT_TO_BCP.get(normalized_lower, normalized)

    @staticmethod
    def _resolve_audio_codec(fmt: str) -> str:
        normalized = (fmt or "wav").lower()
        return normalized if normalized in _VALID_AUDIO_CODECS else "wav"

    @staticmethod
    def _read_int_env(var_name: str, default: int) -> int:
        value = os.getenv(var_name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _read_float_env(var_name: str, default: float) -> float:
        value = os.getenv(var_name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default
