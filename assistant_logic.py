"""Conversation state + fallback heuristics for the assistant."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from sarvam_client import SarvamClient, SarvamAPIError

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, empathetic voice assistant helping with general knowledge, "
    "productivity tips, and casual conversation. Prefer short sentences that sound "
    "good when spoken aloud."
)


@dataclass
class AssistantBrain:
    sarvam_client: SarvamClient
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model_name: str | None = None
    temperature: float = 0.4
    history: List[Dict[str, str]] = field(default_factory=list)

    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def reply(self, user_text: str) -> str:
        messages = self._build_messages(user_text)
        try:
            response = self.sarvam_client.chat_completion(
                messages,
                model=self.model_name,
                temperature=self.temperature,
            )
        except SarvamAPIError:
            response = self._fallback_response(user_text)

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response})
        return response

    @staticmethod
    def _fallback_response(user_text: str) -> str:
        # Extremely light fallback so the UI keeps working if chat endpoint is unavailable.
        return (
            "I heard you say: "
            f"\"{user_text}\". I'm currently offline, but I can still summarize or respond briefly."
        )

    def clear_history(self) -> None:
        self.history.clear()
