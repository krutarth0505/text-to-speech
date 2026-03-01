"""Audio helper utilities for Streamlit WebRTC workflows."""
from __future__ import annotations

import io
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import soundfile as sf


def frames_to_wav_bytes(frames: Sequence[np.ndarray], sample_rate: int) -> bytes:
    """Concatenate audio frames and return WAV bytes."""
    if not frames:
        raise ValueError("No audio frames provided for conversion.")

    audio = np.concatenate(frames, axis=0)
    # streamlit-webrtc gives float32 in [-1, 1]. Convert to int16 for WAV.
    if not np.issubdtype(audio.dtype, np.integer):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)

    with io.BytesIO() as buffer:
        sf.write(buffer, audio, sample_rate, format="wav")
        buffer.seek(0)
        return buffer.read()


def rms(audio: np.ndarray) -> float:
    """Return root-mean-square amplitude for gating logic."""
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def trim_silence(frames: Sequence[np.ndarray], threshold: float = 0.01) -> List[np.ndarray]:
    """Drop silent frames by RMS threshold to reduce payload size."""
    return [frame for frame in frames if rms(frame) > threshold]
