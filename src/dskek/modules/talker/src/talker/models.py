"""DTOs shared across the voice-chat pipeline (no dicts crossing module boundaries)."""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

type FloatAudio = NDArray[np.float32]
"""Mono float32 samples in [-1.0, 1.0]."""


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Utterance:
    """A finished piece of user speech, already transcribed."""

    speaker_id: int
    speaker_name: str
    text: str


@dataclass(frozen=True, slots=True)
class RecognitionUpdate:
    """Result of feeding one audio chunk to a streaming recognizer."""

    text: str
    """Best partial hypothesis for the utterance so far (may be empty)."""
    is_endpoint: bool
    """True when the recognizer's own endpointing decided the utterance is over."""


@dataclass(slots=True)
class SpokenSentence:
    """A sentence handed to TTS, with its byte span in the playback stream."""

    text: str
    start_byte: int
    """Playback-stream offset where this sentence's audio begins."""


@dataclass(slots=True)
class ResponseTurn:
    """One assistant reply in flight (LLM -> sentences -> TTS -> playback)."""

    messages: list[ChatMessage]
    generation: int
    """Source generation this turn may write into; stale writes are dropped."""
    sentences: list[SpokenSentence] = field(default_factory=list)
    played_limit: int | None = None
    """Bytes of playback consumed when the turn was interrupted (None = ran to the end)."""
