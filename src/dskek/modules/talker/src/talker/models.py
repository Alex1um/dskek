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
    TOOL = "tool"


class ToolRisk(StrEnum):
    SAFE = "safe"
    """No side effects outside the conversation: anyone may trigger it."""
    PRIVILEGED = "privileged"
    """Acts on Discord or the outside world: only configured trusted speakers may trigger it."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    """Raw JSON text exactly as the model produced it."""
    index: int = 0
    """Position in the model's streamed tool-call list (links streamed fragments to the final call)."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    """Only on assistant messages."""
    tool_call_id: str | None = None
    """Only on tool messages: which call this is the result of."""


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


# ---- events produced while the LLM streams a reply ------------------------------------
@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolArgsDelta:
    """A fragment of a tool call's JSON arguments (lets ``say`` start speaking early)."""

    index: int
    name: str
    fragment: str


@dataclass(frozen=True, slots=True)
class ReplyComplete:
    tool_calls: tuple[ToolCall, ...]


type LlmEvent = TextDelta | ToolArgsDelta | ReplyComplete


# ---- one assistant reply in flight ----------------------------------------------------
@dataclass(slots=True)
class SpokenSentence:
    """A sentence handed to TTS, with its byte span in the playback stream."""

    text: str
    start_byte: int
    """Playback-stream offset where this sentence's audio begins."""


@dataclass(slots=True)
class Round:
    """One LLM call within a turn. A turn has several rounds when tools are used."""

    sentences: list[SpokenSentence] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)
    """Completed non-``say`` tool calls (only recorded together with their results)."""
    results: list[ChatMessage] = field(default_factory=list)


@dataclass(slots=True)
class ResponseTurn:
    """One assistant reply in flight (LLM -> sentences -> TTS -> playback)."""

    messages: list[ChatMessage]
    generation: int
    """Source generation this turn may write into; stale writes are dropped."""
    speaker_id: int | None = None
    speaker_name: str = ""
    fixed: bool = False
    """A canned phrase (greeting, acknowledgement): spoken, but not added to the chat history."""
    rounds: list[Round] = field(default_factory=list)
    played_limit: int | None = None
    """Bytes of playback consumed when the turn was interrupted (None = ran to the end)."""
