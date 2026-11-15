"""Swappable backends. Implement these to plug in a different ASR/TTS/LLM."""

from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Protocol

from .models import ChatMessage, FloatAudio, LlmEvent, RecognitionUpdate

if TYPE_CHECKING:
    from .tools import ToolSpec


class RecognitionStream(Protocol):
    """One speaker's live recognition state. Called from a single worker thread."""

    def feed(self, samples: FloatAudio, sample_rate: int) -> RecognitionUpdate: ...

    def finalize(self) -> str:
        """Flush, return the final transcript and get ready for the next utterance."""
        ...


class SpeechRecognizer(Protocol):
    """Streaming recognizer: cheap partial results while the user speaks (used for barge-in)."""

    def open_stream(self) -> RecognitionStream: ...


class FinalRecognizer(Protocol):
    """Accurate offline recognizer run once per finished utterance (second pass)."""

    def transcribe(self, samples: FloatAudio, sample_rate: int) -> str:
        """Blocking."""
        ...


class SpeechSynthesizer(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def synthesize(self, text: str, on_chunk: Callable[[FloatAudio], bool]) -> None:
        """Blocking. Push audio through ``on_chunk``; stop as soon as it returns False."""
        ...


class ChatModel(Protocol):
    def stream_reply(
        self, messages: Sequence[ChatMessage], tools: Sequence["ToolSpec"] = ()
    ) -> AsyncIterator[LlmEvent]:
        """Yield text deltas, tool-argument fragments, and finally a ``ReplyComplete``."""
        ...
