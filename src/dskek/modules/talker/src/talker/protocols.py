"""Swappable backends. Implement these to plug in a different ASR/TTS/LLM."""

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Protocol

from .models import ChatMessage, FloatAudio, RecognitionUpdate


class RecognitionStream(Protocol):
    """One speaker's live recognition state. Called from a single worker thread."""

    def feed(self, samples: FloatAudio, sample_rate: int) -> RecognitionUpdate: ...

    def finalize(self) -> str:
        """Flush, return the final transcript and get ready for the next utterance."""
        ...


class SpeechRecognizer(Protocol):
    def open_stream(self) -> RecognitionStream: ...


class SpeechSynthesizer(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def synthesize(self, text: str, on_chunk: Callable[[FloatAudio], bool]) -> None:
        """Blocking. Push audio through ``on_chunk``; stop as soon as it returns False."""
        ...


class ChatModel(Protocol):
    def stream_reply(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Yield the assistant reply as text deltas."""
        ...
