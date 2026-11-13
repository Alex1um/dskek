"""LLM streaming client plus the text plumbing between LLM tokens and TTS."""

import re
from collections.abc import AsyncIterator, Sequence
from typing import override

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from .config import LlmConfig
from .models import ChatMessage, Role
from .protocols import ChatModel


def _to_param(message: ChatMessage) -> ChatCompletionMessageParam:
    match message.role:
        case Role.SYSTEM:
            return ChatCompletionSystemMessageParam(role="system", content=message.content)
        case Role.USER:
            return ChatCompletionUserMessageParam(role="user", content=message.content)
        case Role.ASSISTANT:
            return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)


class OpenAiCompatChat(ChatModel):
    """Streams from any OpenAI-compatible server (llama.cpp ``llama-server``, Ollama, ...)."""

    def __init__(self, config: LlmConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key, timeout=config.timeout_s)

    @override
    async def stream_reply(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        cfg = self._config
        extra_body_tools = {
            "enable_tools": True,
            "enabled_tools": ["web_search", "python", "terminal"],
            "permission_mode": "off",
        }
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if cfg.disable_thinking else None
        if extra_body:
            extra_body_tools |= extra_body
        stream = await self._client.chat.completions.create(
            model=cfg.model,
            messages=[_to_param(m) for m in messages],
            stream=True,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            extra_body=extra_body_tools,
        )
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        finally:
            await stream.close()  # stops generation server-side when we get cancelled

    async def aclose(self) -> None:
        await self._client.close()


# ---------------------------------------------------------------------------------------
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _partial_suffix(text: str, tag: str) -> str:
    """Longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-n:]):
            return text[-n:]
    return ""


class ThinkFilter:
    """Removes ``<think>...</think>`` blocks from a token stream (tags may be split across deltas)."""

    def __init__(self) -> None:
        self._inside = False
        self._held = ""

    def feed(self, text: str) -> str:
        buffer = self._held + text
        self._held = ""
        out: list[str] = []
        while buffer:
            if self._inside:
                end = buffer.find(_THINK_CLOSE)
                if end == -1:
                    self._held = _partial_suffix(buffer, _THINK_CLOSE)
                    break
                buffer = buffer[end + len(_THINK_CLOSE) :]
                self._inside = False
            else:
                start = buffer.find(_THINK_OPEN)
                if start == -1:
                    keep = _partial_suffix(buffer, _THINK_OPEN)
                    out.append(buffer[: len(buffer) - len(keep)])
                    self._held = keep
                    break
                out.append(buffer[:start])
                buffer = buffer[start + len(_THINK_OPEN) :]
                self._inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._inside else self._held
        self._held = ""
        return rest


# ---------------------------------------------------------------------------------------
_HARD_BOUNDARY = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n+")
_SOFT_BOUNDARY = re.compile(r"[,;:—–]\s+")
_URL = re.compile(r"https?://\S+")
_SNAKE = re.compile(r"(?<=\w)_(?=\w)")
_MARKUP = re.compile(r"[*_`#>~|]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])")
_EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f\u200d]")
_SPACES = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    text = _URL.sub(" link ", text)
    text = _EMOJI.sub("", text)
    text = _MARKUP.sub("", _SNAKE.sub(" ", text))
    text = _SPACES.sub(" ", text).strip()
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


class SentenceSplitter:
    """Turns a token stream into speakable chunks as early as is safe.

    The first chunk may end at a comma to cut time-to-first-audio; later chunks
    end at sentence boundaries. Over-long runs are split at a space.
    """

    def __init__(self, *, first_chunk_min_chars: int = 30, max_chars: int = 280) -> None:
        self._first_min = first_chunk_min_chars
        self._max = max_chars
        self._buffer = ""
        self._emitted = False

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        out: list[str] = []
        while (piece := self._take()) is not None:
            if cleaned := self._finish(piece):
                out.append(cleaned)
        return out

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer, ""
        cleaned = self._finish(rest)
        return [cleaned] if cleaned else []

    def _take(self) -> str | None:
        if match := _HARD_BOUNDARY.search(self._buffer):
            return self._cut(match.end())
        if not self._emitted:
            for match in _SOFT_BOUNDARY.finditer(self._buffer):
                if match.end() >= self._first_min:
                    return self._cut(match.end())
        if len(self._buffer) > self._max:
            space = self._buffer.rfind(" ", 0, self._max)
            return self._cut(space + 1 if space > 0 else self._max)
        return None

    def _cut(self, index: int) -> str:
        piece, self._buffer = self._buffer[:index], self._buffer[index:]
        return piece

    def _finish(self, piece: str) -> str:
        cleaned = clean_for_speech(piece)
        if not any(ch.isalnum() for ch in cleaned):
            return ""
        self._emitted = True
        return cleaned
