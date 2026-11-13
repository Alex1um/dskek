"""Loaded models + worker threads, shared by every guild session."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Self

from .config import VoiceChatSettings
from .llm import OpenAiCompatChat
from .protocols import ChatModel, SpeechRecognizer, SpeechSynthesizer
from .stt import SherpaStreamingRecognizer
from .tts import SherpaSynthesizer


@dataclass(slots=True, kw_only=True)
class VoiceEngine:
    settings: VoiceChatSettings
    recognizer: SpeechRecognizer
    synthesizer: SpeechSynthesizer
    chat: ChatModel
    stt_pool: ThreadPoolExecutor
    """Single worker: the recognizer is not re-entrant, and CPU cores are better spent per model."""
    tts_pool: ThreadPoolExecutor

    @classmethod
    async def load(cls, settings: VoiceChatSettings) -> Self:
        """Load models off the event loop (takes a few seconds)."""
        recognizer, synthesizer = await asyncio.gather(
            asyncio.to_thread(SherpaStreamingRecognizer, settings.asr),
            asyncio.to_thread(SherpaSynthesizer, settings.tts),
        )
        return cls(
            settings=settings,
            recognizer=recognizer,
            synthesizer=synthesizer,
            chat=OpenAiCompatChat(settings.llm),
            stt_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-stt"),
            tts_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-tts"),
        )

    def close(self) -> None:
        self.stt_pool.shutdown(wait=False, cancel_futures=True)
        self.tts_pool.shutdown(wait=False, cancel_futures=True)
