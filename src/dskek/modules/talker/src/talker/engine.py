"""Loaded models, tools and worker threads, shared by every guild session."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Self

from .config import VoiceChatSettings
from .llm import OpenAiCompatChat
from .mcp_tools import McpToolbox
from .protocols import ChatModel, FinalRecognizer, SpeechRecognizer, SpeechSynthesizer
from .stt import SherpaFinalRecognizer, SherpaStreamingRecognizer
from .tools import ToolRegistry
from .tools_builtin import register_builtin_tools
from .tts import SherpaSynthesizer


@dataclass(slots=True, kw_only=True)
class VoiceEngine:
    settings: VoiceChatSettings
    recognizer: SpeechRecognizer
    final_recognizer: FinalRecognizer | None
    synthesizer: SpeechSynthesizer
    chat: ChatModel
    tools: ToolRegistry
    mcp: McpToolbox
    stt_pool: ThreadPoolExecutor
    """Single worker: the recognizer is not re-entrant, and CPU cores are better spent per model."""
    final_pool: ThreadPoolExecutor
    tts_pool: ThreadPoolExecutor

    @classmethod
    async def load(cls, settings: VoiceChatSettings) -> Self:
        """Load models and connect MCP servers (takes a few seconds)."""
        loaders = [
            asyncio.to_thread(SherpaStreamingRecognizer, settings.asr),
            asyncio.to_thread(SherpaSynthesizer, settings.tts),
        ]
        if settings.final_asr is not None:
            loaders.append(asyncio.to_thread(SherpaFinalRecognizer, settings.final_asr))
        loaded = await asyncio.gather(*loaders)
        recognizer, synthesizer = loaded[0], loaded[1]
        final = loaded[2] if settings.final_asr is not None else None

        registry = ToolRegistry(settings.tools)
        register_builtin_tools(registry)
        mcp = McpToolbox(settings.tools.mcp_servers, registry)
        if settings.tools.enabled:
            await mcp.start()

        return cls(
            settings=settings,
            recognizer=recognizer,  # type: ignore[arg-type]  # gather() widens the element type
            final_recognizer=final,  # type: ignore[arg-type]
            synthesizer=synthesizer,  # type: ignore[arg-type]
            chat=OpenAiCompatChat(settings.llm),
            tools=registry,
            mcp=mcp,
            stt_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-stt"),
            final_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-final"),
            tts_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-tts"),
        )

    async def aclose(self) -> None:
        await self.mcp.close()
        for pool in (self.stt_pool, self.final_pool, self.tts_pool):
            pool.shutdown(wait=False, cancel_futures=True)
        close = getattr(self.chat, "aclose", None)
        if close is not None:
            await close()
