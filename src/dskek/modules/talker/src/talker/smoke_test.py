"""End-to-end check with the REAL ASR/TTS models (no Discord, no real LLM).

Kokoro speaks as the "user" -> streaming ASR + Parakeet second pass -> wake-word gating -> scripted LLM
(tool call, then an answer) -> Kokoro speaks into the playback source, drained at real-time speed ->
a second utterance interrupts the answer (barge-in).

    export the VOICE_ASR_* / VOICE_FINAL_ASR_* / VOICE_TTS_* variables (see README), VOICE_BOT_NAMES=nova
    python scripts/smoke_test.py
"""

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Sequence
from typing import override

import numpy as np

from dskek.voice_chat.audio import (
    DISCORD_SAMPLE_RATE, FRAME_BYTES, StreamingPcmSource, f32_mono_to_discord_pcm, pcm_stereo_s16_to_mono_f32,
)
from dskek.voice_chat.config import VoiceChatSettings
from dskek.voice_chat.engine import VoiceEngine
from dskek.voice_chat.models import ChatMessage, FloatAudio, LlmEvent, ReplyComplete, Role, TextDelta, ToolCall
from dskek.voice_chat.protocols import ChatModel
from dskek.voice_chat.session import VoiceChatSession
from dskek.voice_chat.tools import ToolSpec

LONG_REPLY = (
    "It is a quiet Saturday afternoon right now. "
    "That is the time on the server where I run. "
    "Weekends are usually calm here, with few requests. "
    "Would you like to hear anything else about it?"
)


class ScriptedChat(ChatModel):
    def __init__(self) -> None:
        self.calls: list[tuple[list[ChatMessage], list[str]]] = []

    @override
    async def stream_reply(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()) -> AsyncIterator[LlmEvent]:
        self.calls.append((list(messages), [t.name for t in tools]))
        if len(self.calls) == 1:  # first turn: ask for the time, then answer
            yield TextDelta("Let me look. ")
            yield ReplyComplete((ToolCall("c1", "get_time", "{}"),))
        else:
            for word in (LONG_REPLY if len(self.calls) == 2 else "Sure, go ahead.").split(" "):
                await asyncio.sleep(0.01)
                yield TextDelta(word + " ")
            yield ReplyComplete(())


def speech_frames(engine: VoiceEngine, text: str) -> list[FloatAudio]:
    chunks: list[FloatAudio] = []
    engine.synthesizer.synthesize(text, lambda c: chunks.append(c) is None)
    mono = pcm_stereo_s16_to_mono_f32(f32_mono_to_discord_pcm(np.concatenate(chunks), engine.synthesizer.sample_rate))
    step = DISCORD_SAMPLE_RATE // 50
    return [mono[i : i + step] for i in range(0, len(mono) - step, step)]


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    settings = VoiceChatSettings.from_env()
    t0 = time.perf_counter()
    engine = await VoiceEngine.load(settings)
    print(f"models loaded in {time.perf_counter() - t0:.1f}s | second pass: {'yes' if engine.final_recognizer else 'no'} "
          f"| bot name: {settings.wake.names}")
    chat = ScriptedChat()
    engine.chat = chat
    source = StreamingPcmSource(prebuffer_ms=settings.turn.prebuffer_ms)

    stop, audible = threading.Event(), 0

    def player() -> None:  # stand-in for discord.py's AudioPlayer thread
        nonlocal audible
        tick = time.perf_counter()
        while not stop.is_set():
            audible += source.read().count(0) != FRAME_BYTES
            tick += 0.02
            time.sleep(max(0.0, tick - time.perf_counter()))

    threading.Thread(target=player, daemon=True).start()
    session = VoiceChatSession(engine=engine, source=source, loop=asyncio.get_running_loop())

    async def user_says(text: str) -> None:
        frames = speech_frames(engine, text)
        for frame in frames:
            session.push_audio(1, "Alice", frame)
        await asyncio.sleep(0.2)

    # 1) not addressed to the bot: must be ignored
    await user_says("What is the capital of France?")
    await asyncio.sleep(3.0)
    assert not chat.calls, "speech without the bot's name must be ignored"
    print("ok  unaddressed speech ignored")

    # 2) wake word through the real recognizers, then a tool call and a spoken answer
    t_end = time.perf_counter()
    await user_says(f"Hey {settings.wake.names[0]}, what time is it?")
    deadline = time.perf_counter() + 40
    while len(chat.calls) < 2 and time.perf_counter() < deadline:
        await asyncio.sleep(0.05)
    assert len(chat.calls) >= 2, "wake word / tool round trip did not happen"
    user_msg = chat.calls[0][0][-1].content
    tool_msg = chat.calls[1][0][-1]
    print(f"ok  woke up; LLM saw {user_msg!r}; tool result {tool_msg.content!r}")
    assert tool_msg.role is Role.TOOL and "20" in tool_msg.content
    while audible == 0 and time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
    assert audible, "no audio was produced"
    print(f"    end of user speech -> first audible frame: {time.perf_counter() - t_end:.2f}s")

    # 3) barge-in
    await asyncio.sleep(2.0)
    await user_says("Wait, stop, I have another question.")
    deadline = time.perf_counter() + 30
    while len(chat.calls) < 3 and time.perf_counter() < deadline:
        await asyncio.sleep(0.05)
    assert len(chat.calls) == 3, "second utterance never reached the LLM"
    heard = [m for m in chat.calls[2][0] if m.role is Role.ASSISTANT and not m.tool_calls]
    print("    history the LLM sees after the interruption:")
    for m in chat.calls[2][0][1:]:
        print(f"      [{m.role.value}] {m.content[:90]!r}{' +tool_calls' if m.tool_calls else ''}")
    assert heard and len(heard[-1].content) < len(LONG_REPLY), "history kept speech the user never heard"
    print("ok  barge-in trimmed history; tool exchange preserved")

    stop.set()
    await session.close()
    await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
