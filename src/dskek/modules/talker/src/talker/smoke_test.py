"""Offline end-to-end check of the voice pipeline: no Discord, no network, no real LLM.

Kokoro synthesizes a "user" saying something -> streaming ASR transcribes it ->
a scripted fake LLM answers -> Kokoro speaks the answer into the playback source,
which a thread drains at real-time speed (like discord.py's player thread).
Then a second "user" utterance interrupts the answer to exercise barge-in.

    VOICE_ASR_* / VOICE_TTS_* env vars as in the README, then:
    python scripts/smoke_test.py
"""

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Sequence
from typing import override

import numpy as np

from talker.audio import (
    DISCORD_SAMPLE_RATE,
    FRAME_BYTES,
    StreamingPcmSource,
    f32_mono_to_discord_pcm,
    pcm_stereo_s16_to_mono_f32,
)
from talker.config import VoiceChatSettings
from talker.engine import VoiceEngine
from talker.models import ChatMessage, FloatAudio, Role
from talker.protocols import ChatModel
from talker.session import VoiceChatSession

REPLY = (
    "The capital of France is Paris, and it sits on the river Seine. "
    "It is famous for the Eiffel Tower, the Louvre museum and its cafes. "
    "People have lived there for more than two thousand years. "
    "Would you like to hear more about the city?"
)
# REPLY = """Столица Франции - Париж, который расположен на реке Сена. Он знаменит Эйфелевой башней, Лувром и своими кафе. Люди живут там уже больше двух тысяч лет. Хотели бы вы узнать больше об этом городе?"""


class ScriptedChat(ChatModel):
    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []

    @override
    async def stream_reply(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        for word in REPLY.split(" "):
            await asyncio.sleep(0.01)  # pretend token latency
            yield word + " "


def speech_frames(engine: VoiceEngine, text: str) -> list[FloatAudio]:
    """Render ``text`` with the TTS and cut it into 20 ms / 48 kHz mono frames like Discord delivers."""
    chunks: list[FloatAudio] = []
    engine.synthesizer.synthesize(text, lambda c: chunks.append(c) is None)
    pcm = f32_mono_to_discord_pcm(np.concatenate(chunks), engine.synthesizer.sample_rate)
    mono = pcm_stereo_s16_to_mono_f32(pcm)
    step = DISCORD_SAMPLE_RATE // 50
    return [mono[i : i + step] for i in range(0, len(mono) - step, step)]


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    settings = VoiceChatSettings.from_env()
    t0 = time.perf_counter()
    engine = await VoiceEngine.load(settings)
    print(f"models loaded in {time.perf_counter() - t0:.1f}s, TTS rate {engine.synthesizer.sample_rate} Hz")

    chat = ScriptedChat()
    engine.chat = chat
    source = StreamingPcmSource(prebuffer_ms=settings.turn.prebuffer_ms)

    # stand-in for discord.py's AudioPlayer thread: pull one 20 ms frame every 20 ms
    stop = threading.Event()
    audible_frames = 0

    def player() -> None:
        nonlocal audible_frames
        next_tick = time.perf_counter()
        while not stop.is_set():
            if source.read().count(0) != FRAME_BYTES:
                audible_frames += 1
            next_tick += 0.02
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    threading.Thread(target=player, daemon=True).start()

    loop = asyncio.get_running_loop()
    session = VoiceChatSession(engine=engine, source=source, loop=loop)

    question = speech_frames(engine, "What is the capital of France?")
    # question = speech_frames(engine, "Какая столица Франции?")
    print(f"user speech: {len(question) * 0.02:.1f}s")
    t_speech_end = time.perf_counter()
    for frame in question:
        session.push_audio(1, "Alice", frame)

    # wait for first audio out
    deadline = time.perf_counter() + 30
    while audible_frames == 0 and time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
    assert audible_frames > 0, "no audio was produced"
    print(f"time from end of speech to first audible frame: {time.perf_counter() - t_speech_end:.2f}s "
          f"(includes {settings.turn.gap_finalize_s}s end-of-speech gap)")
    assert chat.calls, "LLM was never called"
    print("LLM saw:", [(m.role.value, m.content) for m in chat.calls[0][1:]])

    await asyncio.sleep(2.5)  # let a couple of sentences play
    print("--- user barges in ---")
    for frame in speech_frames(engine, "Wait, stop please, I have another question."):
        session.push_audio(1, "Alice", frame)
    await asyncio.sleep(1.5)
    assert not source.is_active or source.fed_bytes == source.played_bytes, "playback was not cut"

    # let the second answer start so the history is committed with the interruption
    deadline = time.perf_counter() + 30
    while len(chat.calls) < 2 and time.perf_counter() < deadline:
        await asyncio.sleep(0.05)
    assert len(chat.calls) == 2, "second utterance never reached the LLM"
    second = chat.calls[1]
    print("second LLM call history:")
    for m in second[1:]:
        print(f"  [{m.role.value}] {m.content}")
    assistant = [m for m in second if m.role is Role.ASSISTANT]
    assert assistant, "interrupted reply missing from history"
    assert len(assistant[0].content) < len(REPLY), "history kept text the user never heard"
    print("OK: barge-in trimmed the history to what was actually spoken")

    stop.set()
    await session.close()
    engine.close()


if __name__ == "__main__":
    asyncio.run(main())
