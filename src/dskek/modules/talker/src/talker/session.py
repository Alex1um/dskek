"""One live voice conversation: listening, turn-taking, barge-in, reply streaming."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np

from .audio import DISCORD_SAMPLE_RATE, StreamingPcmSource, f32_mono_to_discord_pcm
from .engine import VoiceEngine
from .llm import SentenceSplitter, ThinkFilter
from .models import ChatMessage, FloatAudio, ResponseTurn, Role, SpokenSentence, Utterance
from .protocols import RecognitionStream

logger = logging.getLogger("dskek.voice_chat")

type Notifier = Callable[[str], Awaitable[None]]

_MAX_QUEUED_FRAMES = 500  # ~10 s of 20 ms frames per speaker before we start dropping


@dataclass(slots=True)
class _Listener:
    user_id: int
    name: str
    frames: asyncio.Queue[FloatAudio] = field(default_factory=lambda: asyncio.Queue(maxsize=_MAX_QUEUED_FRAMES))
    task: asyncio.Task[None] | None = None


def _normalize_transcript(text: str) -> str:
    text = text.strip()
    if text.isupper():  # many English streaming models emit ALL CAPS
        text = text.lower()
    return text[:1].upper() + text[1:]


class VoiceChatSession:
    """Everything that happens between one bot and one voice channel.

    Threading model: Discord's receive thread calls :meth:`push_audio`; everything else
    runs on the event loop. ASR runs on ``engine.stt_pool``; TTS on ``engine.tts_pool``
    and writes straight into the thread-safe playback source.
    """

    def __init__(
        self,
        *,
        engine: VoiceEngine,
        source: StreamingPcmSource,
        loop: asyncio.AbstractEventLoop,
        notify: Notifier | None = None,
    ) -> None:
        self._engine = engine
        self._settings = engine.settings
        self._source = source
        self._loop = loop
        self._notify = notify
        self._listeners: dict[int, _Listener] = {}
        self._history: list[ChatMessage] = [ChatMessage(Role.SYSTEM, self._settings.llm.system_prompt)]
        self._turn: ResponseTurn | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._turn_lock = asyncio.Lock()
        self._closed = False

    # ---- public API -------------------------------------------------------------------
    def accepts(self, user_id: int) -> bool:
        allowed = self._settings.allowed_user_ids
        return not allowed or user_id in allowed

    def push_audio(self, user_id: int, name: str, samples: FloatAudio) -> None:
        """Thread-safe entry point for 20 ms mono frames at 48 kHz."""
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue_frame, user_id, name, samples)
        except RuntimeError:  # loop already closed during shutdown
            pass

    def drop_speaker(self, user_id: int) -> None:
        """Thread-safe: forget a speaker (e.g. when they leave the channel)."""
        try:
            self._loop.call_soon_threadsafe(self._remove_listener, user_id)
        except RuntimeError:
            pass

    async def say(self, text: str) -> None:
        """Speak a fixed phrase (no LLM)."""
        async with self._turn_lock:
            await self._interrupt_locked()
            self._start_turn(fixed_text=text)

    async def interrupt(self) -> None:
        async with self._turn_lock:
            await self._interrupt_locked()

    def reset_history(self) -> None:
        self._history = self._history[:1]

    async def close(self) -> None:
        self._closed = True
        for listener in list(self._listeners.values()):
            if listener.task is not None:
                listener.task.cancel()
        self._listeners.clear()
        await self.interrupt()
        self._source.clear()

    # ---- listening --------------------------------------------------------------------
    def _enqueue_frame(self, user_id: int, name: str, samples: FloatAudio) -> None:
        if self._closed:
            return
        listener = self._listeners.get(user_id)
        if listener is None:
            listener = _Listener(user_id=user_id, name=name)
            listener.task = asyncio.create_task(self._listen_loop(listener), name=f"voice-listen-{user_id}")
            self._listeners[user_id] = listener
        listener.name = name
        if listener.frames.full():  # recognizer is behind real time: drop the oldest audio
            listener.frames.get_nowait()
        listener.frames.put_nowait(samples)

    def _remove_listener(self, user_id: int) -> None:
        listener = self._listeners.pop(user_id, None)
        if listener is not None and listener.task is not None:
            listener.task.cancel()

    async def _listen_loop(self, listener: _Listener) -> None:
        loop = asyncio.get_running_loop()
        pool = self._engine.stt_pool
        turn_cfg = self._settings.turn
        stream: RecognitionStream = await loop.run_in_executor(pool, self._engine.recognizer.open_stream)

        chunk_samples = DISCORD_SAMPLE_RATE * turn_cfg.chunk_ms // 1000
        pending: list[FloatAudio] = []
        pending_samples = 0
        active = False  # audio received since the last finished utterance

        while True:
            frame: FloatAudio | None = None
            try:
                async with asyncio.timeout(turn_cfg.gap_finalize_s):
                    frame = await listener.frames.get()
            except TimeoutError:
                pass  # Discord sends nothing while the user is silent: treat the gap as end of speech

            try:
                if frame is not None:
                    pending.append(frame)
                    pending_samples += len(frame)
                    if pending_samples < chunk_samples:
                        continue

                if pending:
                    samples = np.concatenate(pending)
                    pending.clear()
                    pending_samples = 0
                    active = True
                    update = await loop.run_in_executor(pool, stream.feed, samples, DISCORD_SAMPLE_RATE)

                    if len(update.text) >= turn_cfg.barge_in_min_chars and self._is_busy():
                        logger.info("barge-in by %s: %r", listener.name, update.text)
                        await self.interrupt()
                    if update.is_endpoint:
                        await self._finish_utterance(listener, stream)
                        active = False
                        continue

                if frame is None and active:
                    await self._finish_utterance(listener, stream)
                    active = False
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("listener for %s failed; continuing", listener.name)

    async def _finish_utterance(self, listener: _Listener, stream: RecognitionStream) -> None:
        text = _normalize_transcript(await self._loop.run_in_executor(self._engine.stt_pool, stream.finalize))
        if not any(ch.isalnum() for ch in text):
            return
        logger.info("heard %s: %s", listener.name, text)
        await self._handle_utterance(Utterance(speaker_id=listener.user_id, speaker_name=listener.name, text=text))

    # ---- turn management --------------------------------------------------------------
    def _is_busy(self) -> bool:
        task = self._turn_task
        return (task is not None and not task.done()) or self._source.is_active

    async def _handle_utterance(self, utterance: Utterance) -> None:
        async with self._turn_lock:
            await self._interrupt_locked()
            label = self._settings.turn.label_speakers
            self._append_user(f"{utterance.speaker_name}: {utterance.text}" if label else utterance.text)
            self._start_turn(fixed_text=None)

    def _append_user(self, content: str) -> None:
        last = self._history[-1]
        if last.role is Role.USER:  # previous reply never got spoken: merge to keep roles alternating
            self._history[-1] = ChatMessage(Role.USER, f"{last.content}\n{content}")
        else:
            self._history.append(ChatMessage(Role.USER, content))
        keep = self._settings.turn.max_history_messages
        if len(self._history) > keep + 1:
            tail = self._history[-keep:]
            while tail and tail[0].role is not Role.USER:
                tail = tail[1:]
            self._history = [self._history[0], *tail]

    def _start_turn(self, *, fixed_text: str | None) -> None:
        turn = ResponseTurn(
            messages=list(self._history) if fixed_text is None else [],
            generation=self._source.generation,
        )
        self._turn = turn
        self._turn_task = asyncio.create_task(self._respond(turn, fixed_text), name="voice-reply")

    async def _interrupt_locked(self) -> None:
        turn, task = self._turn, self._turn_task
        if turn is None:
            return
        turn.played_limit = self._source.played_bytes
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._source.clear()  # also invalidates audio a TTS thread may still be producing
        self._settle(turn)
        self._turn = None
        self._turn_task = None

    def _settle(self, turn: ResponseTurn) -> None:
        """Record in the history only what the user actually had the chance to hear."""
        limit = turn.played_limit
        spoken = [s.text for s in turn.sentences if limit is None or s.start_byte < limit]
        if spoken:
            self._history.append(ChatMessage(Role.ASSISTANT, " ".join(spoken)))

    # ---- reply pipeline: LLM -> sentences -> TTS -> playback --------------------------
    async def _respond(self, turn: ResponseTurn, fixed_text: str | None) -> None:
        sentences: asyncio.Queue[str | None] = asyncio.Queue(maxsize=4)
        producer = asyncio.create_task(self._produce(turn, sentences, fixed_text))
        speaker = asyncio.create_task(self._speak(turn, sentences))
        try:
            await asyncio.gather(producer, speaker)
        except Exception as exc:
            logger.exception("voice reply failed")
            await self._say_error(f"Voice reply failed: {exc}")
        finally:
            for task in (producer, speaker):
                task.cancel()
            await asyncio.gather(producer, speaker, return_exceptions=True)
            self._source.end_response(turn.generation)

    async def _produce(self, turn: ResponseTurn, out: asyncio.Queue[str | None], fixed_text: str | None) -> None:
        cfg = self._settings.turn
        splitter = SentenceSplitter(first_chunk_min_chars=cfg.first_chunk_min_chars, max_chars=cfg.max_sentence_chars)
        if fixed_text is not None:
            for sentence in (*splitter.feed(fixed_text), *splitter.flush()):
                await out.put(sentence)
        else:
            think = ThinkFilter()
            async for delta in self._engine.chat.stream_reply(turn.messages):
                for sentence in splitter.feed(think.feed(delta)):
                    await out.put(sentence)
            for sentence in (*splitter.feed(think.flush()), *splitter.flush()):
                await out.put(sentence)
        await out.put(None)

    async def _speak(self, turn: ResponseTurn, sentences: asyncio.Queue[str | None]) -> None:
        loop = asyncio.get_running_loop()
        synth = self._engine.synthesizer
        rate = synth.sample_rate
        source = self._source

        def on_chunk(chunk: FloatAudio) -> bool:  # runs on the TTS thread
            return source.feed(f32_mono_to_discord_pcm(chunk, rate), turn.generation)

        while (sentence := await sentences.get()) is not None:
            turn.sentences.append(SpokenSentence(text=sentence, start_byte=source.fed_bytes))
            await loop.run_in_executor(self._engine.tts_pool, synth.synthesize, sentence, on_chunk)

    async def _say_error(self, message: str) -> None:
        if self._notify is not None:
            try:
                await self._notify(message)
            except Exception:
                logger.exception("could not deliver error notification")
