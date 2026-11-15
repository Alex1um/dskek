"""One live voice conversation: listening, wake/end phrases, turn-taking, tools, barge-in."""

import asyncio
import logging
import time
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import override

import numpy as np

from .audio import DISCORD_SAMPLE_RATE, StreamingPcmSource, f32_mono_to_discord_pcm
from .config import ReplyMode
from .engine import VoiceEngine
from .llm import SentenceSplitter, ThinkFilter
from .models import (
    ChatMessage,
    FloatAudio,
    ReplyComplete,
    ResponseTurn,
    Role,
    Round,
    SpokenSentence,
    TextDelta,
    ToolArgsDelta,
    ToolCall,
    Utterance,
)
from .protocols import RecognitionStream
from .tools import JsonStringFieldStream, SessionControl, ToolContext, ToolError, ToolSpec, DiscordGateway, parse_arguments
from .tools_builtin import SAY_TOOL, SayArgs
from .wake import WakeAction, WakeWordMatcher

logger = logging.getLogger("dskek.voice_chat")

type Notifier = Callable[[str], Awaitable[None]]
type LeaveHandler = Callable[[], Awaitable[None]]
type QueuedSentence = tuple[Round, str]

_MAX_QUEUED_FRAMES = 500  # ~10 s of 20 ms frames per speaker before we start dropping
_MAX_BUFFERED_UTTERANCE_S = 60


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


class VoiceChatSession(SessionControl):
    """Everything that happens between one bot and one voice channel.

    Threading model: Discord's receive thread calls :meth:`push_audio`; everything else runs on
    the event loop. ASR runs on ``engine.stt_pool`` / ``final_pool``; TTS on ``engine.tts_pool``
    and writes straight into the thread-safe playback source.
    """

    def __init__(
        self,
        *,
        engine: VoiceEngine,
        source: StreamingPcmSource,
        loop: asyncio.AbstractEventLoop,
        gateway: DiscordGateway | None = None,
        notify: Notifier | None = None,
        on_leave: LeaveHandler | None = None,
    ) -> None:
        self._engine = engine
        self._settings = engine.settings
        self._tools = engine.tools
        self._source = source
        self._loop = loop
        self._gateway = gateway
        self._notify = notify
        self._on_leave = on_leave
        self._matcher = WakeWordMatcher(self._settings.wake)
        self._wake_enabled = self._settings.wake.enabled
        self._open = False
        self._touched_at = 0.0
        self._end_after_reply = False
        self._leave_after_reply = False
        self._leave_task: asyncio.Task[None] | None = None
        self._listeners: dict[int, _Listener] = {}
        bot_name = self._settings.wake.names[0].title()
        prompt = self._settings.llm.system_prompt.replace("{bot_name}", bot_name)
        self._history: list[ChatMessage] = [ChatMessage(Role.SYSTEM, prompt)]
        self._turn: ResponseTurn | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._turn_lock = asyncio.Lock()
        self._closed = False

    # ---- public API -------------------------------------------------------------------
    @property
    def gap_fill_limit_s(self) -> float:
        """Longest pause the sink should re-insert (0 = disabled)."""
        turn = self._settings.turn
        return turn.gap_finalize_s if turn.fill_gaps else 0.0

    def accepts(self, user_id: int) -> bool:
        allowed = self._settings.allowed_user_ids
        return not allowed or user_id in allowed

    def push_audio(self, user_id: int, name: str, samples: FloatAudio) -> None:
        """Thread-safe entry point for mono frames at 48 kHz."""
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

    def open_conversation(self) -> None:
        """Start in the 'awake' state (used right after joining)."""
        self._open = True
        self._touch()

    def set_wake_enabled(self, enabled: bool) -> None:
        self._wake_enabled = enabled
        self.open_conversation()

    @property
    def wake_enabled(self) -> bool:
        return self._wake_enabled

    async def say(self, text: str) -> None:
        """Speak a fixed phrase (no LLM, not added to the history)."""
        async with self._turn_lock:
            await self._interrupt_locked()
            self._start_turn(fixed_text=text, speaker=None)

    async def interrupt(self) -> None:
        async with self._turn_lock:
            await self._interrupt_locked()

    def reset_history(self) -> None:
        self._history = self._history[:1]

    async def close(self) -> None:
        self._closed = True
        if self._leave_task is not None:
            self._leave_task.cancel()
        for listener in list(self._listeners.values()):
            if listener.task is not None:
                listener.task.cancel()
        self._listeners.clear()
        await self.interrupt()
        self._source.clear()

    # ---- SessionControl (called by tools) ---------------------------------------------
    @override
    def end_after_reply(self) -> None:
        self._end_after_reply = True

    @override
    def leave_after_reply(self) -> None:
        self._leave_after_reply = True

    # ---- conversation state -----------------------------------------------------------
    def _touch(self) -> None:
        self._touched_at = time.monotonic()

    def _is_active(self) -> bool:
        """Is the bot currently 'in the conversation' (no name needed to talk to it)?"""
        if not self._wake_enabled:
            return True
        if not self._open:
            return False
        if self._is_busy():
            return True
        last = max(self._touched_at, self._source.last_audio_at)
        return time.monotonic() - last < self._settings.wake.follow_up_s

    def _is_busy(self) -> bool:
        task = self._turn_task
        return (task is not None and not task.done()) or self._source.is_active

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
        max_buffered = _MAX_BUFFERED_UTTERANCE_S * DISCORD_SAMPLE_RATE
        pending: list[FloatAudio] = []
        pending_samples = 0
        utterance: deque[FloatAudio] = deque()  # everything heard since the last finished utterance
        utterance_samples = 0

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

                endpoint = False
                if pending:
                    samples = np.concatenate(pending)
                    pending.clear()
                    pending_samples = 0
                    utterance.append(samples)
                    utterance_samples += len(samples)
                    while utterance_samples > max_buffered:
                        utterance_samples -= len(utterance.popleft())
                    update = await loop.run_in_executor(pool, stream.feed, samples, DISCORD_SAMPLE_RATE)

                    if len(update.text) >= turn_cfg.barge_in_min_chars and self._is_busy():
                        logger.info("barge-in by %s: %r", listener.name, update.text)
                        await self.interrupt()
                    endpoint = update.is_endpoint

                if endpoint or (frame is None and utterance):
                    audio = np.concatenate(utterance) if utterance else np.zeros(0, dtype=np.float32)
                    utterance.clear()
                    utterance_samples = 0
                    await self._finish_utterance(listener, stream, audio)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("listener for %s failed; continuing", listener.name)

    async def _finish_utterance(self, listener: _Listener, stream: RecognitionStream, audio: FloatAudio) -> None:
        loop = self._loop
        streaming_text = _normalize_transcript(await loop.run_in_executor(self._engine.stt_pool, stream.finalize))
        if len(audio) < self._settings.turn.min_utterance_s * DISCORD_SAMPLE_RATE:
            return

        text = streaming_text
        final = self._engine.final_recognizer
        # Second pass: more accurate, but costly, so on idle chatter run it only if the name might be in there.
        if final is not None and (self._is_active() or self._matcher.might_contain_wake(streaming_text)):
            final_text = _normalize_transcript(await loop.run_in_executor(self._engine.final_pool, final.transcribe, audio, DISCORD_SAMPLE_RATE))
            if final_text:
                text = final_text
            if final_text != streaming_text:
                logger.info("ASR %s: streaming=%r final=%r", listener.name, streaming_text, final_text)

        if self._settings.debug_dump_dir is not None:
            await loop.run_in_executor(None, self._dump, self._settings.debug_dump_dir, listener, audio, streaming_text, text)
        if not any(ch.isalnum() for ch in text):
            return
        await self._handle_utterance(Utterance(speaker_id=listener.user_id, speaker_name=listener.name, text=text))

    @staticmethod
    def _dump(directory: Path, listener: _Listener, audio: FloatAudio, streaming_text: str, final_text: str) -> None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stem = f"{datetime.now():%Y%m%d-%H%M%S-%f}-{listener.user_id}"
            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
            with wave.open(str(directory / f"{stem}.wav"), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(DISCORD_SAMPLE_RATE)
                out.writeframes(pcm.tobytes())
            (directory / f"{stem}.txt").write_text(f"streaming: {streaming_text}\nfinal: {final_text}\n", encoding="utf-8")
        except OSError:
            logger.exception("could not write debug dump")

    # ---- wake / end phrases and turn management ---------------------------------------
    async def _handle_utterance(self, utterance: Utterance) -> None:
        wake = self._settings.wake
        active = self._is_active()
        result = self._matcher.classify(utterance.text, active=active)
        text = utterance.text

        match result.action:
            case WakeAction.SILENCE:
                logger.info("silence phrase from %s", utterance.speaker_name)
                await self.interrupt()
                self._touch()
                return
            case WakeAction.END if self._wake_enabled:
                logger.info("end phrase from %s: %r", utterance.speaker_name, utterance.text)
                self._open = False
                if wake.reset_history_on_end:
                    self.reset_history()
                await self.say(wake.end_ack_phrase)
                return
            case WakeAction.WAKE:
                logger.info("name heard from %s", utterance.speaker_name)
                self._open = True
                self._touch()
                if not result.remainder:
                    await self.say(wake.ack_phrase)
                    return
                text = result.remainder
            case _:
                if not active:
                    logger.debug("ignored (not addressed to the bot): %r", utterance.text)
                    return

        self._touch()
        async with self._turn_lock:
            await self._interrupt_locked()
            label = self._settings.turn.label_speakers
            self._append_user(f"{utterance.speaker_name}: {text}" if label else text)
            self._start_turn(fixed_text=None, speaker=utterance)

    def _append_user(self, content: str) -> None:
        last = self._history[-1]
        if last.role is Role.USER:  # previous reply never got spoken: merge to keep roles alternating
            self._history[-1] = ChatMessage(Role.USER, f"{last.content}\n{content}")
        else:
            self._history.append(ChatMessage(Role.USER, content))
        keep = self._settings.turn.max_history_messages
        if len(self._history) > keep + 1:
            tail = self._history[-keep:]
            while tail and tail[0].role is not Role.USER:  # never start mid tool-exchange
                tail = tail[1:]
            self._history = [self._history[0], *tail]

    def _start_turn(self, *, fixed_text: str | None, speaker: Utterance | None) -> None:
        turn = ResponseTurn(
            messages=list(self._history) if fixed_text is None else [],
            generation=self._source.generation,
            speaker_id=speaker.speaker_id if speaker else None,
            speaker_name=speaker.speaker_name if speaker else "",
            fixed=fixed_text is not None,
        )
        self._turn = turn
        self._turn_task = asyncio.create_task(self._respond(turn, fixed_text), name="voice-reply")

    async def _interrupt_locked(self) -> None:
        turn, task = self._turn, self._turn_task
        self._end_after_reply = self._leave_after_reply = False
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
        """Record in the history what happened: tool exchanges, and only the speech actually played.

        ``say`` calls are stored as plain assistant text (trimmed to what was heard) rather than
        as tool calls, so an interrupted reply never leaves half-spoken arguments behind.
        """
        if turn.fixed:
            return
        limit = turn.played_limit
        for round_ in turn.rounds:
            spoken = " ".join(s.text for s in round_.sentences if limit is None or s.start_byte < limit)
            if round_.calls:
                self._history.append(ChatMessage(Role.ASSISTANT, spoken, tuple(round_.calls)))
                self._history.extend(round_.results)
            elif spoken:
                self._history.append(ChatMessage(Role.ASSISTANT, spoken))

    # ---- reply pipeline: LLM (+tools) -> sentences -> TTS -> playback -----------------
    async def _respond(self, turn: ResponseTurn, fixed_text: str | None) -> None:
        sentences: asyncio.Queue[QueuedSentence | None] = asyncio.Queue(maxsize=4)
        producer = asyncio.create_task(self._produce(turn, sentences, fixed_text))
        speaker = asyncio.create_task(self._speak(turn, sentences))
        completed = False
        try:
            try:
                await asyncio.gather(producer, speaker)
                completed = True
            except Exception as exc:
                logger.exception("voice reply failed")
                await self._say_error(f"Voice reply failed: {exc}")
            finally:
                for task in (producer, speaker):
                    task.cancel()
                await asyncio.gather(producer, speaker, return_exceptions=True)
                self._source.end_response(turn.generation)
        except asyncio.CancelledError:
            raise
        if completed:
            self._touch()
            self._apply_deferred_actions()

    def _apply_deferred_actions(self) -> None:
        if self._end_after_reply:
            self._end_after_reply = False
            self._open = False
        if self._leave_after_reply:
            self._leave_after_reply = False
            if self._on_leave is not None:
                self._leave_task = asyncio.create_task(self._leave_when_quiet(self._on_leave))

    async def _leave_when_quiet(self, on_leave: LeaveHandler) -> None:
        deadline = time.monotonic() + 30
        while self._source.is_active and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        await on_leave()

    def _new_splitter(self) -> SentenceSplitter:
        cfg = self._settings.turn
        return SentenceSplitter(first_chunk_min_chars=cfg.first_chunk_min_chars, max_chars=cfg.max_sentence_chars)

    async def _produce(self, turn: ResponseTurn, out: asyncio.Queue[QueuedSentence | None], fixed_text: str | None) -> None:
        if fixed_text is not None:
            round_ = Round()
            turn.rounds.append(round_)
            splitter = self._new_splitter()
            for sentence in (*splitter.feed(fixed_text), *splitter.flush()):
                await out.put((round_, sentence))
            await out.put(None)
            return

        tools_cfg = self._settings.tools
        specs = self._tools.specs_for(turn.speaker_id)
        tool_mode = tools_cfg.reply_mode is ReplyMode.TOOL and any(s.name == SAY_TOOL for s in specs)
        messages = list(turn.messages)

        for _ in range(max(1, tools_cfg.max_rounds)):
            round_ = Round()
            turn.rounds.append(round_)
            text, calls = await self._stream_round(round_, messages, specs, tool_mode, out)
            if not calls:
                break
            messages.append(ChatMessage(Role.ASSISTANT, text, calls))
            used_real_tool = False
            for call in calls:
                if call.name == SAY_TOOL:
                    result = "ok"
                else:
                    used_real_tool = True
                    result = await self._run_tool(call, turn)
                    round_.calls.append(call)
                    round_.results.append(ChatMessage(Role.TOOL, result, tool_call_id=call.id))
                messages.append(ChatMessage(Role.TOOL, result, tool_call_id=call.id))
            if not used_real_tool:
                break  # it only spoke: nothing left to follow up on
        await out.put(None)

    async def _stream_round(
        self,
        round_: Round,
        messages: list[ChatMessage],
        specs: list[ToolSpec],
        tool_mode: bool,
        out: asyncio.Queue[QueuedSentence | None],
    ) -> tuple[str, tuple[ToolCall, ...]]:
        splitter = self._new_splitter()
        think = ThinkFilter()
        say_fields: dict[int, JsonStringFieldStream] = {}
        say_splitters: dict[int, SentenceSplitter] = {}
        plain: list[str] = []
        calls: tuple[ToolCall, ...] = ()

        async for event in self._engine.chat.stream_reply(messages, specs):
            match event:
                case TextDelta(text=text):
                    clean = think.feed(text)
                    plain.append(clean)
                    if not tool_mode:
                        for sentence in splitter.feed(clean):
                            await out.put((round_, sentence))
                case ToolArgsDelta(index=index, name=name, fragment=fragment) if name == SAY_TOOL:
                    field_stream = say_fields.setdefault(index, JsonStringFieldStream("text"))
                    say_splitter = say_splitters.setdefault(index, self._new_splitter())
                    for sentence in say_splitter.feed(field_stream.feed(fragment)):
                        await out.put((round_, sentence))
                case ReplyComplete(tool_calls=tool_calls):
                    calls = tool_calls
                case _:
                    pass

        tail = think.flush()
        plain.append(tail)
        if not tool_mode:
            for sentence in (*splitter.feed(tail), *splitter.flush()):
                await out.put((round_, sentence))
        else:
            ignored = "".join(plain).strip()
            if ignored:
                logger.info("tool mode: plain text not spoken: %r", ignored)
        for say_splitter in say_splitters.values():
            for sentence in say_splitter.flush():
                await out.put((round_, sentence))

        # A say() whose text never streamed (e.g. the server sent the call in one piece): speak it now.
        for call in calls:
            if call.name == SAY_TOOL and not (call.index in say_fields and say_fields[call.index].started):
                try:
                    said = parse_arguments(SayArgs, call.arguments).text
                except ToolError:
                    continue
                fallback = self._new_splitter()
                for sentence in (*fallback.feed(said), *fallback.flush()):
                    await out.put((round_, sentence))
        return "".join(plain), calls

    async def _run_tool(self, call: ToolCall, turn: ResponseTurn) -> str:
        ctx = ToolContext(
            speaker_id=turn.speaker_id if turn.speaker_id is not None else 0,
            speaker_name=turn.speaker_name,
            control=self,
            gateway=self._gateway,
        )
        result = await self._tools.call(call, ctx)
        logger.info("tool %s(%s) -> %s", call.name, call.arguments, result[:160])
        return result

    async def _speak(self, turn: ResponseTurn, sentences: asyncio.Queue[QueuedSentence | None]) -> None:
        loop = asyncio.get_running_loop()
        synth = self._engine.synthesizer
        rate = synth.sample_rate
        source = self._source

        def on_chunk(chunk: FloatAudio) -> bool:  # runs on the TTS thread
            return source.feed(f32_mono_to_discord_pcm(chunk, rate), turn.generation)

        while (item := await sentences.get()) is not None:
            round_, sentence = item
            round_.sentences.append(SpokenSentence(text=sentence, start_byte=source.fed_bytes))
            await loop.run_in_executor(self._engine.tts_pool, synth.synthesize, sentence, on_chunk)

    async def _say_error(self, message: str) -> None:
        if self._notify is not None:
            try:
                await self._notify(message)
            except Exception:
                logger.exception("could not deliver error notification")
