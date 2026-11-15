"""Typed settings for the voice-chat module."""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Self

from .models import ToolRisk

# DEFAULT_SYSTEM_PROMPT = (
#     "You are {bot_name}, a friendly voice assistant taking part in a Discord voice call. "
#     "Your replies are spoken aloud, so answer in one to three short, natural sentences. "
#     "Never use markdown, lists, emoji, code blocks or URLs. "
#     "The speech recognizer sometimes mishears words: if a request looks like a mishearing, "
#     "use the most sensible interpretation, or ask briefly to repeat. "
#     "Several people may talk; each message starts with the speaker's name. "
#     "Use tools only when the user clearly asks for an action, and say briefly what you did."
# )
DEFAULT_SYSTEM_PROMPT = (
    "Ты — {bot_name}, дружелюбный голосовой ассистент, участвующий в голосовом звонке Discord. "
    "Твои ответы произносятся вслух, поэтому отвечай одним-тремя короткими, естественными предложениями. "
    "Никогда не используй markdown, списки, эмодзи, блоки кода или URL-адреса. "
    "Распознаватель речи иногда ослышивается: если запрос выглядит как ослышка, "
    "используй наиболее разумную интерпретацию или коротко попроси повторить. "
    "Могут говорить несколько человек; каждое сообщение начинается с имени говорящего. "
    "Используй инструменты только тогда, когда пользователь явно просит выполнить действие, и коротко скажи, что ты сделал."
)


class ConfigError(RuntimeError):
    """Raised when required settings are missing or invalid."""


class TtsKind(StrEnum):
    KOKORO = "kokoro"
    VITS = "vits"
    """VITS-family models exported for sherpa-onnx (including Piper voices)."""


class FinalAsrKind(StrEnum):
    NEMO_TRANSDUCER = "nemo_transducer"
    """NVIDIA Parakeet TDT (encoder/decoder/joiner)."""
    WHISPER = "whisper"
    SENSE_VOICE = "sense_voice"


class ReplyMode(StrEnum):
    DIRECT = "direct"
    """Plain assistant text is spoken as it streams (lowest latency)."""
    TOOL = "tool"
    """Plain text is NOT spoken; the model must call the ``say`` tool (it may also stay silent)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AsrConfig:
    """Streaming transducer (e.g. streaming Zipformer) for sherpa-onnx: partial results + barge-in."""

    encoder: Path
    decoder: Path
    joiner: Path
    tokens: Path
    num_threads: int = 2
    trailing_silence_s: float = 0.8
    """Silence after speech before the recognizer declares an endpoint."""
    tail_padding_s: float = 0.5
    """Zeros appended when flushing so the last word is emitted."""
    decoding_method: str = "greedy_search"
    """``modified_beam_search`` is a little more accurate and a little slower."""
    model_type: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class FinalAsrConfig:
    """Accurate offline model that re-transcribes each finished utterance (second pass)."""

    kind: FinalAsrKind = FinalAsrKind.NEMO_TRANSDUCER
    encoder: Path | None = None
    decoder: Path | None = None
    joiner: Path | None = None
    model: Path | None = None
    """SenseVoice single-file model."""
    tokens: Path
    language: str = ""
    """Whisper / SenseVoice language code ('' = auto-detect)."""
    num_threads: int = 2
    max_utterance_s: float = 40.0


@dataclass(frozen=True, slots=True, kw_only=True)
class TtsConfig:
    kind: TtsKind = TtsKind.KOKORO
    model: Path
    tokens: Path
    voices: Path | None = None
    """Kokoro only."""
    data_dir: Path | None = None
    """espeak-ng-data directory (Kokoro English, Piper)."""
    lexicon: Path | None = None
    dict_dir: Path | None = None
    lang: str = ""
    speaker_id: int = 0
    speed: float = 1.0
    num_threads: int = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class LlmConfig:
    """Any OpenAI-compatible chat endpoint: llama.cpp server, Ollama, LM Studio, vLLM."""

    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "sk-local"
    model: str = "local"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_tokens: int = 160
    temperature: float = 0.7
    disable_thinking: bool = True
    """Ask the server to skip 'thinking' (Qwen3-style templates). Ignored by servers that don't support it."""
    timeout_s: float = 60.0


@dataclass(frozen=True, slots=True, kw_only=True)
class WakeConfig:
    """Voice markers: say the bot's name to start, an end phrase to stop."""

    enabled: bool = True
    names: tuple[str, ...] = ("assistant",)
    """Bot name first, then aliases/common mishearings (e.g. ``("nova", "noah", "nova's")``)."""
    end_phrases: tuple[str, ...] = (
        "goodbye",
        "bye bye",
        "stop listening",
        "that's all",
        "end conversation",
        "never mind",
    )
    silence_phrases: tuple[str, ...] = ("stop", "be quiet", "quiet", "shut up", "hold on", "wait")
    """Whole-utterance phrases that only silence the bot (they are not sent to the LLM)."""
    follow_up_s: float = 25.0
    """The conversation stays open this long after the last exchange; no name needed meanwhile."""
    fuzzy_threshold: float = 0.74
    """0..1 similarity for matching the name against a possibly misheard transcript (0.74 = one wrong letter in a 4-letter name)."""
    max_leading_words: int = 3
    """The name must appear within this many words from the start of the utterance."""
    ack_phrase: str = "Yes?"
    end_ack_phrase: str = "Okay, talk to you later."
    reset_history_on_end: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class McpServerConfig:
    """An external MCP server whose tools are offered to the LLM."""

    name: str
    command: str | None = None
    """stdio transport: executable to launch."""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    url: str | None = None
    """Streamable-HTTP transport: server URL (use instead of ``command``)."""
    allowed_tools: tuple[str, ...] = ()
    """Empty = expose every tool of the server."""
    risk: ToolRisk = ToolRisk.PRIVILEGED


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolsConfig:
    enabled: bool = True
    reply_mode: ReplyMode = ReplyMode.DIRECT
    privileged_user_ids: frozenset[int] = frozenset()
    """Speakers allowed to trigger PRIVILEGED tools. Empty = nobody (safe default)."""
    disabled_tools: frozenset[str] = frozenset()
    command_allowlist: frozenset[str] = frozenset()
    """Bot commands the LLM may run via ``run_bot_command``. Empty = none."""
    send_channel_ids: frozenset[int] = frozenset()
    """Extra text channels ``send_message`` may post to (the channel where ai_join ran is always allowed)."""
    call_timeout_s: float = 20.0
    max_rounds: int = 4
    max_result_chars: int = 1500
    mcp_servers: tuple[McpServerConfig, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnConfig:
    chunk_ms: int = 100
    """Audio is fed to the recognizer in chunks of this size."""
    gap_finalize_s: float = 0.6
    """Discord stops sending packets while you are silent; this much packet silence ends the utterance."""
    fill_gaps: bool = True
    """Re-insert the pauses Discord does not transmit (from RTP timestamps) so words are not glued together."""
    min_utterance_s: float = 0.25
    barge_in_min_chars: int = 4
    """Partial transcript length that counts as the user interrupting the bot."""
    prebuffer_ms: int = 200
    """Audio buffered before playback starts, to avoid stutter on slow CPUs."""
    max_history_messages: int = 24
    label_speakers: bool = True
    first_chunk_min_chars: int = 30
    """Allow splitting the first sentence at a comma once it is this long (lower latency)."""
    max_sentence_chars: int = 280


@dataclass(frozen=True, slots=True, kw_only=True)
class VoiceChatSettings:
    asr: AsrConfig
    tts: TtsConfig
    final_asr: FinalAsrConfig | None = None
    llm: LlmConfig = field(default_factory=LlmConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    allowed_user_ids: frozenset[int] = frozenset()
    """Empty means: listen to everybody in the channel."""
    debug_dump_dir: Path | None = None
    """If set, every utterance is saved as WAV + both transcripts, to build your own test set."""

    @classmethod
    def from_env(cls) -> Self:
        """Build settings from ``VOICE_*`` environment variables (see README)."""
        env = os.environ
        model_type = env.get("VOICE_ASR_MODEL_TYPE")
        if model_type == "t-one-ctc":
            model = (
                _path_no_exc("VOICE_ASR_ENCODER")
                or _path_no_exc("VOICE_ASR_DECODER")
                or _path_no_exc("VOICE_ASR_JOINER")
            )
            if model is None:
                raise ConfigError("For t-one-ctc any VOICE_ASR environment variable is required")
            asr = AsrConfig(
                encoder=model,
                decoder=model,
                joiner=model,
                tokens=_path("VOICE_ASR_TOKENS"),
                num_threads=int(os.environ.get("VOICE_ASR_THREADS", "2")),
                decoding_method=env.get("VOICE_ASR_DECODING", "greedy_search"),
                model_type=model_type,
            )
        else:
            asr = AsrConfig(
                encoder=_path("VOICE_ASR_ENCODER"),
                decoder=_path("VOICE_ASR_DECODER"),
                joiner=_path("VOICE_ASR_JOINER"),
                tokens=_path("VOICE_ASR_TOKENS"),
                num_threads=int(env.get("VOICE_ASR_THREADS", "2")),
                decoding_method=env.get("VOICE_ASR_DECODING", "greedy_search"),
                model_type=model_type
            )
        tts = TtsConfig(
            kind=TtsKind(env.get("VOICE_TTS_KIND", TtsKind.KOKORO)),
            model=_path("VOICE_TTS_MODEL"),
            tokens=_path("VOICE_TTS_TOKENS"),
            voices=_opt_path("VOICE_TTS_VOICES"),
            data_dir=_opt_path("VOICE_TTS_DATA_DIR"),
            lexicon=_opt_path("VOICE_TTS_LEXICON"),
            dict_dir=_opt_path("VOICE_TTS_DICT_DIR"),
            lang=env.get("VOICE_TTS_LANG", ""),
            speaker_id=int(env.get("VOICE_TTS_SPEAKER", "0")),
            speed=float(env.get("VOICE_TTS_SPEED", "1.0")),
            num_threads=int(env.get("VOICE_TTS_THREADS", "2")),
        )
        llm_defaults = LlmConfig()
        llm = LlmConfig(
            base_url=env.get("VOICE_LLM_BASE_URL", llm_defaults.base_url),
            api_key=env.get("VOICE_LLM_API_KEY", llm_defaults.api_key),
            model=env.get("VOICE_LLM_MODEL", llm_defaults.model),
            system_prompt=env.get("VOICE_LLM_SYSTEM_PROMPT", llm_defaults.system_prompt),
            max_tokens=int(env.get("VOICE_LLM_MAX_TOKENS", str(llm_defaults.max_tokens))),
            disable_thinking=env.get("VOICE_LLM_DISABLE_THINKING", "1") != "0",
        )
        wake_defaults = WakeConfig()
        wake = WakeConfig(
            enabled=env.get("VOICE_WAKE_ENABLED", "1") != "0",
            names=_csv("VOICE_BOT_NAMES") or wake_defaults.names,
            end_phrases=_csv("VOICE_END_PHRASES") or wake_defaults.end_phrases,
            follow_up_s=float(env.get("VOICE_FOLLOW_UP_S", str(wake_defaults.follow_up_s))),
            fuzzy_threshold=float(env.get("VOICE_WAKE_THRESHOLD", str(wake_defaults.fuzzy_threshold))),
        )
        tools = ToolsConfig(
            enabled=env.get("VOICE_TOOLS_ENABLED", "1") != "0",
            reply_mode=ReplyMode(env.get("VOICE_REPLY_MODE", ReplyMode.DIRECT)),
            privileged_user_ids=_ids("VOICE_PRIVILEGED_USERS"),
            command_allowlist=frozenset(_csv("VOICE_COMMAND_ALLOWLIST")),
            send_channel_ids=_ids("VOICE_SEND_CHANNELS"),
            disabled_tools=frozenset(_csv("VOICE_DISABLED_TOOLS")),
            mcp_servers=_load_mcp_servers(env.get("VOICE_MCP_CONFIG")),
        )
        return cls(
            asr=asr,
            tts=tts,
            final_asr=_final_asr_from_env(),
            llm=llm,
            wake=wake,
            tools=tools,
            allowed_user_ids=_ids("VOICE_ALLOWED_USERS"),
            debug_dump_dir=_opt_path("VOICE_DEBUG_DUMP_DIR", must_exist=False),
        )


def _final_asr_from_env() -> FinalAsrConfig | None:
    if not os.environ.get("VOICE_FINAL_ASR_TOKENS"):
        return None
    return FinalAsrConfig(
        kind=FinalAsrKind(os.environ.get("VOICE_FINAL_ASR_KIND", FinalAsrKind.NEMO_TRANSDUCER)),
        encoder=_opt_path("VOICE_FINAL_ASR_ENCODER"),
        decoder=_opt_path("VOICE_FINAL_ASR_DECODER"),
        joiner=_opt_path("VOICE_FINAL_ASR_JOINER"),
        model=_opt_path("VOICE_FINAL_ASR_MODEL"),
        tokens=_path("VOICE_FINAL_ASR_TOKENS"),
        language=os.environ.get("VOICE_FINAL_ASR_LANGUAGE", ""),
        num_threads=int(os.environ.get("VOICE_FINAL_ASR_THREADS", "2")),
    )


def _load_mcp_servers(path: str | None) -> tuple[McpServerConfig, ...]:
    """Parse the JSON file: ``{"servers": [{"name": ..., "command": ..., "args": [...]}]}``."""
    if not path:
        return ()
    try:
        raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        servers = raw["servers"]
        return tuple(
            McpServerConfig(
                name=str(entry["name"]),
                command=entry.get("command"),
                args=tuple(str(a) for a in entry.get("args", ())),
                env={str(k): str(v) for k, v in entry.get("env", {}).items()},
                url=entry.get("url"),
                allowed_tools=tuple(str(t) for t in entry.get("allowed_tools", ())),
                risk=ToolRisk(entry.get("risk", ToolRisk.PRIVILEGED)),
            )
            for entry in servers
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ConfigError(f"Invalid VOICE_MCP_CONFIG file {path}: {exc!r}") from exc


def _csv(name: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.environ.get(name, "").split(",") if part.strip())


def _ids(name: str) -> frozenset[int]:
    try:
        return frozenset(int(part) for part in _csv(name))
    except ValueError as exc:
        raise ConfigError(f"{name} must be a comma-separated list of numeric ids") from exc


def _path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"Environment variable {name} is required")
    path = Path(value).expanduser()
    if not path.exists():
        raise ConfigError(f"{name} points to a missing path: {path}")
    return path


def _path_no_exc(name: str, ok: bool = False) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.exists() and not ok:
        return None
    return path


def _opt_path(name: str, *, must_exist: bool = True) -> Path | None:
    if not os.environ.get(name):
        return None
    return _path(name) if must_exist else Path(os.environ[name]).expanduser()
