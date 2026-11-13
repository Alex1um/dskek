"""Typed settings for the voice-chat module."""

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Self

# DEFAULT_SYSTEM_PROMPT = (
#     "You are a friendly voice assistant taking part in a Discord voice call. "
#     "Your replies are spoken aloud, so answer in one to three short, natural sentences. "
#     "Never use markdown, lists, emoji, code blocks or URLs. "
#     "Several people may talk; each message starts with the speaker's name. "
#     "If you did not understand, ask briefly to repeat."
# )
DEFAULT_SYSTEM_PROMPT = (
    "Ты гриша, искусственный интелект."
    "Ты находишься в голосовом чате discord."
    "Любой твой ответ будет воспроизведен, так что отвечай коротко и натурально, но только ели обращение идет к тебе."
    "Ты можешь отвечать на любые вопросы, если они обращены к тебе."
    "Не встревай в разговор и не отвечай, если явно не требуется."
    "Отвечай только если идет явное обращение к тебе."
    "Никогда не используй markdown, списки, emoji, блоки кода или ссылки"
    "Каждое сообщение будет начато с имени говорящего пользователя"
)


class ConfigError(RuntimeError):
    """Raised when required settings are missing or invalid."""


class TtsKind(StrEnum):
    KOKORO = "kokoro"
    VITS = "vits"
    """VITS-family models exported for sherpa-onnx (including Piper voices)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AsrConfig:
    """Streaming transducer (e.g. streaming Zipformer) for sherpa-onnx."""

    encoder: Path
    decoder: Path
    joiner: Path
    tokens: Path
    model_type: str | None = None
    num_threads: int = 2
    trailing_silence_s: float = 0.8
    """Silence after speech before the recognizer declares an endpoint."""
    tail_padding_s: float = 0.5
    """Zeros appended when flushing so the last word is emitted."""


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
class TurnConfig:
    chunk_ms: int = 100
    """Audio is fed to the recognizer in chunks of this size."""
    gap_finalize_s: float = 0.6
    """Discord stops sending packets while you are silent; this much packet silence ends the utterance."""
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
    llm: LlmConfig = field(default_factory=LlmConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)
    allowed_user_ids: frozenset[int] = frozenset()
    """Empty means: listen to everybody in the channel."""

    @classmethod
    def from_env(cls) -> Self:
        """Build settings from ``VOICE_*`` environment variables (see README)."""
        model_type = os.environ.get("VOICE_ASR_MODEL_TYPE")
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
                model_type=model_type,
            )
        else:
            asr = AsrConfig(
                encoder=_path("VOICE_ASR_ENCODER"),
                decoder=_path("VOICE_ASR_DECODER"),
                joiner=_path("VOICE_ASR_JOINER"),
                tokens=_path("VOICE_ASR_TOKENS"),
                num_threads=int(os.environ.get("VOICE_ASR_THREADS", "2")),
                model_type=model_type,
            )
        tts = TtsConfig(
            kind=TtsKind(os.environ.get("VOICE_TTS_KIND", TtsKind.KOKORO)),
            model=_path("VOICE_TTS_MODEL"),
            tokens=_path("VOICE_TTS_TOKENS"),
            voices=_opt_path("VOICE_TTS_VOICES"),
            data_dir=_opt_path("VOICE_TTS_DATA_DIR"),
            lexicon=_opt_path("VOICE_TTS_LEXICON"),
            dict_dir=_opt_path("VOICE_TTS_DICT_DIR"),
            lang=os.environ.get("VOICE_TTS_LANG", ""),
            speaker_id=int(os.environ.get("VOICE_TTS_SPEAKER", "0")),
            speed=float(os.environ.get("VOICE_TTS_SPEED", "1.0")),
            num_threads=int(os.environ.get("VOICE_TTS_THREADS", "2")),
        )
        defaults = LlmConfig()
        llm = LlmConfig(
            base_url=os.environ.get("VOICE_LLM_BASE_URL", defaults.base_url),
            api_key=os.environ.get("VOICE_LLM_API_KEY", defaults.api_key),
            model=os.environ.get("VOICE_LLM_MODEL", defaults.model),
            system_prompt=os.environ.get("VOICE_LLM_SYSTEM_PROMPT", defaults.system_prompt),
            max_tokens=int(os.environ.get("VOICE_LLM_MAX_TOKENS", str(defaults.max_tokens))),
            disable_thinking=os.environ.get("VOICE_LLM_DISABLE_THINKING", "1") != "0",
        )
        allowed = frozenset(int(part) for part in os.environ.get("VOICE_ALLOWED_USERS", "").split(",") if part.strip())
        return cls(asr=asr, tts=tts, llm=llm, allowed_user_ids=allowed)


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


def _opt_path(name: str) -> Path | None:
    return _path(name) if os.environ.get(name) else None
