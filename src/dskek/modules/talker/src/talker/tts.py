"""Local speech synthesis on CPU via sherpa-onnx (Kokoro or VITS/Piper)."""

from collections.abc import Callable
from typing import override

import numpy as np
import sherpa_onnx  # type: ignore[import-untyped]  # no stubs published

from .config import ConfigError, TtsConfig, TtsKind
from .models import FloatAudio
from .protocols import SpeechSynthesizer


def _s(path: object) -> str:
    return "" if path is None else str(path)


class SherpaSynthesizer(SpeechSynthesizer):
    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        model_config = self._build_model_config(config)
        self._tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model_config, max_num_sentences=1))

    @staticmethod
    def _build_model_config(config: TtsConfig) -> sherpa_onnx.OfflineTtsModelConfig:
        match config.kind:
            case TtsKind.KOKORO:
                if config.voices is None:
                    raise ConfigError("Kokoro needs VOICE_TTS_VOICES (voices.bin)")
                return sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=str(config.model),
                        voices=str(config.voices),
                        tokens=str(config.tokens),
                        lexicon=_s(config.lexicon),
                        data_dir=_s(config.data_dir),
                        dict_dir=_s(config.dict_dir),
                        lang=config.lang,
                    ),
                    num_threads=config.num_threads,
                    provider="cpu",
                )
            case TtsKind.VITS:
                return sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(config.model),
                        lexicon=_s(config.lexicon),
                        tokens=str(config.tokens),
                        data_dir=_s(config.data_dir),
                        dict_dir=_s(config.dict_dir),
                    ),
                    num_threads=config.num_threads,
                    provider="cpu",
                )

    @property
    @override
    def sample_rate(self) -> int:
        return int(self._tts.sample_rate)

    @override
    def synthesize(self, text: str, on_chunk: Callable[[FloatAudio], bool]) -> None:
        delivered = 0
        cancelled = False

        def callback(samples: FloatAudio, _progress: float) -> int:
            nonlocal delivered, cancelled
            chunk = np.array(samples, dtype=np.float32, copy=True)
            if not on_chunk(chunk):
                cancelled = True
                return 0
            delivered += len(chunk)
            return 1

        audio = self._tts.generate(text, sid=self._config.speaker_id, speed=self._config.speed, callback=callback)

        # Some models only report part of the audio through the callback: deliver the rest.
        if not cancelled and len(audio.samples) > delivered:
            rest = np.asarray(audio.samples[delivered:], dtype=np.float32)
            on_chunk(rest)
