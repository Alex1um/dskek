"""Streaming speech recognition on CPU via sherpa-onnx (online transducer)."""

from typing import override

import numpy as np
import sherpa_onnx  # type: ignore[import-untyped]  # no stubs published

from .config import AsrConfig
from .models import FloatAudio, RecognitionUpdate
from .protocols import RecognitionStream, SpeechRecognizer

_MODEL_RATE = 16_000


class SherpaStreamingRecognizer(SpeechRecognizer):
    """One shared recognizer; one lightweight stream per speaker.

    sherpa-onnx resamples inside ``accept_waveform``, so Discord's 48 kHz audio
    can be fed as-is.
    """

    def __init__(self, config: AsrConfig) -> None:
        self._config = config
        if config.model_type == "t-one-ctc":
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_t_one_ctc(
                tokens=str(config.tokens),
                model=str(config.encoder) or str(config.decoder) or str(config.joiner),
                num_threads=config.num_threads,
                sample_rate=_MODEL_RATE,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=config.trailing_silence_s,
                rule3_min_utterance_length=20.0,
                decoding_method="greedy_search",
                provider="cpu",
            )
        else:
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(config.tokens),
                encoder=str(config.encoder),
                decoder=str(config.decoder),
                joiner=str(config.joiner),
                num_threads=config.num_threads,
                sample_rate=_MODEL_RATE,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=config.trailing_silence_s,
                rule3_min_utterance_length=20.0,
                decoding_method="greedy_search",
                provider="cpu",
            )

    @override
    def open_stream(self) -> RecognitionStream:
        return _SherpaStream(self._recognizer, self._config.tail_padding_s)


class _SherpaStream(RecognitionStream):
    def __init__(self, recognizer: sherpa_onnx.OnlineRecognizer, tail_padding_s: float) -> None:
        self._recognizer = recognizer
        self._tail_padding_s = tail_padding_s
        self._stream = recognizer.create_stream()
        self._input_rate: int | None = None
        """sherpa-onnx requires one fixed input rate per stream, so remember it for the tail padding."""

    @override
    def feed(self, samples: FloatAudio, sample_rate: int) -> RecognitionUpdate:
        self._input_rate = sample_rate
        self._stream.accept_waveform(sample_rate, samples)
        self._decode_ready()
        return RecognitionUpdate(
            text=str(self._recognizer.get_result(self._stream)).strip(),
            is_endpoint=self._recognizer.is_endpoint(self._stream),
        )

    @override
    def finalize(self) -> str:
        if self._input_rate is None:  # nothing was ever fed
            return ""
        tail = np.zeros(int(self._tail_padding_s * self._input_rate), dtype=np.float32)
        self._stream.accept_waveform(self._input_rate, tail)
        self._decode_ready()
        text = str(self._recognizer.get_result(self._stream)).strip()
        self._stream = self._recognizer.create_stream()  # clean state for the next utterance
        self._input_rate = None
        return text

    def _decode_ready(self) -> None:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
