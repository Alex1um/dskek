"""Audio conversion and the streaming playback source for Discord (48 kHz, 16-bit, stereo)."""

import threading
from typing import override

import discord
import numpy as np

from .models import FloatAudio

DISCORD_SAMPLE_RATE = 48_000
DISCORD_CHANNELS = 2
BYTES_PER_SAMPLE = 2
FRAME_MS = 20
FRAME_BYTES = DISCORD_SAMPLE_RATE * FRAME_MS // 1000 * DISCORD_CHANNELS * BYTES_PER_SAMPLE  # 3840
_SILENCE = bytes(FRAME_BYTES)


def pcm_stereo_s16_to_mono_f32(pcm: bytes) -> FloatAudio:
    """Decoded Discord PCM (interleaved L/R int16) -> mono float32."""
    usable = len(pcm) - len(pcm) % (DISCORD_CHANNELS * BYTES_PER_SAMPLE)
    stereo = np.frombuffer(pcm, dtype="<i2", count=usable // BYTES_PER_SAMPLE).reshape(-1, DISCORD_CHANNELS)
    return (stereo.mean(axis=1, dtype=np.float32) / 32768.0).astype(np.float32, copy=False)


def f32_mono_to_discord_pcm(samples: FloatAudio, sample_rate: int) -> bytes:
    """Mono float32 at any rate -> 48 kHz interleaved stereo int16.

    Uses linear interpolation. That is plenty for speech and keeps the module
    free of native resampler dependencies.
    """
    if sample_rate != DISCORD_SAMPLE_RATE and len(samples) > 1:
        out_len = round(len(samples) * DISCORD_SAMPLE_RATE / sample_rate)
        positions = np.linspace(0, len(samples) - 1, out_len, dtype=np.float64)
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    mono16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    return np.column_stack((mono16, mono16)).tobytes()


def _ms_to_bytes(ms: int) -> int:
    return DISCORD_SAMPLE_RATE * ms // 1000 * DISCORD_CHANNELS * BYTES_PER_SAMPLE


class StreamingPcmSource(discord.AudioSource):
    """A never-ending audio source fed from other threads.

    * Plays silence while idle, so one ``voice_client.play()`` call lasts the whole session.
    * Waits for ``prebuffer_ms`` of audio before starting, and again after an underrun.
    * ``clear()`` drops queued audio at once (barge-in) and bumps ``generation`` so
      late chunks from a cancelled reply are rejected instead of being played.
    """

    def __init__(self, *, prebuffer_ms: int = 200) -> None:
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._prebuffer_bytes = _ms_to_bytes(prebuffer_ms)
        self._started = False
        self._finishing = False
        self._generation = 0
        self._fed = 0
        self._played = 0

    # ---- producer side (any thread) ---------------------------------------------------
    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def fed_bytes(self) -> int:
        with self._lock:
            return self._fed

    @property
    def played_bytes(self) -> int:
        with self._lock:
            return self._played

    @property
    def is_active(self) -> bool:
        """True while queued or playing audio."""
        with self._lock:
            return self._started or bool(self._buffer)

    def feed(self, pcm: bytes, generation: int) -> bool:
        """Queue audio. Returns False (and drops it) if ``generation`` is stale."""
        with self._lock:
            if generation != self._generation:
                return False
            self._buffer.extend(pcm)
            self._fed += len(pcm)
            return True

    def end_response(self, generation: int) -> None:
        """No more audio is coming for this reply: play out whatever is left."""
        with self._lock:
            if generation == self._generation:
                self._finishing = True

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._fed = self._played
            self._started = False
            self._finishing = False
            self._generation += 1

    # ---- discord.py player thread -----------------------------------------------------
    @override
    def read(self) -> bytes:
        with self._lock:
            if not self._started:
                ready = len(self._buffer) >= self._prebuffer_bytes or (self._finishing and self._buffer)
                if not ready:
                    return _SILENCE
                self._started = True

            if len(self._buffer) >= FRAME_BYTES:
                frame = bytes(self._buffer[:FRAME_BYTES])
                del self._buffer[:FRAME_BYTES]
                self._played += FRAME_BYTES
                return frame

            if self._finishing:  # last partial frame of the reply
                tail = bytes(self._buffer)
                self._buffer.clear()
                self._played += len(tail)
                self._started = False
                self._finishing = False
                return tail.ljust(FRAME_BYTES, b"\x00")

            self._started = False  # underrun: re-buffer before resuming
            return _SILENCE

    @override
    def is_opus(self) -> bool:
        return False

    @override
    def cleanup(self) -> None:
        self.clear()
