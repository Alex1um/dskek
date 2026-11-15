"""Discord voice-receive sink that forwards each speaker's audio to a session."""

from typing import override

import discord
import numpy as np
from discord.ext import voice_recv

from .audio import GapFiller, pcm_stereo_s16_to_mono_f32
from .session import VoiceChatSession


class VoiceChatSink(voice_recv.AudioSink):
    """Runs on discord-ext-voice-recv's receive thread: keep ``write`` tiny and non-blocking."""

    def __init__(self, session: VoiceChatSession) -> None:
        super().__init__()
        self._session = session
        limit = session.gap_fill_limit_s
        self._gaps = GapFiller(max_fill_s=limit) if limit > 0 else None

    @override
    def wants_opus(self) -> bool:
        return False  # we want decoded 48 kHz stereo PCM

    @override
    def write(self, user: discord.User | discord.Member | None, data: voice_recv.VoiceData) -> None:
        if user is None or user.bot or not data.pcm or not self._session.accepts(user.id):
            return
        samples = pcm_stereo_s16_to_mono_f32(data.pcm)
        timestamp = getattr(data.packet, "timestamp", None)
        if self._gaps is not None and isinstance(timestamp, int):
            if missing := self._gaps.missing_samples(user.id, timestamp, len(samples)):
                self._session.push_audio(user.id, user.display_name, np.zeros(missing, dtype=np.float32))
        self._session.push_audio(user.id, user.display_name, samples)

    @voice_recv.AudioSink.listener()  # type: ignore[untyped-decorator]
    def on_voice_member_disconnect(self, member: discord.Member, ssrc: int | None) -> None:
        if self._gaps is not None:
            self._gaps.forget(member.id)
        self._session.drop_speaker(member.id)

    @override
    def cleanup(self) -> None:
        pass
