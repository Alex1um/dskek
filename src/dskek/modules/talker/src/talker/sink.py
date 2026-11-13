"""Discord voice-receive sink that forwards each speaker's audio to a session."""

from typing import override

import discord
from discord.ext import voice_recv

from .audio import pcm_stereo_s16_to_mono_f32
from .session import VoiceChatSession


class VoiceChatSink(voice_recv.AudioSink):
    """Runs on discord-ext-voice-recv's receive thread: keep ``write`` tiny and non-blocking."""

    def __init__(self, session: VoiceChatSession) -> None:
        super().__init__()
        self._session = session

    @override
    def wants_opus(self) -> bool:
        return False  # we want decoded 48 kHz stereo PCM

    @override
    def write(self, user: discord.User | discord.Member | None, data: voice_recv.VoiceData) -> None:
        if user is None or user.bot or not data.pcm or not self._session.accepts(user.id):
            return
        self._session.push_audio(user.id, user.display_name, pcm_stereo_s16_to_mono_f32(data.pcm))

    @voice_recv.AudioSink.listener()  # type: ignore[untyped-decorator]
    def on_voice_member_disconnect(self, member: discord.Member, ssrc: int | None) -> None:
        self._session.drop_speaker(member.id)

    @override
    def cleanup(self) -> None:
        pass
