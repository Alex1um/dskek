from dskek.discord_bot import bot
from dskek.channels import Stream
from dskek.converters import AudioType, AudioData
from dskek.gemini import AudioLoop
from discord.ext import voice_recv, commands
import discord
import asyncio
import logging


logger = logging.getLogger(__name__)


class VoiceBot(discord.AudioSource, voice_recv.AudioSink):
    def __init__(self):
        discord.AudioSource.__init__(self)
        voice_recv.AudioSink.__init__(self)
        self.stream = Stream()
        self.audio = AudioLoop(self.stream.audio_in_queue, self.stream.audio_out_queue)
        self.audio_buffer: AudioData = AudioData.from_raw(
            data=b"", atype=AudioType.DISCORD
        )

    async def run(self):
        await self.audio.run()

    def wants_opus(self):
        return False

    def is_opus(self):
        return False

    # def read(self):
    # if not self.audio_buffer:
    #     nxt: AudioData = self.stream.audio_out_queue.get_nowait()
    #     self.audio_buffer = nxt.convert(AudioType.DISCORD)
    # discord_chunk_size = AudioType.DISCORD.value.chunk_size
    # if len(self.audio_buffer.data.raw_data) < discord_chunk_size:
    #     if self.stream.audio_out_queue.empty():
    #         silence = AudioData.from_raw(data=b"\x00" * (
    #             discord_chunk_size - len(self.audio_buffer.data.raw_data)
    #         ), atype=AudioType.DISCORD)
    #         self.audio_buffer.data.append(silence.data, crossfade=0)
    #     else:
    # nxt: AudioData = self.stream.audio_out_queue.get_nowait()
    # self.audio_buffer.data.append(nxt.convert(AudioType.DISCORD).data, crossfade=0)
    # chunk = self.audio_buffer[:discord_chunk_size]
    # self.audio_buffer = self.audio_buffer[discord_chunk_size:]
    # return chunk

    def read(self):
        if not self.stream.audio_out_queue.empty():
            chunk: AudioData = self.stream.audio_out_queue.get_nowait()
            return chunk.convert(AudioType.DISCORD).data.raw_data

    def write(self, user: discord.Member, data: voice_recv.VoiceData):
        if user:
            self.stream.audio_in_queue.put_nowait(
                AudioData.from_raw(data=data.pcm, atype=AudioType.DISCORD)
            )

    def cleanup(self):
        self.stream.cleanup()
        return super().cleanup()


@bot.command("join")
async def on_join(ctx: commands.Context):
    if ctx.author == bot.user:
        return
    if not ctx.author.voice:
        logging.info(f"User {ctx.author} is not connected to a voice channel.")
        await ctx.reply("You are not connected to a voice channel.")
        return

    voice_channel = ctx.author.voice.channel
    guild_id = ctx.guild.id

    if guild_id in bot.voice_clients:
        logging.info(f"Bot is already in a voice channel for guild {guild_id}.")
        await ctx.reply("I'm already in a voice channel.")
        return

    try:
        vc = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)

        gemini = VoiceBot()

        vc.listen(gemini)
        vc.play(gemini)

        await gemini.run()
    except Exception as e:
        logging.exception(e)
        await ctx.reply(f"Exception: {e}")
