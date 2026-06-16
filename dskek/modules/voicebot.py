from dskek.discord_bot import bot
from dskek.channels import Stream
from dskek.converters import AudioType, AudioData
from dskek.gemini import AudioLoop
from discord.ext import voice_recv, commands
import discord
import asyncio
import logging
import traceback
import time


logger = logging.getLogger("discord")


class VoiceBot(discord.AudioSource, voice_recv.AudioSink):
    def __init__(self):
        discord.AudioSource.__init__(self)
        voice_recv.AudioSink.__init__(self)
        self.stream = Stream()
        self.audio = AudioLoop(self.stream)
        self.write_time = time.time()
        self.write_bytes = 0
        self.audio_buffer = b""

    async def run(self):
        await self.audio.run()

    def wants_opus(self):
        return False

    def is_opus(self):
        return False

    # def read(self):
    #     if not self.audio_buffer:
    #         nxt: AudioData = self.stream.audio_out_queue.get()
    #         next_converted = nxt.convert(AudioType.DISCORD)
    #         self.audio_buffer = next_converted.data.raw_data
    #     discord_chunk_size = AudioType.DISCORD.value.chunk_size
    #     if len(self.audio_buffer) < discord_chunk_size:
    #         if self.stream.audio_out_queue.empty():
    #             self.audio_buffer += b"\x00" * (discord_chunk_size - len(self.audio_buffer))
    #         else:
    #             nxt: AudioData = self.stream.audio_out_queue.get()
    #             next_converted = nxt.convert(AudioType.DISCORD)
    #             self.audio_buffer += next_converted.data.raw_data
    #     chunk = self.audio_buffer[:discord_chunk_size]
    #     self.audio_buffer = self.audio_buffer[discord_chunk_size:]
    #     return chunk

    def read(self):
        if not self.stream.audio_out_queue.empty():
            chunk: AudioData = self.stream.audio_out_queue.get()
            logger.info(f"Bot is reading {len(chunk.data.raw_data)} bytes of audio")
            return chunk.convert(AudioType.DISCORD).data.raw_data

    def write(self, user: discord.Member, data: voice_recv.VoiceData):
        if user:
            self.stream.audio_in_queue.put_nowait(
                AudioData.from_raw(data=data.pcm, atype=AudioType.DISCORD)
            )
            if time.time() - self.write_time > 10:
                logger.info(
                    f"Bot has written {self.write_bytes=} of audio over 10s"
                )
                self.write_time = time.time()
                self.write_bytes = 0

    def cleanup(self):
        self.stream.cleanup()
        return super().cleanup()


@bot.command("join")
async def on_join(ctx: commands.Context):
    if ctx.author == bot.user:
        return
    if not ctx.author.voice:
        logger.info(f"User {ctx.author} is not connected to a voice channel.")
        await ctx.reply("You are not connected to a voice channel.")
        return

    voice_channel = ctx.author.voice.channel
    guild_id = ctx.guild.id

    if guild_id in bot.voice_clients:
        logger.info(f"Bot is already in a voice channel for guild {guild_id}.")
        await ctx.reply("I'm already in a voice channel.")
        return

    try:
        logger.info(f"Attempting to join voice channel for guild {guild_id}.")
        vc = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)

        logger.info(f"Joined voice channel for guild {guild_id}.")
        gemini = VoiceBot()

        logger.info(f"Starting listen and play for guild {guild_id}.")
        vc.listen(gemini)
        vc.play(gemini)

        await ctx.reply("Joined voice channel. Starting Gemini stream...")
        logger.info(f"Starting gemini stream for guild {guild_id}.")
        await gemini.run()
    except Exception as e:
        logger.exception(f"Bot error: {e}\n{traceback.format_exc()}")
        await ctx.reply(f"Exception: {e}")
