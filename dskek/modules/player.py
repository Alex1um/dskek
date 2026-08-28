from dskek.discord_bot import bot
from dskek.channels import Stream
from dskek.converters import AudioType, AudioData, AudioInfo
from discord.ext import commands
from pathlib import Path
from pydub import AudioSegment
import discord
import asyncio
import logging
import traceback
import time


logger = logging.getLogger("discord")


class AudioFileLoop:

    def __init__(self, stream: Stream, filename: str | Path):
        self.out_queue = stream.audio_out_queue
        self.segment: AudioSegment = AudioSegment.from_file(filename)
        self.info = AudioInfo(
            sample_width=self.segment.sample_width,
            sample_rate=self.segment.frame_rate,
            channels=self.segment.channels,
        )

    async def run(self):
        slice_start = 0
        slice_step = 25
        slice_end = slice_start + slice_step
        total = len(self.segment)
        while True:
            chunk = self.segment[slice_start % total:slice_end % total]
            self.out_queue.put_nowait(AudioData(chunk, self.info))
            slice_start = slice_end
            slice_end += slice_step
            if slice_end >= total:
                break
            await asyncio.sleep(0.025)
        logger.info("Audio file loop finished")


class PlayerBot(discord.AudioSource):
    def __init__(self):
        discord.AudioSource.__init__(self)
        self.stream = Stream()
        self.audio = AudioFileLoop(self.stream, "/home/alex1um/Desktop/Browser/Downloads/rapidsave.com_CMAF_AUDIO_128.mp4")
        self.write_time = time.time()
        self.write_bytes = 0
        self.audio_buffer = b""

    async def run(self):
        await self.audio.run()

    def is_opus(self):
        return False

    def read(self):
        if not self.stream.audio_out_queue.empty():
            chunk: AudioData = self.stream.audio_out_queue.get()
            logger.info(f"Bot is reading {len(chunk.data.raw_data)} bytes of audio")
            return chunk.convert(AudioType.DISCORD).data.raw_data

    def cleanup(self):
        self.stream.cleanup()
        return super().cleanup()


@bot.command("player")
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
        vc = await voice_channel.connect()

        logger.info(f"Joined voice channel for guild {guild_id}.")
        player = PlayerBot()

        logger.info(f"Playing for guild {guild_id}.")
        vc.play(player)

        await ctx.reply("Joined voice channel. Starting Gemini stream...")
        logger.info(f"Starting gemini stream for guild {guild_id}.")
        await player.run()
    except Exception as e:
        logger.exception(f"Bot error: {e}\n{traceback.format_exc()}")
        await ctx.reply(f"Exception: {e}")


@bot.command("player2")
async def on_join2(ctx: commands.Context):
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
        vc = await voice_channel.connect()

        logger.info(f"Playing for guild {guild_id}.")
        vc.play(discord.FFmpegPCMAudio("/home/alex1um/Desktop/Browser/Downloads/rapidsave.com_CMAF_AUDIO_128.mp4"))

        await ctx.reply("Joined voice channel. Starting playing file...")
        logger.info(f"Joined voice channel. Starting playing file...")
        while vc.is_playing():
            await asyncio.sleep(.1)
    except Exception as e:
        logger.exception(f"Bot error: {e}\n{traceback.format_exc()}")
        await ctx.reply(f"Exception: {e}")
