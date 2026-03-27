from dskek.discord_bot import bot
from dskek.env import YT_PROXY, FFMPEG_PROXY, COOKIE_FILE, USER_AGENT
import logging
import yt_dlp
import discord
import asyncio
from discord.ext.commands import Context
import traceback
from pathlib import Path
import json


logger = logging.getLogger("discord")


yt_dlp.utils.bug_reports_message = lambda before: logger.info(before)


def format_selector(ctx):
    """ Select the best audio.
    NOTE: This is just an example and does not handle all cases """

    # formats are already sorted worst to best
    formats = ctx.get('formats')[::-1]

    # vcodec='none' means there is no video
    best_audio = next(f for f in formats if (
        f['acodec'] != 'none' and f['vcodec'] == 'none'))

    # These are the minimum required fields for a merged format
    yield {
        'format_id': f'{best_audio["format_id"]}',
        'ext': best_audio['ext'],
        'requested_formats': [best_audio],
        # Must be + separated list of protocols
        'protocol': f'{best_audio["protocol"]}'
    }


ytdl_format_options = {
    "format": format_selector,
    "outtmpl": "%(extractor)s-%(id)s-%(title)s.%(ext)s",
    "restrictfilenames": True,
    "noplaylist": True,
    "nocheckcertificate": True,
    "ignoreerrors": False,
    "logtostderr": False,
    "quiet": False,
    "no_warnings": True,
    "default_search": "auto",
    # bind to ipv4 since ipv6 addresses cause issues sometimes
    "source_address": "0.0.0.0",
    "js_runtimes": {"node": {}},
}

if YT_PROXY:
    ytdl_format_options["proxy"] = YT_PROXY

if COOKIE_FILE and Path(COOKIE_FILE).exists():
    print(f"Using cookie file: {COOKIE_FILE}")
    ytdl_format_options["cookiefile"] = COOKIE_FILE
    ytdl_format_options["cookies"] = COOKIE_FILE

if USER_AGENT:
    ytdl_format_options["user_agent"] = USER_AGENT

ffmpeg_options = {
    "options": "-vn",
}

if FFMPEG_PROXY:
    logger.info(f"Using ffmpeg proxy: {FFMPEG_PROXY}")
    ffmpeg_options["before_options"] = f'-http_proxy "{FFMPEG_PROXY}"'


ytdl = yt_dlp.YoutubeDL(ytdl_format_options)

queues = {}


class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=0.5):
        super().__init__(source, volume)
        self.data = data
        self.title = data.get("title")
        self.url = data.get("url")

    @classmethod
    async def from_url(cls, url: str, *, loop=None, stream=True):
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, lambda: ytdl.extract_info(url, download=not stream)
        )
        print(json.dumps(ytdl.sanitize_info(data)))
        if "entries" in data:
            data = data["entries"][0]
        filename = data["url"] if stream else ytdl.prepare_filename(data)

        return cls(discord.FFmpegPCMAudio(filename, **ffmpeg_options), data=data)


def play_next(ctx: Context):
    if ctx.guild.id in queues and queues[ctx.guild.id]:
        queue = queues[ctx.guild.id]
        if not ctx.voice_client.is_playing():
            player = queue.pop(0)
            ctx.voice_client.play(player, after=lambda e: play_next(ctx))
            asyncio.run_coroutine_threadsafe(
                ctx.send(f"**Now playing:** {player.title}"), bot.loop
            )


@bot.command(name="yt", help="Tells the bot to join the voice channel")
async def join(ctx: Context):
    if not ctx.message.author.voice:
        await ctx.send(
            "{} is not connected to a voice channel".format(ctx.message.author.name)
        )
        return
    else:
        channel = ctx.message.author.voice.channel
    await channel.connect()


@bot.command(name="leave", help="To make the bot leave the voice channel")
async def leave(ctx: Context):
    voice_client = ctx.message.guild.voice_client
    if voice_client is not None and voice_client.is_connected():
        await voice_client.disconnect()
    else:
        await ctx.send("The bot is not connected to a voice channel.")


@bot.command(name="play", help="To play song")
async def play(ctx: Context, url: str):
    try:
        server = ctx.message.guild
        if server.voice_client is None:
            await join(ctx)

        async with ctx.typing():
            player = await YTDLSource.from_url(url, loop=bot.loop, stream=True)
            if server.id not in queues:
                queues[server.id] = []
            queue = queues[server.id]
            queue.append(player)
            await ctx.send(f"**Added to queue:** {player.title}")

        if not server.voice_client.is_playing():
            play_next(ctx)

    except Exception as e:
        logger.error(f"An error occurred: {e} {traceback.format_exc()}")
        await ctx.send(f"An error occurred: {e}")


@bot.command(name="skip", help="To skip the current song")
async def skip(ctx: Context):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.stop()
        await ctx.send("Skipped the song.")
    else:
        await ctx.send("Not playing anything to skip.")


@bot.command(name="queue", help="To show the music queue")
async def show_queue(ctx: Context):
    if ctx.guild.id in queues and queues[ctx.guild.id]:
        queue_list = "\n".join(
            [
                f"{i + 1}. {player.title}"
                for i, player in enumerate(queues[ctx.guild.id])
            ]
        )
        await ctx.send(f"**Current Queue:**\n{queue_list}")
    else:
        await ctx.send("The queue is empty.")


@bot.command(name="nowplaying", help="To show the currently playing song")
async def now_playing(ctx: Context):
    if ctx.voice_client and ctx.voice_client.is_playing():
        await ctx.send(f"**Now playing:** {ctx.voice_client.source.title}")
    else:
        await ctx.send("Not playing anything at the moment.")


@bot.command(name="pause", help="To pause the current song")
async def pause(ctx: Context):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.pause()
        await ctx.send("Paused the song.")
    else:
        await ctx.send("Not playing anything to pause.")


@bot.command(name="resume", help="To resume the current song")
async def resume(ctx: Context):
    if ctx.voice_client and ctx.voice_client.is_paused():
        ctx.voice_client.resume()
        await ctx.send("Resumed the song.")
    else:
        await ctx.send("The song is not paused.")


@bot.command(name="stop", help="To stop the music and clear the queue")
async def stop(ctx: Context):
    if ctx.guild.id in queues:
        queues[ctx.guild.id] = []
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.stop()
        await ctx.send("Stopped the music and cleared the queue.")
    else:
        await ctx.send("Not playing anything to stop.")
