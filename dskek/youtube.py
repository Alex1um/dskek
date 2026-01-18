from dskek.discord_bot import bot
from dskek.env import YT_PROXY, FFMPEG_PROXY
import logging
import yt_dlp
import discord
import asyncio
from discord.ext.commands import Context
import traceback


logger = logging.getLogger("discord")


yt_dlp.utils.bug_reports_message = lambda before: logger.info(before)


ytdl_format_options = {
    'format': 'bestaudio/best',
    'outtmpl': '%(extractor)s-%(id)s-%(title)s.%(ext)s',
    'restrictfilenames': True,
    'noplaylist': True,
    'nocheckcertificate': True,
    'ignoreerrors': False,
    'logtostderr': False,
    'quiet': True,
    'no_warnings': True,
    'default_search': 'auto',
    # bind to ipv4 since ipv6 addresses cause issues sometimes
    'source_address': '0.0.0.0',
}

if YT_PROXY:
    ytdl_format_options['proxy'] = YT_PROXY

ffmpeg_options = {
    'options': '-vn',
}

if FFMPEG_PROXY:
    logger.info(f"Using ffmpeg proxy: {FFMPEG_PROXY}")
    ffmpeg_options['before_options'] = f'-http_proxy "{FFMPEG_PROXY}"'


ytdl = yt_dlp.YoutubeDL(ytdl_format_options)


class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=0.5):
        super().__init__(source, volume)
        self.data = data
        self.title = data.get('title')
        self.url = data.get('url')

    @classmethod
    async def from_url(cls, url: str, *, loop=None, stream=True):
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: ytdl.extract_info(url, download=not stream))
        if 'entries' in data:
            data = data['entries'][0]
        filename = data['url'] if stream else ytdl.prepare_filename(data)

        return cls(discord.FFmpegPCMAudio(filename, **ffmpeg_options), data=data)


@bot.command(name='yt', help='Tells the bot to join the voice channel')
async def join(ctx: Context):
    if not ctx.message.author.voice:
        await ctx.send("{} is not connected to a voice channel".format(ctx.message.author.name))
        return
    else:
        channel = ctx.message.author.voice.channel
    await channel.connect()


@bot.command(name='leave', help='To make the bot leave the voice channel')
async def leave(ctx: Context):
    voice_client = ctx.message.guild.voice_client
    if voice_client is not None and voice_client.is_connected():
        await voice_client.disconnect()
    else:
        await ctx.send("The bot is not connected to a voice channel.")


@bot.command(name='play', help='To play song')
async def play(ctx: Context, url: str):
    try:
        server = ctx.message.guild
        if server.voice_client is None:
            await join(ctx)
        voice_channel = server.voice_client

        async with ctx.typing():
            player = await YTDLSource.from_url(url, loop=bot.loop, stream=True)
            voice_channel.play(player, after=lambda e: print(
                'Player error: %s' % e) if e else None)

        await ctx.send('**Now playing:** {}'.format(player.title))
    except Exception as e:
        logger.error(f"An error occurred: {e}\n{traceback.format_exc()}")
        await ctx.send(f'An error occurred: {e}')
