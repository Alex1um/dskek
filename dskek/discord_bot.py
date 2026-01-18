import discord
from discord.ext import commands
import os
from aiohttp_socks import ProxyConnector
from dskek.env import DISCORD_BOT_TOKEN, PROXY
from dskek.proxy_clients import get_aio_proxy_connector_checked
import logging


intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
logger = logging.getLogger(__name__)


class ProxiedBot(commands.Bot):
    async def start(self, *args, **kwargs):
        connector = get_aio_proxy_connector_checked(PROXY)
        if connector:
            self.http.connector = connector
            logger.info(f"Using proxy: {PROXY}")
        logger.info("Starting bot...")
        await super().start(*args, **kwargs)


bot = ProxiedBot(command_prefix='!', intents=intents)


def main():
    if not DISCORD_BOT_TOKEN:
        print("ERROR: Please set the DISCORD_BOT_TOKEN environment variable.")
    else:
        bot.run(DISCORD_BOT_TOKEN)
