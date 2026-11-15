"""Discord side of the tools: send messages, run registered bot commands."""

import copy
import logging
from typing import override

import discord
from discord.ext.commands import Bot

from .config import ToolsConfig
from .tools import CommandInfo, DiscordGateway, ToolError

logger = logging.getLogger("dskek.voice_chat.gateway")

_MAX_MESSAGE = 2000
_OWN_PREFIX = "ai_"  # never let the model start/stop its own voice session


class BotGateway(DiscordGateway):
    def __init__(
        self,
        *,
        bot: Bot,
        guild: discord.Guild,
        text_channel: discord.abc.Messageable,
        base_message: discord.Message,
        voice_channel: discord.abc.Connectable,
        config: ToolsConfig,
    ) -> None:
        self._bot = bot
        self._guild = guild
        self._text_channel = text_channel
        self._base_message = base_message
        self._voice_channel = voice_channel
        self._config = config

    @override
    async def send_message(self, text: str, channel: str | None) -> str:
        target, label = self._resolve_channel(channel)
        # No pings from voice-controlled messages: neither @everyone nor user/role mentions.
        await target.send(text[:_MAX_MESSAGE], allowed_mentions=discord.AllowedMentions.none())
        return f"Message sent to {label}."

    def _resolve_channel(self, name: str | None) -> tuple[discord.abc.Messageable, str]:
        if not name:
            return self._text_channel, "the text channel"
        wanted = name.strip().lstrip("#").lower()
        for channel in self._guild.text_channels:
            if channel.name.lower() != wanted:
                continue
            if channel.id not in self._config.send_channel_ids:
                raise ToolError(f"I am not allowed to post in #{channel.name}")
            if not channel.permissions_for(self._guild.me).send_messages:
                raise ToolError(f"I cannot write in #{channel.name}")
            return channel, f"#{channel.name}"
        raise ToolError(f"no text channel named '{name}'")

    @override
    def list_commands(self) -> list[CommandInfo]:
        allow = self._config.command_allowlist
        return [
            CommandInfo(name=cmd.qualified_name, help=cmd.help or cmd.short_doc or "", allowed=cmd.qualified_name in allow)
            for cmd in self._bot.commands
            # if not cmd.hidden and not cmd.name.startswith(_OWN_PREFIX)
        ]

    @override
    async def run_command(self, speaker_id: int, command: str, arguments: str) -> str:
        cmd = self._bot.get_command(command)
        if cmd is None or cmd.name.startswith(_OWN_PREFIX):
            raise ToolError(f"unknown command '{command}'")
        if cmd.qualified_name not in self._config.command_allowlist:
            raise ToolError(f"I am not allowed to run '{cmd.qualified_name}'")
        member = self._guild.get_member(speaker_id)
        if member is None:
            raise ToolError("cannot find who asked")

        prefixes = await self._bot.get_prefix(self._base_message)
        prefix = prefixes if isinstance(prefixes, str) else next(iter(prefixes), "!")

        # Run it exactly as if the speaker had typed it, so the command's own checks apply to *them*.
        message = copy.copy(self._base_message)
        message.author = member
        message.content = f"{prefix}{cmd.qualified_name} {arguments}".strip()
        ctx = await self._bot.get_context(message)
        if not ctx.valid:
            raise ToolError(f"could not run '{cmd.qualified_name}'")
        logger.info("voice-triggered command %r by %s", message.content, member)
        await self._bot.invoke(ctx)
        return f"Ran '{cmd.qualified_name}'. Any output was posted in the text channel."

    @override
    def voice_members(self) -> list[str]:
        channel = self._voice_channel
        members = getattr(channel, "members", [])
        return [m.display_name for m in members if not m.bot]
