"""Discord commands: ``!ai_join``, ``!ai_leave``, ``!ai_reset``, ``!ai_wake``.

Import this module once from your bot's startup code so the commands register::

    import dskek.voice_chat.commands  # noqa: F401
"""

import asyncio
import logging
from dataclasses import dataclass

from discord.ext import voice_recv
from discord.ext.commands import Bot, Context

from dskek.discord_bot import bot

from .audio import StreamingPcmSource
from .config import ConfigError, VoiceChatSettings
from .engine import VoiceEngine
from .gateway import BotGateway
from .session import VoiceChatSession
from .sink import VoiceChatSink

logger = logging.getLogger("dskek.voice_chat")

_DAVE_READY_TIMEOUT_S = 10.0


@dataclass(slots=True)
class _ActiveChat:
    session: VoiceChatSession
    client: voice_recv.VoiceRecvClient
    source: StreamingPcmSource


_active: dict[int, _ActiveChat] = {}
_engine: VoiceEngine | None = None
_engine_lock = asyncio.Lock()


async def _get_engine() -> VoiceEngine:
    global _engine
    async with _engine_lock:
        if _engine is None:
            _engine = await VoiceEngine.load(VoiceChatSettings.from_env())
        return _engine


async def _wait_dave_ready(client: voice_recv.VoiceRecvClient) -> bool:
    """Discord voice is end-to-end encrypted (DAVE); don't talk before the MLS group is ready."""
    deadline = asyncio.get_running_loop().time() + _DAVE_READY_TIMEOUT_S
    while asyncio.get_running_loop().time() < deadline:
        if getattr(client, "dave_ready", True):  # attribute only exists on DAVE-capable forks
            return True
        await asyncio.sleep(0.1)
    return False


@bot.command(name="ai_join", help="Join your voice channel and talk with the local AI")
async def ai_join(ctx: Context[Bot]) -> None:
    guild = ctx.guild
    voice = getattr(ctx.author, "voice", None)
    if guild is None or voice is None or voice.channel is None:
        await ctx.send("Join a voice channel first.")
        return
    if guild.id in _active:
        await ctx.send("Already in a voice chat here. Use `ai_leave` first.")
        return
    if guild.voice_client is not None:
        await ctx.send("The bot is already connected to voice (music?). Use `leave` first.")
        return

    try:
        async with ctx.typing():
            engine = await _get_engine()
    except ConfigError as exc:
        await ctx.send(f"Voice AI is not configured: {exc}")
        return

    client = await voice.channel.connect(cls=voice_recv.VoiceRecvClient)
    if not await _wait_dave_ready(client):
        logger.warning("DAVE not ready after %.0fs; continuing anyway", _DAVE_READY_TIMEOUT_S)

    source = StreamingPcmSource(prebuffer_ms=engine.settings.turn.prebuffer_ms)

    async def notify(message: str) -> None:
        await ctx.send(message)

    async def on_leave() -> None:
        await _stop_chat(guild.id)

    gateway = BotGateway(
        bot=bot,
        guild=guild,
        text_channel=ctx.channel,
        base_message=ctx.message,
        voice_channel=voice.channel,
        config=engine.settings.tools,
    )
    session = VoiceChatSession(
        engine=engine, source=source, loop=asyncio.get_running_loop(), gateway=gateway, notify=notify, on_leave=on_leave
    )
    client.listen(VoiceChatSink(session))
    client.play(source)
    _active[guild.id] = _ActiveChat(session=session, client=client, source=source)

    session.open_conversation()
    wake = engine.settings.wake
    if wake.enabled:
        await session.say(f"Hi! I'm {wake.names[0].title()}. Say my name when you need me.")
    else:
        await session.say("Hi! I'm listening. Just start talking.")


async def _stop_chat(guild_id: int) -> bool:
    chat = _active.pop(guild_id, None)
    if chat is None:
        return False
    await chat.session.close()
    chat.client.stop()
    await chat.client.disconnect()
    return True


async def shutdown_voice_chat() -> None:
    """Call from your bot's shutdown hook: closes sessions, MCP connections and worker threads."""
    for guild_id in list(_active):
        await _stop_chat(guild_id)
    if _engine is not None:
        await _engine.aclose()


@bot.command(name="ai_leave", help="Stop the AI voice chat and leave the channel")
async def ai_leave(ctx: Context[Bot]) -> None:
    if ctx.guild is None or not await _stop_chat(ctx.guild.id):
        await ctx.send("No AI voice chat is running here.")
        return
    await ctx.send("Bye!")


@bot.command(name="ai_wake", help="Require the bot's name before it answers: ai_wake on|off")
async def ai_wake(ctx: Context[Bot], mode: str = "") -> None:
    if ctx.guild is None or (chat := _active.get(ctx.guild.id)) is None:
        await ctx.send("No AI voice chat is running here.")
        return
    if mode.lower() in ("on", "off"):
        chat.session.set_wake_enabled(mode.lower() == "on")
    state = "needs its name" if chat.session.wake_enabled else "answers everything"
    await ctx.send(f"The bot {state}.")


@bot.command(name="ai_reset", help="Forget the AI conversation history")
async def ai_reset(ctx: Context[Bot]) -> None:
    if ctx.guild is None or (chat := _active.get(ctx.guild.id)) is None:
        await ctx.send("No AI voice chat is running here.")
        return
    chat.session.reset_history()
    await ctx.send("Conversation history cleared.")
