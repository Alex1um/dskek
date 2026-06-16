import asyncio
import json
import logging
import os
import signal
import subprocess
import sys

import discord
from discord.ext import commands

from dskek.discord_bot import bot
from dskek.env import MONITOR_SOURCE, SINK_DESCRIPTION, SINK_NAME

log = logging.getLogger("discord-audio-bot")

ffmpeg_log_file = open("ffmpeg.log", "w")

_pulse_module_id: str | None = None


def sink_create() -> None:
    """Create the virtual null sink via pactl and remember its module ID."""
    global _pulse_module_id

    if not _which("pactl"):
        log.error("pactl not found – install pipewire-pulse (or pulseaudio-utils).")
        sys.exit(1)

    # Tear down any stale sink from a previous crashed run first.
    _unload_existing_sink()

    log.info("Creating virtual sink '%s' (%s)…", SINK_NAME, SINK_DESCRIPTION)
    result = subprocess.run(
        [
            "pactl",
            "load-module",
            "module-null-sink",
            f"sink_name={SINK_NAME}",
            f"sink_properties=device.description={SINK_DESCRIPTION}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log.error("Failed to create sink: %s", result.stderr.strip())
        sys.exit(1)

    _pulse_module_id = result.stdout.strip()
    log.info(
        "Sink ready – route audio into '%s', captured from '%s' (module id %s).",
        SINK_NAME,
        MONITOR_SOURCE,
        _pulse_module_id,
    )


def sink_destroy() -> None:
    """Unload the sink module; safe to call multiple times."""
    global _pulse_module_id

    if _pulse_module_id is None:
        return

    log.info("Destroying virtual sink (module id %s)…", _pulse_module_id)
    result = subprocess.run(
        ["pactl", "unload-module", _pulse_module_id],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log.info("Sink '%s' destroyed.", SINK_NAME)
    else:
        log.warning(
            "Could not unload module %s: %s",
            _pulse_module_id,
            result.stderr.strip(),
        )

    _pulse_module_id = None


def _unload_existing_sink() -> None:
    """Remove an orphaned sink with the same name, if any (e.g. from a previous crash)."""
    result = subprocess.run(
        ["pactl", "list", "short", "modules"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if "module-null-sink" in line and f"sink_name={SINK_NAME}" in line:
            mod_id = line.split()[0]
            log.info("Removing stale sink module %s from previous run.", mod_id)
            subprocess.run(["pactl", "unload-module", mod_id], capture_output=True)


def _which(cmd: str) -> bool:
    import shutil

    return shutil.which(cmd) is not None


# ---------------------------------------------------------------------------
# Discord bot
# ---------------------------------------------------------------------------


def make_audio_source() -> discord.FFmpegPCMAudio:
    """Open a live audio stream from the PipeWire monitor source via ffmpeg."""
    log.info(f"Starting audio stream from {MONITOR_SOURCE}")
    return discord.FFmpegPCMAudio(
        source=MONITOR_SOURCE,
        # -f pulse tells ffmpeg to treat the source as a PulseAudio device
        # (PipeWire exposes a Pulse-compatible server via pipewire-pulse).
        before_options="-report -loglevel level+trace -f pulse",
        stderr=ffmpeg_log_file,
    )


def _auto_restart(vc: discord.VoiceClient, error: Exception | None) -> None:
    """Called by discord.py when the ffmpeg process exits; restarts it."""
    if error:
        log.warning("Audio stream ended with error: %s", error)
    if vc.is_connected():
        log.info("Restarting audio stream from %s", MONITOR_SOURCE)
        vc.play(make_audio_source(), after=lambda e: _auto_restart(vc, e))


# ---------------------------------------------------------------------------
# Bot events
# ---------------------------------------------------------------------------


@bot.event
async def on_ready() -> None:
    log.info("Logged in as %s (id=%s).", bot.user, bot.user.id if bot.user else "?")
    log.info(
        "Sink '%s' active. Route app audio into it, then use !join in Discord.",
        SINK_NAME,
    )


@bot.event
async def on_disconnect() -> None:
    # Voice clients are already cleaned up by discord.py before this fires.
    log.info("Bot disconnected from Discord.")


# ---------------------------------------------------------------------------
# Bot commands
# ---------------------------------------------------------------------------


@bot.command()
async def stream(ctx: commands.Context) -> None:
    """Join your current voice channel and start streaming the PipeWire monitor."""
    if ctx.author.voice is None or ctx.author.voice.channel is None:
        await ctx.send("❌ You need to be connected to a voice channel first.")
        return

    channel = ctx.author.voice.channel

    if ctx.voice_client is not None:
        await ctx.voice_client.move_to(channel)
        vc = ctx.voice_client
    else:
        vc = await channel.connect()

    if vc.is_playing():
        vc.stop()

    vc.play(make_audio_source(), after=lambda e: _auto_restart(vc, e))
    await ctx.send(
        f"🎵 Joined **{channel.name}** and streaming `{MONITOR_SOURCE}`.\n"
        f"Route any app's audio output to **{SINK_DESCRIPTION}** to hear it here.\n"
        f"*(pavucontrol → Playback tab → change device, or use `pw-link` / `qpwgraph`)*"
    )


@bot.command()
async def no_stream(ctx: commands.Context) -> None:
    """Stop streaming and disconnect from the voice channel."""
    vc = ctx.voice_client
    if vc is not None:
        vc.stop()
        await vc.disconnect()
        await ctx.send("👋 Disconnected.")
    else:
        await ctx.send("I'm not in a voice channel.")


@bot.command()
async def status(ctx: commands.Context) -> None:
    """Show current connection and sink status."""
    vc = ctx.voice_client
    sink_ok = _pulse_module_id is not None

    lines = [
        f"**Sink:** `{SINK_NAME}` ({'active ✅' if sink_ok else 'not running ❌'})",
        f"**Monitor:** `{MONITOR_SOURCE}`",
    ]
    if vc is None:
        lines.append("**Voice:** not connected")
    else:
        state = "streaming 🎵" if vc.is_playing() else "connected, idle"
        lines.append(f"**Voice:** {vc.channel.name} – {state}")

    await ctx.send("\n".join(lines))


@bot.command()
async def sources(ctx: commands.Context) -> None:
    """List available PulseAudio/PipeWire sources (for finding monitor names)."""
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.strip() or "No sources found."
    except FileNotFoundError:
        output = "pactl not found. Install pipewire-pulse or pulseaudio-utils."
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr.strip()}"

    await ctx.send(f"```\n{output[:1900]}\n```")


@bot.command()
async def sinks(ctx: commands.Context) -> None:
    """List available PulseAudio/PipeWire sinks."""
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.strip() or "No sinks found."
    except FileNotFoundError:
        output = "pactl not found. Install pipewire-pulse or pulseaudio-utils."
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr.strip()}"

    await ctx.send(f"```\n{output[:1900]}\n```")


# ---------------------------------------------------------------------------
# Shutdown helpers
# ---------------------------------------------------------------------------


async def _shutdown(sig_name: str) -> None:
    """Gracefully disconnect voice clients, close the bot, then destroy the sink."""
    log.info("Received %s – shutting down…", sig_name)

    # Stop all active voice streams and disconnect.
    for vc in list(bot.voice_clients):
        try:
            vc.stop()
            await vc.disconnect(force=True)
        except Exception as exc:
            log.debug("Error disconnecting voice client: %s", exc)

    await bot.close()

    sink_destroy()


def _register_signals(loop: asyncio.AbstractEventLoop) -> None:
    """Register SIGINT/SIGTERM handlers on the running event loop."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.ensure_future(_shutdown(s.name), loop=loop),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


sink_create()

# loop = asyncio.get_running_loop()
# _register_signals(loop)
