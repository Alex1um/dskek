"""Built-in tools: speak, end the conversation, time, and (for trusted speakers) Discord actions."""

from dataclasses import dataclass
from datetime import datetime
from typing import override
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import ToolRisk
from .tools import DiscordGateway, ToolContext, ToolError, ToolRegistry, TypedTool, described

SAY_TOOL = "say"


@dataclass(frozen=True, slots=True)
class NoArgs:
    pass


@dataclass(frozen=True, slots=True)
class SayArgs:
    text: str = described("Exactly what to say out loud, in one to three short sentences. No markdown or emoji.")


@dataclass(frozen=True, slots=True)
class TimeArgs:
    timezone: str | None = described("IANA timezone such as 'Europe/Paris'. Omit for the bot's local time.", default=None)


@dataclass(frozen=True, slots=True)
class SendMessageArgs:
    text: str = described("The message text to post.")
    channel: str | None = described("Text channel name. Omit to use the channel where the voice chat was started.", default=None)


@dataclass(frozen=True, slots=True)
class RunCommandArgs:
    command: str = described("Name of the bot command, without the prefix (see list_bot_commands).")
    arguments: str = described("Arguments to pass to the command, if any.", default="")


def _gateway(ctx: ToolContext) -> DiscordGateway:
    if ctx.gateway is None:
        raise ToolError("Discord is not available in this session")
    return ctx.gateway


class SayTool(TypedTool[SayArgs]):
    """Speaking as a tool call. The session streams the text to TTS while the arguments arrive."""

    def __init__(self) -> None:
        super().__init__(
            name=SAY_TOOL,
            description="Speak text aloud in the voice channel. Use it for everything the user should hear.",
            params=SayArgs,
        )

    @override
    async def run(self, ctx: ToolContext, args: SayArgs) -> str:
        return "ok"  # the session has already spoken it


class EndConversationTool(TypedTool[NoArgs]):
    def __init__(self) -> None:
        super().__init__(
            name="end_conversation",
            description="End the conversation after your reply, when the user is clearly done. "
            "The bot then waits to hear its name again.",
            params=NoArgs,
        )

    @override
    async def run(self, ctx: ToolContext, args: NoArgs) -> str:
        ctx.control.end_after_reply()
        return "The conversation will end after your reply."


class GetTimeTool(TypedTool[TimeArgs]):
    def __init__(self) -> None:
        super().__init__(name="get_time", description="Get the current date and time.", params=TimeArgs)

    @override
    async def run(self, ctx: ToolContext, args: TimeArgs) -> str:
        try:
            now = datetime.now(ZoneInfo(args.timezone)) if args.timezone else datetime.now().astimezone()
        except ZoneInfoNotFoundError as exc:
            raise ToolError(f"unknown timezone '{args.timezone}'") from exc
        return now.strftime("%A, %d %B %Y, %H:%M (%Z)")


class ListVoiceMembersTool(TypedTool[NoArgs]):
    def __init__(self) -> None:
        super().__init__(name="list_voice_members", description="List who is in the voice channel.", params=NoArgs)

    @override
    async def run(self, ctx: ToolContext, args: NoArgs) -> str:
        return ", ".join(_gateway(ctx).voice_members()) or "Nobody else is here."


class ListBotCommandsTool(TypedTool[NoArgs]):
    def __init__(self) -> None:
        super().__init__(
            name="list_bot_commands",
            description="List the bot commands you may run with run_bot_command.",
            params=NoArgs,
        )

    @override
    async def run(self, ctx: ToolContext, args: NoArgs) -> str:
        runnable = [c for c in _gateway(ctx).list_commands() if c.allowed]
        if not runnable:
            return "There are no commands you may run."
        return "\n".join(f"{c.name}: {c.help or 'no description'}" for c in runnable)


class RunBotCommandTool(TypedTool[RunCommandArgs]):
    def __init__(self) -> None:
        super().__init__(
            name="run_bot_command",
            description="Run one of the bot's own commands as the person who asked.",
            params=RunCommandArgs,
            risk=ToolRisk.PRIVILEGED,
        )

    @override
    async def run(self, ctx: ToolContext, args: RunCommandArgs) -> str:
        return await _gateway(ctx).run_command(ctx.speaker_id, args.command.strip().lstrip("!/"), args.arguments.strip())


class SendMessageTool(TypedTool[SendMessageArgs]):
    def __init__(self) -> None:
        super().__init__(
            name="send_message",
            description="Post a text message in a Discord text channel.",
            params=SendMessageArgs,
            risk=ToolRisk.PRIVILEGED,
        )

    @override
    async def run(self, ctx: ToolContext, args: SendMessageArgs) -> str:
        if not args.text.strip():
            raise ToolError("the message is empty")
        return await _gateway(ctx).send_message(args.text.strip(), args.channel)


class LeaveVoiceTool(TypedTool[NoArgs]):
    def __init__(self) -> None:
        super().__init__(
            name="leave_voice",
            description="Leave the voice channel after your reply.",
            params=NoArgs,
            risk=ToolRisk.PRIVILEGED,
        )

    @override
    async def run(self, ctx: ToolContext, args: NoArgs) -> str:
        ctx.control.leave_after_reply()
        return "I will leave the voice channel after this reply."


def register_builtin_tools(registry: ToolRegistry) -> None:
    for tool in (
        SayTool(),
        EndConversationTool(),
        GetTimeTool(),
        ListVoiceMembersTool(),
        ListBotCommandsTool(),
        RunBotCommandTool(),
        SendMessageTool(),
        LeaveVoiceTool(),
    ):
        registry.register(tool)
