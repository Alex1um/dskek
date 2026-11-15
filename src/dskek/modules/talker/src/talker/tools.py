"""Tool-calling framework: typed parameters -> JSON schema, permissions, execution."""

import asyncio
import json
import logging
import re
import types
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Literal, Protocol, Union, get_args, get_origin, get_type_hints

from .config import ToolsConfig
from .models import ToolCall, ToolRisk

logger = logging.getLogger("dskek.voice_chat.tools")

type JsonSchema = dict[str, object]
"""JSON Schema is free-form by nature; this is the one place dicts are unavoidable."""


class ToolError(Exception):
    """Raised by a tool for a problem the *model* should hear about (bad argument, not allowed...)."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: JsonSchema
    risk: ToolRisk = ToolRisk.SAFE


@dataclass(frozen=True, slots=True)
class CommandInfo:
    name: str
    help: str
    allowed: bool
    """Whether the LLM is allowed to run it."""


class SessionControl(Protocol):
    """What tools may ask of the running voice session."""

    def end_after_reply(self) -> None: ...

    def leave_after_reply(self) -> None: ...


class DiscordGateway(Protocol):
    """The Discord operations tools can use. Implemented by :class:`gateway.BotGateway`."""

    async def send_message(self, text: str, channel: str | None) -> str: ...

    def list_commands(self) -> list[CommandInfo]: ...

    async def run_command(self, speaker_id: int, command: str, arguments: str) -> str: ...

    def voice_members(self) -> list[str]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolContext:
    speaker_id: int
    speaker_name: str
    control: SessionControl
    gateway: DiscordGateway | None


# ---- schema generation / argument parsing ---------------------------------------------
def described(description: str, **kwargs: Any) -> Any:
    """``field()`` with a description for the tool schema: ``text: str = described("What to say")``."""
    return field(metadata={"description": description}, **kwargs)


def _schema_for_type(tp: object) -> JsonSchema:
    origin = get_origin(tp)
    if tp is str:
        return {"type": "string"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if origin is Literal:
        values = list(get_args(tp))
        return {"type": "integer" if all(isinstance(v, int) for v in values) else "string", "enum": values}
    if origin in (types.UnionType, Union):
        inner = [a for a in get_args(tp) if a is not type(None)]
        if len(inner) == 1:
            return _schema_for_type(inner[0])
    if origin is list:
        return {"type": "array", "items": _schema_for_type(get_args(tp)[0])}
    raise TypeError(f"Unsupported tool parameter type: {tp!r}")


def schema_for(params: type) -> JsonSchema:
    hints = get_type_hints(params)
    properties: dict[str, object] = {}
    required: list[str] = []
    for f in fields(params):
        prop = _schema_for_type(hints[f.name])
        if description := f.metadata.get("description"):
            prop["description"] = description
        properties[f.name] = prop
        if f.default is MISSING and f.default_factory is MISSING:
            required.append(f.name)
    return {"type": "object", "properties": properties, "required": required}


def _coerce(value: object, tp: object, name: str) -> object:
    """Strictly typed, but forgiving about what small models commonly do ("5" for 5)."""
    origin = get_origin(tp)
    if origin in (types.UnionType, Union):
        if value is None and type(None) in get_args(tp):
            return None
        inner = [a for a in get_args(tp) if a is not type(None)]
        return _coerce(value, inner[0], name)
    if tp is str:
        if isinstance(value, str):
            return value
        if isinstance(value, int | float) and not isinstance(value, bool):
            return str(value)
    elif tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
    elif tp is int:
        if isinstance(value, bool):
            pass
        elif isinstance(value, int):
            return value
        elif isinstance(value, float) and value.is_integer():
            return int(value)
        elif isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value)
    elif tp is float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    elif origin is Literal:
        if value in get_args(tp):
            return value
        raise ToolError(f"argument '{name}' must be one of {list(get_args(tp))}")
    elif origin is list:
        if isinstance(value, list):
            return [_coerce(v, get_args(tp)[0], name) for v in value]
    raise ToolError(f"argument '{name}' has the wrong type")


def parse_arguments[P](params: type[P], raw: str) -> P:
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise ToolError(f"arguments are not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ToolError("arguments must be a JSON object")
    hints = get_type_hints(params)
    kwargs: dict[str, object] = {}
    for f in fields(params):  # type: ignore[arg-type]  # P is a dataclass by contract
        if f.name in data:
            kwargs[f.name] = _coerce(data[f.name], hints[f.name], f.name)
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ToolError(f"missing required argument '{f.name}'")
    return params(**kwargs)


# ---- tools ----------------------------------------------------------------------------
class Tool(ABC):
    @property
    @abstractmethod
    def spec(self) -> ToolSpec: ...

    @abstractmethod
    async def invoke(self, ctx: ToolContext, arguments: str) -> str:
        """``arguments`` is the raw JSON the model produced. Return text for the model."""


class TypedTool[P](Tool):
    """A tool whose arguments are a dataclass ``P`` (schema + validation come for free)."""

    def __init__(self, *, name: str, description: str, params: type[P], risk: ToolRisk = ToolRisk.SAFE) -> None:
        self._params = params
        self._spec = ToolSpec(name=name, description=description, parameters=schema_for(params), risk=risk)

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, ctx: ToolContext, arguments: str) -> str:
        return await self.run(ctx, parse_arguments(self._params, arguments))

    @abstractmethod
    async def run(self, ctx: ToolContext, args: P) -> str: ...


class ToolRegistry:
    def __init__(self, config: ToolsConfig) -> None:
        self._config = config
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            logger.warning("tool %r registered twice; keeping the latest", tool.spec.name)
        self._tools[tool.spec.name] = tool

    def unregister_prefix(self, prefix: str) -> None:
        for name in [n for n in self._tools if n.startswith(prefix)]:
            del self._tools[name]

    def is_allowed(self, spec: ToolSpec, user_id: int | None) -> bool:
        cfg = self._config
        if not cfg.enabled or spec.name in cfg.disabled_tools:
            return False
        if spec.risk is ToolRisk.PRIVILEGED:
            return user_id is not None and user_id in cfg.privileged_user_ids
        return True

    def specs_for(self, user_id: int | None) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values() if self.is_allowed(t.spec, user_id)]

    async def call(self, call: ToolCall, ctx: ToolContext) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return f"Error: unknown tool '{call.name}'."
        if not self.is_allowed(tool.spec, ctx.speaker_id):  # enforced again: never trust the model's choice
            logger.warning("blocked tool %s for user %s", call.name, ctx.speaker_id)
            return f"Error: {ctx.speaker_name} is not allowed to use '{call.name}'."
        try:
            async with asyncio.timeout(self._config.call_timeout_s):
                result = await tool.invoke(ctx, call.arguments)
        except ToolError as exc:
            result = f"Error: {exc}"
        except TimeoutError:
            result = f"Error: '{call.name}' timed out."
        except Exception:
            logger.exception("tool %s failed", call.name)
            result = f"Error: '{call.name}' failed."
        limit = self._config.max_result_chars
        return result if len(result) <= limit else result[:limit] + " ...[truncated]"


# ---- streaming JSON string extraction (so say(text=...) can start speaking early) -----
_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "/": "/", '"': '"', "\\": "\\"}


class JsonStringFieldStream:
    """Incrementally extracts one string field from JSON text that arrives in fragments."""

    def __init__(self, field_name: str) -> None:
        self._opener = re.compile(r'"' + re.escape(field_name) + r'"\s*:\s*"')
        self._head = ""
        self._in_value = False
        self._done = False
        self._carry = ""
        self._high: int | None = None

    @property
    def started(self) -> bool:
        return self._in_value

    def feed(self, fragment: str) -> str:
        if self._done:
            return ""
        if not self._in_value:
            self._head += fragment
            match = self._opener.search(self._head)
            if match is None:
                self._head = self._head[-64:]
                return ""
            self._in_value = True
            fragment, self._head = self._head[match.end() :], ""
        text, self._carry = self._carry + fragment, ""
        out: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == '"':
                self._done = True
                break
            if ch != "\\":
                out.append(self._take_char(ch))
                i += 1
                continue
            if i + 1 >= len(text):
                self._carry = text[i:]
                break
            esc = text[i + 1]
            if esc == "u":
                if i + 6 > len(text):
                    self._carry = text[i:]
                    break
                try:
                    code = int(text[i + 2 : i + 6], 16)
                except ValueError:
                    code = 0xFFFD
                out.append(self._take_code_unit(code))
                i += 6
            else:
                out.append(self._take_char(_SIMPLE_ESCAPES.get(esc, esc)))
                i += 2
        return "".join(out)

    def _take_char(self, ch: str) -> str:
        prefix = "\ufffd" if self._high is not None else ""
        self._high = None
        return prefix + ch

    def _take_code_unit(self, code: int) -> str:
        if 0xD800 <= code <= 0xDBFF:  # high surrogate: wait for its pair
            prefix = "\ufffd" if self._high is not None else ""
            self._high = code
            return prefix
        if 0xDC00 <= code <= 0xDFFF:
            if self._high is not None:
                combined = 0x10000 + ((self._high - 0xD800) << 10) + (code - 0xDC00)
                self._high = None
                return chr(combined)
            return "\ufffd"
        return self._take_char(chr(code))
