"""Real-time voice conversation with a local LLM in Discord voice channels (CPU-friendly)."""

from .config import VoiceChatSettings
from .engine import VoiceEngine
from .session import VoiceChatSession
from .commands import _ActiveChat

__all__ = ["VoiceChatSettings", "VoiceChatSession", "VoiceEngine"]
