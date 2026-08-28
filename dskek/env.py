from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
PROXY = os.environ.get("PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
YT_PROXY = os.environ.get("YT_PROXY")
FFMPEG_PROXY = os.environ.get("FFMPEG_PROXY")
GEMINI_PROXY = os.environ.get("GEMINI_PROXY")
COOKIE_FILE = os.environ.get("COOKIE_FILE")
USER_AGENT = os.environ.get("USER_AGENT")
SINK_NAME        = os.environ.get("SINK_NAME", "discord_sink")
SINK_DESCRIPTION = os.environ.get("SINK_DESCRIPTION", "Discord Bot Audio")
MONITOR_SOURCE   = f"{SINK_NAME}.monitor"

if PROXY:
    os.environ["wss_proxy"] = PROXY
    os.environ["ws_proxy"] = PROXY

