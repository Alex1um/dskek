from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
PROXY = os.environ.get("PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")

if PROXY:
    os.environ["wss_proxy"] = PROXY
    os.environ["ws_proxy"] = PROXY

