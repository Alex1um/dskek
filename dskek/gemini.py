"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss
```
"""

import os
import asyncio
import base64
import io
import traceback
import janus

from PIL import Image

import argparse

from google import genai
from google.genai import types
from dskek.models import QueueData, AudioData, AudioType
from dskek.converters import AudioData, AudioType
from pydub import AudioSegment
import logging

# CHANNELS = 1
# SEND_SAMPLE_RATE = 16000
# RECEIVE_SAMPLE_RATE = 24000
# CHUNK_SIZE = 1024


logger = logging.getLogger(__name__)


MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

tools = [
    types.Tool(google_search=types.GoogleSearch()),
]

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    # media_resolution="MEDIA_RESOLUTION_LOW",
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    # context_window_compression=types.ContextWindowCompressionConfig(
    #     trigger_tokens=25600,
    #     sliding_window=types.SlidingWindow(target_tokens=12800),
    # ),
    tools=tools,
    system_instruction=types.Content(
        parts=[
            types.Part.from_text(
                text="Ты гриша, искусственный интелект. Ты можешь отвечать на любые вопросы, если они обращены к тебе. Не встревай в разговор и не отвечай, если явно не требуется."
            )
        ],
        role="user",
    ),
)


class AudioLoop:
    def __init__(self, in_queue: asyncio.Queue, out_queue: asyncio.Queue):
        self.in_queue = in_queue
        self.out_queue = out_queue

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self, text: str):
        logging.info(f"Sending text: {text}")
        await self.in_queue.put(text or ".")

    async def send_image(self, img: Image):
        logging.info("Sending image")
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        frame = {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
        await self.in_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg: AudioData = await self.in_queue.get()
            logging.info(f"Received {len(msg.data.raw_data)} bytes of audio")
            msg_converted = msg.convert(AudioType.GEMINI_SEND).to_google_segment()
            logging.info(f"Converted to {len(msg.data.raw_data)} bytes of audio")
            await self.session.send(input=msg_converted)

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    logging.info(f"Received {len(data)} bytes of audio from gemini")
                    self.out_queue.put_nowait(
                        AudioData.from_raw(
                            data=data, atype=AudioType.GEMINI_RECEIVE
                        )
                    )
                    continue
                # if text := response.text:
                #     self.out_queue.put_nowait(text)

            # # If you interrupt the model, it sends a turn_complete.
            # # For interruptions to work, we need to stop playback.
            # # So empty out the audio queue because it may have loaded
            # # much more audio than has played yet.
            while not self.out_queue.empty():
                self.out_queue.get_nowait()

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                # send_text_task = tg.create_task(self.send_text())
                t1 = tg.create_task(self.send_realtime())
                t2 = tg.create_task(self.receive_audio())

                await asyncio.gather(t1, t2)
                # await send_text_task
                # raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            logger.info("User requested exit")
            pass
        except ExceptionGroup as EG:
            logger.error(f"An error occurred in the Gemini stream: {EG}")
            # traceback.print_exception(EG)
