import asyncio
from collections import defaultdict
import janus


class Stream:
    def __init__(self):
        self.audio_in_queue = asyncio.Queue()
        self.audio_out_queue = asyncio.Queue()

    def cleanup(self):
        self.audio_in_queue.shutdown()
        self.audio_out_queue.shutdown()


class StreamController:
    def __init__(self):
        self.streams: dict[str, Stream] = defaultdict(Stream)

    def create_stream(self, key: str):
        self.streams[key] = Stream()

    def __getitem__(self, key: str) -> Stream:
        return self.streams[key]
