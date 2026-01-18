import asyncio
from collections import defaultdict
import queue
import janus


class Stream:
    def __init__(self):
        self.audio_in_queue = queue.Queue()
        self.audio_out_queue = queue.Queue()

    def cleanup(self):
        # self.audio_in_queue.shutdown()
        # self.audio_out_queue.shutdown()
        pass


class StreamController:
    def __init__(self):
        self.streams: dict[str, Stream] = defaultdict(Stream)

    def create_stream(self, key: str):
        self.streams[key] = Stream()

    def __getitem__(self, key: str) -> Stream:
        return self.streams[key]
