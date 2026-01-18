import asyncio
from collections import defaultdict
import queue
import culsans


class Stream:
    def __init__(self):
        self._in_queue = culsans.Queue()
        self._out_queue = culsans.Queue()
        self.audio_in_queue = self._in_queue.async_q
        self.audio_out_queue = self._out_queue.sync_q

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
