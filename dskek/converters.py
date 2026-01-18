from enum import Enum
from dataclasses import dataclass
from pydub import AudioSegment
from io import BytesIO


@dataclass
class AudioInfo:
    sample_rate: int
    channels: int
    sample_width: int
    custom_chunk_size: int | None = None

    def __post_init__(self):
        if self.custom_chunk_size is not None:
            self.chunk_size = self.custom_chunk_size
        else:
            self.chunk_size = self.sample_rate // 50 * self.channels * self.sample_width


class AudioType(Enum):
    DISCORD = AudioInfo(
        sample_rate=48000, channels=2, sample_width=2
    )  # chunk_size = sample_rate // 50 * channels * sample_width
    GEMINI_SEND = AudioInfo(
        sample_rate=16000, channels=1, sample_width=2, custom_chunk_size=1024
    )
    GEMINI_RECEIVE = AudioInfo(
        sample_rate=24000, channels=1, sample_width=2, custom_chunk_size=1024
    )


@dataclass
class AudioData:
    data: AudioSegment
    atype: AudioType

    def convert(self, to_type: AudioType):
        to_type_info = to_type.value
        return AudioData(
            data=self.data.set_frame_rate(to_type_info.sample_rate)
            .set_sample_width(to_type_info.sample_width)
            .set_channels(to_type_info.channels),
            atype=to_type,
        )

    @classmethod
    def from_raw(cls, data: bytes, atype: AudioType):
        return cls(
            data=AudioSegment.from_raw(
                BytesIO(data),
                sample_width=atype.value.sample_width,
                frame_rate=atype.value.sample_rate,
                channels=atype.value.channels,
            ),
            atype=atype,
        )
