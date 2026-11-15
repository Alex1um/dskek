"""Compare the streaming and second-pass recognizers on recorded utterances.

1. Set VOICE_DEBUG_DUMP_DIR=./dump and talk to the bot normally: each utterance is saved as
   <stem>.wav plus <stem>.txt (what each pass heard live).
2. Optionally write what you REALLY said into <stem>.ref (one line) for the ones you care about.
3. python scripts/asr_compare.py ./dump     (uses the VOICE_ASR_* and VOICE_FINAL_ASR_* settings)

Point VOICE_FINAL_ASR_* at another model (Whisper, Parakeet v3, SenseVoice...) and rerun to compare.
Prints the transcript of each pass per file, and word error rate where a .ref exists.
"""

import re
import sys
import wave
from pathlib import Path

import numpy as np

from dskek.voice_chat.config import VoiceChatSettings
from dskek.voice_chat.models import FloatAudio
from dskek.voice_chat.stt import SherpaFinalRecognizer, SherpaStreamingRecognizer


def read_wav(path: Path) -> tuple[FloatAudio, int]:
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "expected 16-bit mono WAV"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return (pcm.astype(np.float32) / 32768.0), w.getframerate()


def words(text: str) -> list[str]:
    return re.sub(r"[^\w' ]", " ", text.lower()).split()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    ref, hyp = words(reference), words(hypothesis)
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0], d[0, :] = range(len(ref) + 1), range(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return int(d[-1, -1]), len(ref)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    folder = Path(sys.argv[1])
    settings = VoiceChatSettings.from_env()
    streaming = SherpaStreamingRecognizer(settings.asr)
    final = SherpaFinalRecognizer(settings.final_asr) if settings.final_asr else None
    totals = {"streaming": [0, 0], "final": [0, 0]}

    for wav in sorted(folder.glob("*.wav")):
        audio, rate = read_wav(wav)
        stream = streaming.open_stream()
        step = rate // 10
        for i in range(0, len(audio), step):
            stream.feed(audio[i : i + step], rate)
        results = {"streaming": stream.finalize()}
        if final is not None:
            results["final"] = final.transcribe(audio, rate)
        print(f"\n{wav.name}  ({len(audio) / rate:.1f}s)")
        ref_path = wav.with_suffix(".ref")
        reference = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else None
        if reference:
            print(f"  said      : {reference}")
        for name, text in results.items():
            print(f"  {name:10s}: {text}")
            if reference:
                errors, count = word_errors(reference, text)
                totals[name][0] += errors
                totals[name][1] += count

    for name, (errors, count) in totals.items():
        if count:
            print(f"\nWER {name:9s}: {100 * errors / count:5.1f}%  ({errors} errors / {count} words)")


if __name__ == "__main__":
    main()
