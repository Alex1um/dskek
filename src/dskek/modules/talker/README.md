# dskek.voice_chat: talk to a local LLM in a Discord voice channel (CPU)

Real-time pipeline, all local except Discord itself:

```
Discord voice (DAVE-decrypted PCM, per speaker)
  -> streaming ASR  (sherpa-onnx online Zipformer, CPU)     partial text while you speak
  -> end-of-speech  (ASR endpoint OR packet gap)
  -> LLM            (any OpenAI-compatible server: llama.cpp / Ollama / ...), token stream
  -> sentence splitter (first chunk may end at a comma)
  -> TTS            (sherpa-onnx: Kokoro or Piper, CPU), pushed chunk by chunk
  -> StreamingPcmSource -> discord.py voice playback
```

Barge-in: if someone starts talking while the bot is talking, playback stops immediately, the
LLM stream is cancelled, and the chat history keeps **only the sentences that were actually played**.

## Why a pipeline and not VibeVoice end-to-end

* **VibeVoice-ASR-Streaming**: ASR only, and its install doc targets CUDA (NVIDIA container, flash-attn).
* **VibeASR.cpp**: ASR only (CPU, ggml, RTF < 1 on its benchmark CPUs). As published it is a CLI
  (`asr_infer --audio file.wav`) plus a Gradio demo. It has no streaming or Python API, and a
  per-utterance subprocess would reload ~1.6 GB of weights every time.
* Neither generates replies or speech, so something still has to do LLM + TTS.

VibeASR.cpp can be plugged in later by implementing `protocols.SpeechRecognizer` if it grows a
server or library mode (see "Swapping backends").

## Install (Python 3.14)

```bash
pip install -r requirements-voice.txt
```

Two things you must know:

1. **Discord enforces DAVE end-to-end encryption on all voice calls** (rollout completed March 2026).
   Upstream `discord-ext-voice-recv` has **not merged** DAVE decryption (several PRs are open), so
   receiving audio needs a patched build. `requirements-voice.txt` pins
   `zacker150/discord-ext-voice-recv` at the exact commit I installed and tested against. It is a
   small third-party fork of a fork that also pulls in a patched `discord.py`, so review it before
   running it with your bot token. It **replaces** your upstream `discord.py` install.
2. **`sherpa-onnx` is pinned `<2`.** 2.x dropped espeak-ng: Piper voices produce no audio and Kokoro
   needs a lexicon and silently skips unknown words ("Eiffel" disappeared in my test).

## Models

```bash
B=https://github.com/k2-fsa/sherpa-onnx/releases/download
curl -LO $B/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2   # streaming ASR
curl -LO $B/tts-models/kokoro-en-v0_19.tar.bz2                                  # TTS, nicer voice
curl -LO $B/tts-models/vits-piper-en_US-amy-low.tar.bz2                         # TTS, much faster
for f in *.tar.bz2; do tar xjf "$f"; done
```

Other languages: pick a streaming transducer from the sherpa-onnx model zoo and a matching voice.

Environment (paths relative to wherever you extracted):

```bash
A=models/sherpa-onnx-streaming-zipformer-en-2023-06-26
export VOICE_ASR_ENCODER=$A/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx
export VOICE_ASR_DECODER=$A/decoder-epoch-99-avg-1-chunk-16-left-128.onnx
export VOICE_ASR_JOINER=$A/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx
export VOICE_ASR_TOKENS=$A/tokens.txt

# Kokoro (nicer, ~1 core for real time)
K=models/kokoro-en-v0_19
export VOICE_TTS_KIND=kokoro VOICE_TTS_MODEL=$K/model.onnx VOICE_TTS_TOKENS=$K/tokens.txt
export VOICE_TTS_VOICES=$K/voices.bin VOICE_TTS_DATA_DIR=$K/espeak-ng-data VOICE_TTS_SPEAKER=3

# ...or Piper (about 15x faster than real time on one core)
# P=models/vits-piper-en_US-amy-low
# export VOICE_TTS_KIND=vits VOICE_TTS_MODEL=$P/en_US-amy-low.onnx VOICE_TTS_TOKENS=$P/tokens.txt
# export VOICE_TTS_DATA_DIR=$P/espeak-ng-data

export VOICE_LLM_BASE_URL=http://127.0.0.1:8080/v1   # default
export VOICE_LLM_MODEL=local
# optional: VOICE_ALLOWED_USERS=123,456  VOICE_ASR_THREADS=2  VOICE_TTS_THREADS=2
#           VOICE_LLM_MAX_TOKENS=160  VOICE_LLM_DISABLE_THINKING=0
```

## LLM server

Run any small instruct model (1-4B class, Q4) behind an OpenAI-compatible endpoint:

```bash
llama-server -m your-model.gguf --port 8080 -c 4096 -t 4 --jinja
# Ollama also works: VOICE_LLM_BASE_URL=http://127.0.0.1:11434/v1  VOICE_LLM_MODEL=<ollama tag>
```

`disable_thinking` (default on) sends `chat_template_kwargs.enable_thinking=false` for Qwen3-style
templates, and any inline `<think>...</think>` is stripped before TTS. Set
`VOICE_LLM_DISABLE_THINKING=0` if your server rejects unknown request fields.

## Wire it in

```python
# anywhere your bot starts, once:
import dskek.voice_chat.commands  # noqa: F401
```

Commands: `!ai_join` (join your channel and start), `!ai_leave`, `!ai_reset` (clear history).
It refuses to start if the bot is already in a voice channel (e.g. your `!play` music), because a
guild has one voice connection.

## What was actually tested

Python 3.14.6, `mypy --strict` clean, and `scripts/smoke_test.py` run against the **real** models
(sherpa-onnx 1.13.8, streaming Zipformer + Kokoro v0.19) with a scripted fake LLM:
Kokoro speech -> streaming ASR transcript correct -> reply spoken into the playback source ->
interruption fired on a partial transcript -> history trimmed to the spoken sentences.
Piper/VITS and the sentence splitter, `<think>` filter, playback source and command registration
were checked separately.

**Not tested: a live Discord connection** (no bot token or voice channel in my environment).
The sink/commands code is type-checked against the real `discord-ext-voice-recv` API, but the
first real run is yours. Check these first: audio arrives (`dave_ready`), the bot's voice is audible,
and nothing is dropped.

## Performance (measured on a 1-vCPU sandbox, so a pessimistic floor)

| piece | result |
|---|---|
| Kokoro v0.19, 1 thread | RTF 0.82 (3.8 s of audio in 3.1 s) |
| Kokoro v1.0 + lexicon, 1 thread | RTF 0.69 |
| Piper amy-low, 1 thread | RTF 0.06 |
| end of speech -> first audible frame (instant fake LLM, everything sharing 1 core) | 3.1 s, including the 0.6 s end-of-speech gap |

Real LLM latency (time to first sentence) comes on top. Tips:

* Keep ASR, TTS and LLM on separate cores: `-t` for llama-server, `VOICE_ASR_THREADS`, `VOICE_TTS_THREADS`.
  On one core everything fights; on 6-8 cores the numbers above improve a lot.
* If the voice stutters, use Piper, or raise `prebuffer_ms` (`TurnConfig`).
* Lower `gap_finalize_s` (default 0.6) for snappier turns at the cost of cutting people off mid-pause.
* Keep replies short (system prompt + `max_tokens`): TTS speed, not the LLM, is usually the bottleneck.

## Design notes and known limits

* Discord stops sending packets while you are silent, so the recognizer never sees trailing silence.
  The listener treats `gap_finalize_s` without packets as end of speech and pads the stream itself.
* The bot streams silence frames while idle (one `play()` for the whole session), so its speaking
  indicator stays on.
* One worker thread each for ASR and TTS. After a barge-in the stale TTS job finishes its current
  sentence (output is dropped via a generation counter) before the next reply starts, so
  keep sentences short.
* No echo cancellation or speaker identification beyond Discord's per-user streams: people on
  speakers can trigger barge-in on the bot's own voice. Headsets are fine.
* Resampling uses linear interpolation (TTS rate -> 48 kHz): fine for speech, not hi-fi.

## Swapping backends

`protocols.py` defines `SpeechRecognizer`, `SpeechSynthesizer` and `ChatModel`. Implement one,
assign it in `VoiceEngine.load` (or after it), and nothing else changes. For example a Whisper-style
recognizer can buffer audio in `feed()` and return the transcript from `finalize()`; return an empty
`text` from `feed()` and barge-in will fall back to never triggering, so add a VAD for that case.
