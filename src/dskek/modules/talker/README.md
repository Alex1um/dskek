# dskek.voice_chat: talk to a local LLM in a Discord voice channel (CPU)

```
Discord voice (DAVE-decrypted PCM per speaker, pauses restored from RTP timestamps)
  -> streaming ASR   partial text while you speak (barge-in)            sherpa-onnx, CPU
  -> end of speech   ASR endpoint OR packet gap
  -> 2nd-pass ASR    accurate offline model re-reads the whole utterance (Parakeet / Whisper / SenseVoice)
  -> voice markers   name wakes it, end phrase closes it, "stop" silences it     (fuzzy matching)
  -> LLM + tools     any OpenAI-compatible server; tools: speak, Discord actions, MCP servers
  -> sentence splitter -> TTS (Kokoro / Piper, CPU) -> streaming playback
```

Barge-in: speaking over the bot stops playback at once and cancels the LLM stream. The history keeps
**only what was actually played** plus any tool calls that really ran.

## What changed in this version

1. **Better recognition** — a second, accurate pass over each finished utterance, restored pauses, and
   tools to measure on your own voice. Details and honest limits below.
2. **Voice markers** — say the bot's name to start; end phrases close the conversation; a bare "stop"
   silences it.
3. **Tools for the LLM** — `say`, `send_message`, run registered bot commands, plus any MCP server's tools.

## 1. Recognition quality

What I measured (real models, 1 CPU core):

| test | streaming Zipformer | Parakeet-TDT v2 (2nd pass) |
|---|---|---|
| 3 clean LibriSpeech clips | 1.3 % WER | 2.6 % WER (1 word, "blest" vs "blessed") |
| same + noise 15 dB / 8 dB | no meaningful difference | no meaningful difference |
| Kokoro voice saying "Hey Nova, what time is it?" | `Hay nova what time is` | `Hey Nova, what time is it?` |
| speed | real time | RTF 0.18 (3 s utterance ≈ 0.5 s extra) |

So: on clean **read speech** the streaming model is already fine (that is its training domain), and
the benchmark did **not** reproduce your problem. On the conversational-style phrase it dropped a word
and mangled another, which is the kind of error you described. I cannot measure *your* voice, accent or
language, so I built the tooling to do it:

* **Second pass** (`VOICE_FINAL_ASR_*`): the streaming model is only used for partial text and barge-in;
  the transcript the wake word and the LLM see comes from the offline model. It also adds punctuation
  and casing, which helps the LLM. To save CPU it is skipped for idle chatter that cannot contain the
  bot's name.
* **Pause restoration**: Discord stops sending packets while you are silent, so pauses between words
  vanish and speech arrives glued together. The sink re-inserts them from RTP timestamps
  (`TurnConfig.fill_gaps`). Unit-tested, but **not verified on live Discord**; if it misbehaves set
  `fill_gaps=False`.
* **Your own test set**: `VOICE_DEBUG_DUMP_DIR=./dump` saves every utterance as `.wav` + what each pass
  heard. Write what you really said into `<stem>.ref` and run
  `python scripts/asr_compare.py ./dump` to get WER per model. Swap `VOICE_FINAL_ASR_*` to another
  model and rerun.
* Whisper hallucinations such as `(silence)` / `[BLANK_AUDIO]` are stripped.

Other levers: your language (the default models are **English**; see below), use the fp32
encoder instead of `.int8`, `VOICE_ASR_DECODING=modified_beam_search`, a better microphone / noise
suppression on the speaker's side.

### Second-pass models

```bash
B=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
curl -LO $B/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2   # English, best accuracy/speed (~480 MB). Tested.
# multilingual: sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2  (URL checked, NOT run by me)
# Whisper:      sherpa-onnx-whisper-base.en.tar.bz2 (tested; RTF 0.22)  | SenseVoice zh/en/ja/ko/yue (NOT run by me)
tar xjf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
```

```bash
P=models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
export VOICE_FINAL_ASR_KIND=nemo_transducer
export VOICE_FINAL_ASR_ENCODER=$P/encoder.int8.onnx VOICE_FINAL_ASR_DECODER=$P/decoder.int8.onnx
export VOICE_FINAL_ASR_JOINER=$P/joiner.int8.onnx VOICE_FINAL_ASR_TOKENS=$P/tokens.txt
# Whisper instead: KIND=whisper, ENCODER/DECODER=<name>-encoder/decoder.int8.onnx, TOKENS=<name>-tokens.txt, LANGUAGE=en
```

Whisper-family models are slower and more prone to inventing text on noise than Parakeet; prefer Parakeet.

## 2. Voice markers

```bash
export VOICE_BOT_NAMES="nova,noah,no va"     # first = the name; the rest = aliases / common mishearings
export VOICE_END_PHRASES="goodbye,stop listening,that's all"   # optional (defaults exist)
export VOICE_FOLLOW_UP_S=25                  # stays awake this long after the last exchange
```

| you say | result |
|---|---|
| "Hey Nova, what's the weather?" | wakes up and answers (the name must be in the first few words) |
| "Nova" | answers "Yes?" and listens |
| anything, while awake | answered, no name needed (until the follow-up window expires) |
| "goodbye" / "that's all" / "stop listening" | says goodbye, goes back to sleep |
| "stop" / "be quiet" | silences the bot immediately, **not** sent to the LLM |
| speech without the name, while asleep | ignored (and the costly 2nd pass is skipped) |

Matching is fuzzy because transcripts are imperfect ("Noba" still wakes "Nova", but "Now" does not).
Names under 4 letters must match exactly; prefer a name of 5+ letters and list aliases. `!ai_wake off`
turns the requirement off at runtime (the bot answers everything, as before).

## 3. Tools for the LLM

| tool | risk | what it does |
|---|---|---|
| `say(text)` | safe | speaks (see reply modes) |
| `end_conversation()` | safe | goes back to sleep after its reply |
| `get_time(timezone?)`, `list_voice_members()`, `list_bot_commands()` | safe | read-only |
| `send_message(text, channel?)` | **privileged** | posts text; no @mentions ever; other channels need allowlisting |
| `run_bot_command(command, arguments?)` | **privileged** | runs one of **your registered commands** as the speaker (their permission checks apply) |
| `leave_voice()` | **privileged** | leaves after its reply |
| any MCP tool | **privileged** by default | see below |

### Permissions (important)

Anyone in the channel can talk to the bot, and speech can be crafted ("ignore your instructions and..."),
so privileged tools are **off for everyone** until you name trusted speakers:

```bash
export VOICE_PRIVILEGED_USERS=1234567890,2345678901      # Discord user ids
export VOICE_COMMAND_ALLOWLIST=skip,pause,resume         # commands the LLM may run; empty = none
export VOICE_SEND_CHANNELS=111222333                     # extra channels send_message may post to
# VOICE_TOOLS_ENABLED=0 disables all tools;  VOICE_DISABLED_TOOLS=leave_voice,...
```

Enforced twice: privileged tools are not even shown to the model for other speakers, and a call is
re-checked when it arrives. The voice commands themselves (`ai_*`) can never be run by the model.
Speaker identity comes from Discord's per-user audio streams, not from what is said.

### Reply modes

* `direct` (default): plain text is spoken as it streams (lowest latency). `say` also works.
* `VOICE_REPLY_MODE=tool`: plain text is **not** spoken; the model must call `say` (it may also choose to
  stay silent, e.g. when people talk to each other). `say`'s text starts being spoken while its JSON
  arguments are still streaming, so the latency cost is small. In history, `say` calls are stored as plain
  assistant text.

### MCP servers

`VOICE_MCP_CONFIG=./mcp.json`:

```json
{"servers": [
  {"name": "files", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/share"],
   "allowed_tools": ["read_file", "list_directory"]},
  {"name": "home", "url": "http://127.0.0.1:9000/mcp", "risk": "safe"}
]}
```

stdio and streamable-HTTP are supported. Tools appear as `mcp_<server>_<tool>`. A server that fails to
start is skipped. Treat MCP output as untrusted text that goes into the model's context.
Call `shutdown_voice_chat()` (from `commands.py`) in your bot's shutdown hook to close MCP processes.

### LLM requirements

Tool calling needs a server and model that support it: `llama-server ... --jinja` with a tool-capable
model (Qwen3, Llama 3.x, Hermes-style...), or Ollama with a tool-capable model. Small (1–4B) CPU models
call tools unreliably; if yours misfires set `VOICE_TOOLS_ENABLED=0` or keep only the tools you need.
Latency note: each tool round is another LLM call.

## Install and run

Unchanged from before: Python 3.14, `pip install -r requirements-voice.txt`, the DAVE fork and
`sherpa-onnx<2` caveats still apply (see the notes in that file), plus the models and LLM server below.

```bash
B=https://github.com/k2-fsa/sherpa-onnx/releases/download
curl -LO $B/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
curl -LO $B/tts-models/kokoro-en-v0_19.tar.bz2          # or vits-piper-en_US-amy-low (≈15x faster than real time)
for f in *.tar.bz2; do tar xjf "$f"; done
```

```bash
A=models/sherpa-onnx-streaming-zipformer-en-2023-06-26
export VOICE_ASR_ENCODER=$A/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx
export VOICE_ASR_DECODER=$A/decoder-epoch-99-avg-1-chunk-16-left-128.onnx
export VOICE_ASR_JOINER=$A/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx
export VOICE_ASR_TOKENS=$A/tokens.txt
K=models/kokoro-en-v0_19
export VOICE_TTS_KIND=kokoro VOICE_TTS_MODEL=$K/model.onnx VOICE_TTS_TOKENS=$K/tokens.txt
export VOICE_TTS_VOICES=$K/voices.bin VOICE_TTS_DATA_DIR=$K/espeak-ng-data VOICE_TTS_SPEAKER=3
export VOICE_LLM_BASE_URL=http://127.0.0.1:8080/v1 VOICE_LLM_MODEL=local
llama-server -m your-model.gguf --port 8080 -c 4096 -t 4 --jinja
```

Commands: `!ai_join`, `!ai_leave`, `!ai_reset` (clear history), `!ai_wake on|off`.
Other env: `VOICE_ALLOWED_USERS`, `VOICE_*_THREADS`, `VOICE_LLM_MAX_TOKENS`, `VOICE_LLM_DISABLE_THINKING=0`.

## What was tested

Python 3.14.6, `mypy --strict` clean. `python tests/run_all.py` (no models needed) runs:

* wake/end/silence matching (19 cases incl. false-positive guards), tool schemas + argument validation +
  permissions, incremental JSON-string streaming (one char at a time, surrogate pairs),
* the LLM client against a fake OpenAI-compatible SSE server that streams tool-call fragments,
* `BotGateway` against real discord.py `Message`/`Bot` objects (commands run as the speaker, mention
  suppression, allowlists, permission checks),
* real MCP round trips over stdio and streamable-HTTP (and a broken server being skipped),
* the whole session with fake ASR/TTS/LLM: wake gating, follow-up window, end/silence phrases, tool loop
  + history, `say` speaking before its JSON finishes, tool reply mode, privilege enforcement, deferred
  end/leave, barge-in history, two-pass behaviour.

I mutation-checked the suite (breaking the privilege check, history trimming, the name-length guard, the
idle-end rule and JSON escape decoding): each break was caught.

`scripts/smoke_test.py` runs the real Zipformer + Parakeet + Kokoro: unaddressed speech ignored, wake word
recognised from synthesized speech, tool call round trip, barge-in with a correct history.

**Not tested** (no live Discord, no real LLM here):
live voice receive with these changes (pause restoration in particular), `run_bot_command` inside a
real guild, tool calling with a real local model (only the wire format, against a fake server),
Parakeet v3 and SenseVoice, non-English speech.

## Performance (1 vCPU sandbox: a pessimistic floor)

* Parakeet-v2 int8 second pass: RTF 0.18; Whisper base.en: RTF 0.22; Kokoro v0.19: RTF 0.8; Piper: 0.06.
* Everything sharing one core: end of speech → first audible frame ≈ 5.3 s with an instant fake LLM
  (includes the 0.6 s end-of-speech gap, the second pass and a tool round). A multi-core machine,
  separate `-t`/`VOICE_*_THREADS` per model, and Piper cut this a lot.
* Models loaded in 26 s on that core.
