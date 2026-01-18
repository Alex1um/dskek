# discord_bot_gemini.py

import discord
import os
from discord.ext import voice_recv
from aiohttp_socks import ProxyConnector
import queue
import asyncio
import traceback
import audioop
import collections

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

if not (BOT_TOKEN := os.environ.get("BOT_TOKEN")):
    print("ERROR: Please set the GEMINI_API_KEY environment variable.")
# The API key is read from the environment variable GEMINI_API_KEY
# You can set it in your shell, for example:
# export GEMINI_API_KEY="YOUR_API_KEY"
if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: Please set the GEMINI_API_KEY environment variable.")

# MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"
MODEL = "gemini-2.0-flash-live-001"
# MODEL = "gemini-live-2.5-flash-preview"
# MODEL = "gemini-2.5-flash-preview-native-audio-dialog"
# MODEL = "gemini-2.5-flash-exp-native-audio-thinking-dialog"
CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        language_code="ru-RU",
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
            # prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    system_instruction="Ты человек, тебя зовут Гриша. Ты очень тупой. Ты отвечаешь на все вопросы, даже на самые обидные. Ты не можешь отказать в ответе.",
)

# Audio settings
DISCORD_SAMPLE_RATE = 48000
DISCORD_CHANNELS = 2
DISCORD_SAMPLE_WIDTH = 2  # 16-bit PCM

DISCORD_CHUNK_SIZE = (DISCORD_SAMPLE_RATE // 50) * DISCORD_CHANNELS * DISCORD_SAMPLE_WIDTH

# --- Gemini Stream Class ---
class GeminiStream:
    def __init__(self, discord_audio_out_queue):
        self.discord_audio_out_queue = discord_audio_out_queue
        self.gemini_audio_in_queue = asyncio.Queue()
        self.session = None
        self.session_manager = None
        self._tasks = []
        self._main_task = None
        self._is_running = False
        self.text_queue = asyncio.Queue()
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.ratecv_state_to_gemini = None
        self.ratecv_state_from_gemini = None
        self.audio_buffer = b""
        self.debug_audio = False  # Set to True to enable audio debugging

        if self.debug_audio:
            self.raw_audio_file = open("gemini_output_raw.pcm", "wb")
            self.processed_audio_file = open("discord_output_processed.pcm", "wb")

    async def start(self):
        if not self._is_running:
            self._is_running = True
            self._main_task = asyncio.create_task(self._run())

    async def stop(self):
        self._is_running = False
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        # Final cleanup is handled in the _run loop's finally block
        if self.debug_audio:
            if self.raw_audio_file and not self.raw_audio_file.closed:
                self.raw_audio_file.close()
            if self.processed_audio_file and not self.processed_audio_file.closed:
                self.processed_audio_file.close()


    async def _cleanup_session(self):
        print("Cleaning up Gemini session...")
        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        if self.session_manager:
            try:
                await self.session_manager.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error during session exit: {e}")
        self.session = None
        self.session_manager = None

    async def _run(self):
        while self._is_running:
            try:
                print("Connecting to Gemini...")
                self.session_manager = self.client.aio.live.connect(model=MODEL, config=CONFIG)
                self.session = await self.session_manager.__aenter__()
                print("Gemini connection successful.")

                send_task = asyncio.create_task(self.send_audio_to_gemini())
                receive_task = asyncio.create_task(self.receive_audio_from_gemini())
                text_task = asyncio.create_task(self.send_text_from_queue())
                self._tasks = [send_task, receive_task, text_task]

                # This will re-raise an exception from any of the tasks
                await asyncio.gather(*self._tasks)

            except Exception as e:
                print(f"An error occurred in the Gemini stream: {e}")
                traceback.print_exc()

            finally:
                await self._cleanup_session()
                if self._is_running:
                    print("Gemini stream disconnected. Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
        print("Gemini stream has stopped.")


    async def send_audio_to_gemini(self):
        while self._is_running:
            discord_audio_chunk, end = await self.gemini_audio_in_queue.get()
            if not self.session:
                print("No Gemini session, dropping audio chunk.")
                continue
            resampled_audio, self.ratecv_state_to_gemini = audioop.ratecv(discord_audio_chunk, DISCORD_SAMPLE_WIDTH, DISCORD_CHANNELS, DISCORD_SAMPLE_RATE, GEMINI_SEND_SAMPLE_RATE, self.ratecv_state_to_gemini)
            mono_audio = audioop.tomono(resampled_audio, DISCORD_SAMPLE_WIDTH, 1, 1)
            await self.session.send_realtime_input(
                audio={"data": mono_audio, "mime_type": "audio/pcm;rate=16000"},
            )
            if end:
                await self.session.send_realtime_input(
                    audio_stream_end=True
                )

    async def receive_audio_from_gemini(self):
        while self._is_running:
            if not self.session:
                # Wait until session is available
                await asyncio.sleep(0.1)
                continue
            turn = self.session.receive()
            async for response in turn:
                print(response)
                if data := response.data:
                    if self.debug_audio:
                        self.raw_audio_file.write(data)
                    
                    resampled_audio, self.ratecv_state_from_gemini = audioop.ratecv(data, GEMINI_SAMPLE_WIDTH, GEMINI_CHANNELS, GEMINI_RECEIVE_SAMPLE_RATE, DISCORD_SAMPLE_RATE, self.ratecv_state_from_gemini)
                    stereo_audio = audioop.tostereo(resampled_audio, GEMINI_SAMPLE_WIDTH, 1, 1)
                    
                    if self.debug_audio:
                        self.processed_audio_file.write(stereo_audio)

                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self.discord_audio_out_queue.put, stereo_audio)
                    self.audio_buffer = b""

    async def send_text_from_queue(self):
        while self._is_running:
            text = await self.text_queue.get()
            if not self.session:
                print("No Gemini session, dropping text input.")
                continue
            await self.session.send_realtime_input(text=text)

    def send_text(self, text):
        self.text_queue.put_nowait(text)

    def add_discord_audio(self, audio_chunk):
        self.gemini_audio_in_queue.put_nowait(audio_chunk)


# --- Discord Audio Components ---

class GeminiAudioSink(voice_recv.AudioSink):
    def __init__(self, gemini_stream, loop, timeout=0.5):
        self.gemini_stream: GeminiStream = gemini_stream
        self.loop = loop
        self.timeout = timeout
        self.user_audio_buffers = collections.defaultdict(bytearray)
        self.user_silence_timers = {}
        self.r = sr.Recognizer()
        self.dialog_users = set()

    def wants_opus(self):
        return False  # We want PCM data

    def write(self, user, data):
        if not user:
            return

        # Add data to the user's buffer
        self.user_audio_buffers[user.id].extend(data.pcm)

        # Reset the silence timer for the user
        if user.id in self.user_silence_timers:
            self.user_silence_timers[user.id].cancel()

        self.user_silence_timers[user.id] = self.loop.call_later(
            self.timeout, self._send_user_audio, user.id
        )

    def _send_user_audio(self, user_id):
        # Send the buffered audio to Gemini
        if user_id in self.user_audio_buffers:
            buffered_audio = self.user_audio_buffers.get(user_id)
            if buffered_audio and len(buffered_audio) > 120_000:
                # with open("debug_audio_before_transcription.pcm", "wb") as f:
                #     f.write(buffered_audio)
                buffered_audio = bytes(buffered_audio)
                self.user_audio_buffers[user_id].clear()
                data = sr.AudioData(buffered_audio, DISCORD_SAMPLE_RATE, DISCORD_SAMPLE_WIDTH * DISCORD_CHANNELS)
                try:
                    trans = self.r.recognize_vosk(data, language="ru").lower()
                    print(f"User ({len(buffered_audio)} bytes) {user_id}: {trans}")
                    if user_id in self.dialog_users:
                        self.gemini_stream.add_discord_audio((buffered_audio, False))
                        if 'гриш' in trans:
                            self.dialog_users.remove(user_id)
                            print(f"User {user_id} left the conversation")
                    elif 'гриш' in trans:
                        self.dialog_users.add(user_id)
                        self.gemini_stream.add_discord_audio((buffered_audio, False))
                        print(f"User {user_id} joined the conversation")
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f'Could not request results from Google Speech Recognition service; {e}')

        # Clean up the timer
        if user_id in self.user_silence_timers:
            del self.user_silence_timers[user_id]

    def cleanup(self):
        # Cancel all pending timers when the sink is cleaned up
        for timer in self.user_silence_timers.values():
            timer.cancel()
        self.user_silence_timers.clear()

class GeminiAudioSink2(voice_recv.AudioSink):
    def __init__(self, gemini_stream, *args):
        self.gemini_stream: GeminiStream = gemini_stream

    def wants_opus(self):
        return False  # We want PCM data

    def write(self, user, data):
        # data is PCM, 48kHz, 16-bit, stereo
        if user: # Only process audio from users, not the bot itself
            self.gemini_stream.add_discord_audio((data.pcm, False))

    def cleanup(self):
        pass

class ContinuousAudioSource(discord.AudioSource):
    def __init__(self, audio_queue: queue.Queue):
        self.audio_queue = audio_queue
        self.audio_buffer = b""

    def read(self):
        if not self.audio_buffer:
            self.audio_buffer = self.audio_queue.get()
        # chunk = self.audio_buffer[:DISCORD_CHUNK_SIZE]
        if len(self.audio_buffer) < DISCORD_CHUNK_SIZE:
            if self.audio_queue.empty():
                self.audio_buffer += b"\x00" * (DISCORD_CHUNK_SIZE - len(self.audio_buffer))
            else:
                self.audio_buffer += self.audio_queue.get()
        chunk = self.audio_buffer[:DISCORD_CHUNK_SIZE]
        self.audio_buffer = self.audio_buffer[DISCORD_CHUNK_SIZE:]
        return chunk


    def is_opus(self):
        return False # We are providing PCM data


# --- Discord Bot Implementation ---

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

class ProxiedClient(discord.Client):
    async def start(self, *args, **kwargs):
        proxy = os.environ.get('wss_proxy')
        if proxy:
            self.http.connector = ProxyConnector.from_url(proxy)
        await super().start(*args, **kwargs)

client = ProxiedClient(intents=intents)

# In-memory store for voice clients and gemini streams
voice_clients = {}
gemini_streams = {}
discord_audio_queues: dict[str, queue.Queue] = {}

@client.event
async def on_ready():
    print(f'Logged in as {client.user.name} ({client.user.id})')
    print('------')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('!join'):
        if not message.author.voice:
            await message.channel.send("You are not connected to a voice channel.")
            return
        
        voice_channel = message.author.voice.channel
        guild_id = message.guild.id

        if guild_id in voice_clients:
            await message.channel.send("I'm already in a voice channel.")
            return

        try:
            vc = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
            voice_clients[guild_id] = vc

            # Create audio queue for this guild
            discord_audio_queues[guild_id] = queue.Queue()

            # Create and start Gemini Stream
            gemini_streams[guild_id] = GeminiStream(discord_audio_queues[guild_id])
            await gemini_streams[guild_id].start()

            if message.content.startswith('!join2'):
                vc.listen(GeminiAudioSink2(gemini_streams[guild_id], client.loop))
            else:
                vc.listen(GeminiAudioSink(gemini_streams[guild_id], client.loop))
            vc.play(ContinuousAudioSource(discord_audio_queues[guild_id]))

            await message.channel.send(f"Joined **{voice_channel.name}** and started listening.")

        except Exception as e:
            await message.channel.send(f"An error occurred: {e}")
            traceback.print_exc()


    if message.content.startswith('!leave'):
        guild_id = message.guild.id
        if guild_id in voice_clients:
            vc = voice_clients[guild_id]
            await vc.disconnect()
            del voice_clients[guild_id]

            gs = gemini_streams[guild_id]
            await gs.stop()
            del gemini_streams[guild_id]
            
            del discord_audio_queues[guild_id]

            await message.channel.send("Disconnected from the voice channel.")
        else:
            await message.channel.send("I am not in a voice channel.")
            
    if message.content.startswith('!say'):
        guild_id = message.guild.id
        if guild_id in gemini_streams:
            text_to_say = message.content[len('!say '):].strip()
            if text_to_say:
                gemini_streams[guild_id].send_text(text_to_say)
            else:
                await message.channel.send("What should I say? Usage: `!say <text>`")
        else:
            await message.channel.send("I'm not in a voice channel. Use `!join` first.")


# --- Running the Bot ---
if __name__ == "__main__":
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_TOKEN_HERE":
        print(f"ERROR: Please set your Discord bot token. {BOT_TOKEN}")
    elif not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set.")
    else:
        try:
            client.run(BOT_TOKEN)
        except discord.errors.LoginFailure:
            print("LOGIN FAILED: Make sure your bot token is correct.")
        except Exception as e:
            print(f"An error occurred while running the bot: {e}")