from discord.message import Message
from openai import AsyncOpenAI

from dskek.discord_bot import bot
from dskek.env import OPENAI_ADDRESS, OPENAI_KEY, OPENAI_MODEL

client = AsyncOpenAI(api_key=OPENAI_KEY, base_url=OPENAI_ADDRESS)


@bot.event
async def on_message(message: Message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message):

        async with message.channel.typing():

            content = message.content.replace(bot.user.mention, "").strip()

            response = await client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {"role": "user", "content": content},
                ],
            )

            await message.reply(response.output[0].content[0].text)

    else:
        await bot.process_commands(message)


if __name__ == "__main__":
    import asyncio

    resp = asyncio.run(
        client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "user", "content": "Hi, how are you?"},
            ],
        )
    )
    print(resp.output[0].content[0].text)
