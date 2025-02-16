import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.methods import DeleteWebhook
from aiogram import types, F, Router
from aiogram.types import Message
from aiogram.filters import Command
import requests
from mistralai import Mistral

api_key = "XSFzvyw9LNYEjKYPFYFhYCzerqjeAr7Y"
model = "mistral-small-latest"

client = Mistral(api_key=api_key)

logging.basicConfig(level=logging.INFO)
bot = Bot(token="7602719591:AAER_dkEQXD9x0O4RNnya5nzWss3RAnPqGE")
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    botToken = '7602719591:AAER_dkEQXD9x0O4RNnya5nzWss3RAnPqGE'
    channel = 'direcode'

    url = f"https://api.telegram.org/bot{botToken}/getChatMembersCount?chat_id=@{channel}"
    response = requests.get(url)
    data = response.json()
    memberscount = data['result']
    print(memberscount)

    await bot.send_message(message.chat.id)


@dp.message()
async def message_handler(msg: Message):
    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": msg.text,
            }
        ]
    )
    await bot.send_message(msg.chat.id, chat_response.choices[0].message.content, parse_mode="Markdown")


async def main():
    await bot(DeleteWebhook(drop_pending_updates=True))
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
