import asyncio
import logging
import sqlite3
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import requests
from mistralai import Mistral

api_key = "XSFzvyw9LNYEjKYPFYFhYCzerqjeAr7Y"
model = "mistral-small-latest"
bot_token = "7602719591:AAER_dkEQXD9x0O4RNnya5nzWss3RAnPqGE"

client = Mistral(api_key=api_key)
logging.basicConfig(level=logging.INFO)
bot = Bot(token=bot_token)
dp = Dispatcher()

conn = sqlite3.connect("users_data.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        user_id INTEGER,
        message TEXT
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS story (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        original TEXT,
        corrected TEXT
    )
""")
conn.commit()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я бот-новелла. Отвечай, и я буду запоминать твои слова!\n"
                         "Команды:\n"
                         "/history — посмотреть историю сообщений\n"
                         "/clear_history — очистить историю сообщений\n"
                         "/upload_story — загрузить файл с сюжетом\n"
                         "/delete_story — удалить текущий сюжет\n"
                         "/correct — исправить ответ бота")

@dp.message(Command("history"))
async def cmd_history(message: types.Message):
    user_id = message.from_user.id
    cursor.execute("SELECT message FROM messages WHERE user_id = ?", (user_id,))
    history = cursor.fetchall()
    
    if history:
        history_text = "\n".join([msg[0] for msg in history])
        await message.answer(f"📜 *Твоя история сообщений:*\n\n{history_text}", parse_mode="Markdown")
    else:
        await message.answer("У тебя пока нет сохранённых сообщений.")

@dp.message(Command("clear_history"))
async def cmd_clear_history(message: types.Message):
    user_id = message.from_user.id
    cursor.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    conn.commit()
    await message.answer("🗑 Твоя история сообщений очищена!")

@dp.message(Command("correct"))
async def cmd_correct(message: types.Message):
    parts = message.text.split("\n", 2)
    if len(parts) < 3:
        await message.answer("Формат: /correct <оригинальный ответ>\n<правильный ответ>")
        return

    original_text = parts[1]
    corrected_text = parts[2]

    cursor.execute("INSERT INTO corrections (user_id, original, corrected) VALUES (?, ?, ?)", 
                   (message.from_user.id, original_text, corrected_text))
    conn.commit()

    await message.answer("✅ Исправление сохранено!")

@dp.message(Command("upload_story"))
async def cmd_upload_story(message: types.Message):
    await message.answer("Отправь мне `.txt` файл с сюжетом.")

@dp.message(lambda message: message.document and message.document.mime_type == "text/plain")
async def handle_text_file(message: types.Message):
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    file_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"

    response = requests.get(file_url)
    story_text = response.text

    cursor.execute("DELETE FROM story")
    cursor.execute("INSERT INTO story (text) VALUES (?)", (story_text,))
    conn.commit()

    await message.answer("✅ Сюжет успешно загружен!")

@dp.message(Command("delete_story"))
async def cmd_delete_story(message: types.Message):
    cursor.execute("DELETE FROM story")
    conn.commit()
    await message.answer("❌ Сюжет удалён!")

@dp.message()
async def message_handler(msg: types.Message):
    user_id = msg.from_user.id
    user_message = msg.text

    cursor.execute("INSERT INTO messages (user_id, message) VALUES (?, ?)", (user_id, user_message))
    conn.commit()

    cursor.execute("SELECT message FROM messages WHERE user_id = ?", (user_id,))
    history = cursor.fetchall()
    chat_history = []
    for m in history:
        if m[0] and isinstance(m[0], str):
            chat_history.append({"role": "user", "content": m[0]})


    cursor.execute("SELECT text FROM story LIMIT 1")
    story_data = cursor.fetchone()
    if story_data:
        chat_history.insert(0, {"role": "system", "content": story_data[0]})

    chat_response = client.chat.complete(
        model=model,
        messages=chat_history
    )
    
    response_text = chat_response.choices[0].message.content

    cursor.execute("SELECT corrected FROM corrections WHERE original = ?", (response_text,))
    correction = cursor.fetchone()
    if correction:
        response_text = correction[0]

    await msg.answer(response_text, parse_mode="Markdown")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
