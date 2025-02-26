import asyncio
import logging
import sqlite3
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
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
    CREATE TABLE IF NOT EXISTS story (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_progress (
        user_id INTEGER PRIMARY KEY,
        progress INTEGER DEFAULT 0
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        user_id INTEGER,
        message TEXT
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

start_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Начать игру")],
        [KeyboardButton(text="Продолжить игру")]
    ],
    resize_keyboard=True
)

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    cursor.execute("INSERT OR IGNORE INTO user_progress (user_id, progress) VALUES (?, 0)", (user_id,))
    conn.commit()
    
    await message.answer(
        "Добро пожаловать в новую игру, чтобы начать нажми 'Начать игру'!\n\n"
        "Если хотите продолжить игру, нажмите кнопку 'Продолжить игру'.\n\n"
        "Чтобы узнать список всех команд, используй /commands.",
        reply_markup=start_keyboard
    )

@dp.message(Command("commands"))
async def cmd_commands(message: types.Message):
    commands_list = (
        "📜 Список команд:\n\n"
        "/start — начать взаимодействие с ботом\n"
        "/commands — показать список всех команд\n"
        "/history — посмотреть историю сообщений\n"
        "/clear_history — очистить историю сообщений\n"
        "/upload_story — загрузить файл с сюжетом\n"
        "/delete_story — удалить текущий сюжет\n"
        "/correct — исправить ответ бота"
    )
    await message.answer(commands_list)  # Без parse_mode="Markdown"

@dp.message(Command("history"))
async def cmd_history(message: types.Message):
    user_id = message.from_user.id
    cursor.execute("SELECT message FROM messages WHERE user_id = ?", (user_id,))
    history = cursor.fetchall()
    
    if history:
        history_text = "\n".join([msg[0] for msg in history])
        await message.answer(f"📜 Твоя история сообщений:\n\n{history_text}")
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

@dp.message(lambda message: message.text == "Начать игру")
async def start_game(message: types.Message):
    user_id = message.from_user.id
    user_message = "Начать игру"

    cursor.execute("INSERT INTO messages (user_id, message) VALUES (?, ?)", (user_id, user_message))
    conn.commit()

    cursor.execute("SELECT progress FROM user_progress WHERE user_id = ?", (user_id,))
    progress = cursor.fetchone()
    if not progress:
        progress = 0
    else:
        progress = progress[0]

    cursor.execute("SELECT text FROM story LIMIT 1")
    story_data = cursor.fetchone()
    if story_data:
        story_text = story_data[0]
        story_parts = story_text.split("\n\n")
        if progress < len(story_parts):
            current_story_part = story_parts[progress]
        else:
            current_story_part = "Сюжет завершен!"
    else:
        current_story_part = "Сюжет не загружен."

    cursor.execute("UPDATE user_progress SET progress = ? WHERE user_id = ?", (progress + 1, user_id))
    conn.commit()

    cursor.execute("SELECT message FROM messages WHERE user_id = ?", (user_id,))
    history = cursor.fetchall()
    
    MAX_HISTORY = 15
    chat_history = []
    
    for m in history[-MAX_HISTORY:]:
        if m[0] and isinstance(m[0], str):
            chat_history.append({"role": "user", "content": m[0]})

    chat_history.insert(0, {"role": "system", "content": current_story_part})

    chat_response = client.chat.complete(
        model=model,
        messages=chat_history
    )
    
    response_text = chat_response.choices[0].message.content

    cursor.execute("SELECT corrected FROM corrections WHERE user_id = ? AND original = ?", (user_id, response_text))
    correction = cursor.fetchone()
    if correction:
        response_text = correction[0]

    await message.answer(response_text)

@dp.message(lambda message: message.text == "Продолжить игру")
async def continue_game(message: types.Message):
    user_id = message.from_user.id
    user_message = "Продолжить игру"

    cursor.execute("INSERT INTO messages (user_id, message) VALUES (?, ?)", (user_id, user_message))
    conn.commit()

    cursor.execute("SELECT progress FROM user_progress WHERE user_id = ?", (user_id,))
    progress = cursor.fetchone()
    if not progress:
        progress = 0
    else:
        progress = progress[0]

    cursor.execute("SELECT text FROM story LIMIT 1")
    story_data = cursor.fetchone()
    if story_data:
        story_text = story_data[0]
        story_parts = story_text.split("\n\n")
        if progress < len(story_parts):
            current_story_part = story_parts[progress]
        else:
            current_story_part = "Сюжет завершен!"
    else:
        current_story_part = "Сюжет не загружен."

    cursor.execute("UPDATE user_progress SET progress = ? WHERE user_id = ?", (progress + 1, user_id))
    conn.commit()

    cursor.execute("SELECT message FROM messages WHERE user_id = ?", (user_id,))
    history = cursor.fetchall()
    
    MAX_HISTORY = 15
    chat_history = []
    
    for m in history[-MAX_HISTORY:]:
        if m[0] and isinstance(m[0], str):
            chat_history.append({"role": "user", "content": m[0]})

    chat_history.insert(0, {"role": "system", "content": current_story_part})

    chat_response = client.chat.complete(
        model=model,
        messages=chat_history
    )
    
    response_text = chat_response.choices[0].message.content

    cursor.execute("SELECT corrected FROM corrections WHERE user_id = ? AND original = ?", (user_id, response_text))
    correction = cursor.fetchone()
    if correction:
        response_text = correction[0]

    await message.answer(response_text)

@dp.message()
async def message_handler(msg: types.Message):
    user_id = msg.from_user.id
    user_message = msg.text

    cursor.execute("INSERT INTO messages (user_id, message) VALUES (?, ?)", (user_id, user_message))
    conn.commit()

    cursor.execute("SELECT progress FROM user_progress WHERE user_id = ?", (user_id,))
    progress = cursor.fetchone()
    if not progress:
        progress = 0
    else:
        progress = progress[0]

    cursor.execute("SELECT text FROM story LIMIT 1")
    story_data = cursor.fetchone()
    if story_data:
        story_text = story_data[0]
        story_parts = story_text.split("\n\n")
        if progress < len(story_parts):
            current_story_part = story_parts[progress]
        else:
            current_story_part = "Сюжет завершен!"
    else:
        current_story_part = "Сюжет не загружен."

    cursor.execute("UPDATE user_progress SET progress = ? WHERE user_id = ?", (progress + 1, user_id))
    conn.commit()

    cursor.execute("SELECT message FROM messages WHERE user_id = ?", (user_id,))
    history = cursor.fetchall()
    
    MAX_HISTORY = 15
    chat_history = []
    
    for m in history[-MAX_HISTORY:]:
        if m[0] and isinstance(m[0], str):
            chat_history.append({"role": "user", "content": m[0]})

    chat_history.insert(0, {"role": "system", "content": current_story_part})

    chat_response = client.chat.complete(
        model=model,
        messages=chat_history
    )
    
    response_text = chat_response.choices[0].message.content

    cursor.execute("SELECT corrected FROM corrections WHERE user_id = ? AND original = ?", (user_id, response_text))
    correction = cursor.fetchone()
    if correction:
        response_text = correction[0]

    await msg.answer(response_text)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())