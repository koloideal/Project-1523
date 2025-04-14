import asyncio
import logging
import sqlite3
import time
import os
import re

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandStart
from aiogram import F

from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

global retriever, bot, conn

import requests
from mistralai import Mistral
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage

# Устанавливаем ключ для Mistral
os.environ["MISTRAL_API_KEY"] = "XSFzvyw9LNYEjKYPFYFhYCzerqjeAr7Y"
llm = ChatMistralAI(model="mistral-small-latest")

# Подключаемся к базе данных и создаём таблицу (если ещё не создана)
conn = sqlite3.connect("users_data.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_progress (
        user_id INTEGER PRIMARY KEY,
        progress INTEGER DEFAULT 0,
        story TEXT DEFAULT '',
        thread_id TEXT DEFAULT ''
    )
""")
conn.commit()

# Определяем FSM состояния
class Reg(StatesGroup):
    start = State()    # Начальное состояние: выбор действия
    dialog = State()   # Игровой диалог (игра идёт)
    cont = State()     # Продолжить игру
    new = State()      # Новая игра (очистка истории)

# Клавиатура для выбора действий
start_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Начать игру")],
        [KeyboardButton(text="Продолжить игру")],
        [KeyboardButton(text="Новая игра")]
    ],
    resize_keyboard=True
)

# Инициализация бота и диспетчера
BOT_TOKEN = '7602719591:AAER_dkEQXD9x0O4RNnya5nzWss3RAnPqGE'
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

logging.basicConfig(level=logging.INFO)

# Загрузка сюжета, разбиение текста, создание retriever
loader = Docx2txtLoader("story.docx")
data = loader.load()
embeddings = MistralAIEmbeddings(model="mistral-embed")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()  # Глобальная переменная

# Формируем системное сообщение с плейсхолдером {context}
system_prompt = (
    "Ты чат-бот, который ведет игру под названием Crystals of Fate. "
    "Игра ведется от персонажа по имени Кай. "
    "По сюжету, Кай оказывается в центре борьбы между магическим Орденом Света, высокотехнологичными Технократами и таинственными Изгнанниками. "
    "Согласно лору игры, кай должен использовать древний Кристалл, чтобы установить порядок, продвинуть прогресс или разрушить оковы власти, позволяя миру самому решать свою судьбу. "
    "Ты должен вести диалог от лица ведущего игры, основываясь только на загруженном документе, игнорируя ответы игрока, не связанные с игрой. "
    "В начале игры, когда игрок напишет 'начать игру' или 'Новая история', основываясь на загруженном файле, в котором описан мир и сюжет игры, начни игру с самого начала, при этом сгенерируй другое начало. "
    "На основе ответов игрока генерируй сюжет дальше, основываясь только на предыдущих ответах игрока, не отступая от сюжета. "
    "Если тебе приходит сообщение, что 'нет предыдущих игр', то ответь пользователю, что у него нет сохранённых игр, и что он должен начать новую игру. "
    "В каждом сообщении генерируй продолжение сюжета и предлагай варианты развития, нумеруя их, спрашивая: 'Какое действие ты сделаешь?'. "
    "Также тебе будет передаваться значение — прогресс игрока, возвращай его в каждом сообщении, и если значение равно 15, то закончи историю, предоставив её финал, причем в этом финальном сообщении, в отличие от предыдущих, не должно быть вариантов ответов. \n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Инициализация цепочек LangChain (для генерации ответа)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

workflow = StateGraph(state_schema=MessagesState)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError),
)
def safe_llm_invoke(messages):
    return llm.invoke(messages)


def call_model(state: MessagesState):
    try:
        response = llm.invoke(state["messages"])
        return {"messages": response}
    except Exception as e:
        logging.error(f"Model error: {str(e)}")
        return {"messages": "⚠️ Ошибка генерации ответа"}
workflow.add_node("model", call_model)

workflow.set_entry_point("model")
workflow.set_finish_point("model")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Функция для разделения длинного текста (если нужно)
def split_text(text, chunk_size=4096):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Функция для преобразования числа в emoji (только для 1-9)
def number_to_emoji(number: int) -> str:
    return {
        1: "1️⃣",
        2: "2️⃣",
        3: "3️⃣"
    }.get(number, str(number))

# Функция для парсинга ответа, чтобы отделить основной текст от вариантов ответа.
def parse_response(response: str):
    marker = "Какое действие ты сделаешь?"
    if marker in response:
        parts = response.split(marker)
        main_text = parts[0].strip()
        options_raw = parts[1].strip()
        
        options = []
        pattern = re.compile(r'^\s*(\d+)[.)]?\s*(.+?)(?=\s*\d+[.)]?|\s*$)', re.MULTILINE)
        matches = pattern.findall(options_raw)
        
        for num, text in matches[:3]:
            options.append((int(num), text.strip()))
        
        return main_text, options
    else:
        return response, []

# Обработчик команды /start
@dp.message(CommandStart())
async def start_command(message: Message, state: FSMContext):
    await state.set_state(Reg.start)
    await message.reply(
        "🌟 Добро пожаловать в Crystals of Fate! 🌟\n\n"
        "🔹 Начать квест ✅ - погрузиться в новое приключение\n"
        "🔹 Продолжить ▶️ - вернуться к сохранённой игре\n"
        "🔹 Новая игра 🌍 - начать с чистого листа",
        reply_markup=start_keyboard
    )

# Обработчик выбора действия (Reg.start)
@dp.message(Reg.start)
async def start_story(message: Message, state: FSMContext):
    user_id = message.from_user.id
    try:
        if message.text == "Начать игру":
            with sqlite3.connect("users_data.db") as conn:
                cursor = conn.cursor()
                
                # Удаляем предыдущие записи
                cursor.execute("DELETE FROM user_progress WHERE user_id = ?", (user_id,))
                
                # Генерируем новый идентификатор сессии
                new_thread_id = f"{message.chat.id}_{int(time.time())}"
                
                # Вставляем новую запись с начальными значениями
                cursor.execute(
                    "INSERT INTO user_progress (user_id, progress, story, thread_id) VALUES (?, ?, ?, ?)",
                    (user_id, 1, json.dumps([]), new_thread_id)
                )
                conn.commit()

                # Обновляем состояние
                await state.update_data(
                    thread_id=new_thread_id,
                    progress=1,
                    story=json.dumps([])
                )
                await state.set_state(Reg.dialog)
                
                # Логируем успешный старт
                logging.info(f"New game started for user {user_id}")
                
                # Отправляем первое сообщение
                await handle_dialog(message, state)

        elif message.text == "Продолжить игру":
            with sqlite3.connect("users_data.db") as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT thread_id, progress, story FROM user_progress WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
                
            if not row or not row[2]:
                await message.answer("❌ Нет сохранённой игры. Начните новую!", reply_markup=start_keyboard)
                return
                
            # Парсим историю
            try:
                story_data = json.loads(row[2])
            except json.JSONDecodeError:
                story_data = []
                
            await state.update_data(
                thread_id=row[0],
                progress=row[1],
                story=story_data
            )
            await state.set_state(Reg.dialog)
            await message.answer("⏯ Продолжаем ваше приключение!")
            await handle_dialog(message, state)

        elif message.text == "Новая игра":
            with sqlite3.connect("users_data.db") as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM user_progress WHERE user_id = ?", (user_id,))
                conn.commit()
            
            await state.clear()
            await message.answer("🔄 Ваша старая история удалена. Начните новую игру!", reply_markup=start_keyboard)
            await state.set_state(Reg.start)

    except Exception as e:
        logging.error(f"Start error: {str(e)}")
        await message.answer("⚠️ Произошла ошибка. Попробуйте еще раз.")
import json
# Основной обработчик диалога (Reg.dialog)
@dp.message(Reg.dialog)
async def handle_dialog(message: Message, state: FSMContext):
    global retriever, bot
    user_id = message.from_user.id
    data = await state.get_data()
    if data.get("awaiting_custom_input"):
        await state.update_data(awaiting_custom_input=False)
        # Обрабатываем как обычное сообщение с введенным текстом
        fake_message = Message(
            text=message.text,  # Используем введенный текст как есть
            from_user=message.from_user,
            chat=message.chat,
            message_id=message.message_id,
            date=message.date,
            bot=bot
        )
        await handle_dialog(fake_message, state)
        return
    try:
        data = await state.get_data()
        thread_id = data.get("thread_id", str(message.chat.id))

        with sqlite3.connect("users_data.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT progress, story FROM user_progress WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            current_progress = row[0] if row else 1

            # Парсим историю из существующего поля story
            try:
                messages_history = json.loads(row[1]) if row and row[1] else []
            except (json.JSONDecodeError, TypeError):
                messages_history = []

            if current_progress >= 15:
                await message.answer("🎉 Вы достигли конца игры! Спасибо за участие!")
                return

            # Добавляем новое сообщение пользователя
            user_message = {"role": "user", "content": message.text}
            messages_history.append(user_message)
            
            # Формируем системное сообщение только для первого шага
            if current_progress == 1:
                retrieved_docs = retriever.invoke(message.text)
                context = "\n".join([doc.page_content for doc in retrieved_docs])
                system_message = {
                    "role": "system", 
                    "content": system_prompt.format(context=context)
                }
                full_chain = [system_message] + messages_history
            else:
                full_chain = messages_history[-6:]  # Берем последние 3 пары сообщений

            # Конвертируем в формат LangChain сообщений
            langchain_messages = []
            for msg in full_chain:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                else:
                    langchain_messages.append(HumanMessage(content=msg["content"]))

            # Отправляем запрос к модели
            try:
                config = {"configurable": {"thread_id": thread_id, "progress": current_progress}}
                results = app.invoke({"messages": langchain_messages}, config)
                assistant_response = results["messages"][-1].content

                # Добавляем ответ ассистента в историю
                messages_history.append({"role": "assistant", "content": assistant_response})

                # Обновляем данные в базе
                cursor.execute(
                    """UPDATE user_progress 
                       SET progress = ?, 
                           story = ?
                       WHERE user_id = ?""",
                    (current_progress + 1, json.dumps(messages_history), user_id)
                )
                conn.commit()

                # Парсим ответ
                main_text, options = parse_response(assistant_response)

                
                if not options:
                    await bot.send_message(chat_id=message.chat.id, text=main_text)
                    return 

                # Создаем клавиатуру
                buttons_row1 = []
                for num, text in options[:3]:
                    buttons_row1.append(InlineKeyboardButton(
                        text=number_to_emoji(num),
                        callback_data=f"option_{num}"
                    ))


                buttons_row2 = [
                    InlineKeyboardButton(
                        text="4️⃣ Свой вариант",
                        callback_data="custom_option"
                    )
                ]

                # Форматируем сообщение
                options_text = "\n".join([
                    f"{number_to_emoji(num)} {text}" for num, text in options[:3]
                ])
                response_text = (
                    f"{main_text}\n\n{options_text}"
                )

                await bot.send_message(
                    chat_id=message.chat.id,
                    text=response_text,
                    reply_markup=InlineKeyboardMarkup(
                        inline_keyboard=[
                            buttons_row1,
                            buttons_row2
                        ]
                    )
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    await message.answer("⚠️ Слишком много запросов. Подождите 10 секунд...")
                    await asyncio.sleep(10)
                else:
                    raise

            except Exception as e:
                logging.error(f"Ошибка модели: {str(e)}")
                await message.answer("⚠️ Ошибка генерации ответа")

    except Exception as e:
        logging.error(f"Общая ошибка: {str(e)}")
        await message.answer("⚠️ Произошла ошибка, попробуйте еще раз")


@dp.callback_query(F.data == "custom_option")
async def handle_custom_option(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    await state.update_data(awaiting_custom_input=True)
    await bot.send_message(
        chat_id=callback.message.chat.id,
        text="✏️ Напишите свой вариант:"
    )


# Обработчик callback_query для выбора варианта ответа
@dp.callback_query(F.data.startswith("option_"))
async def process_option(callback: types.CallbackQuery, state: FSMContext):
    try:
        option_number = callback.data.split("_")[1]
        await callback.answer()
        
        # Создаем фейковое сообщение с явным указанием бота
        fake_message = Message(
            text=option_number,
            from_user=callback.from_user,
            chat=callback.message.chat,
            message_id=callback.message.message_id,
            date=callback.message.date,
            bot=bot  # Важно добавить эту строку
        )
        
        await handle_dialog(fake_message, state)
        
    except Exception as e:
        logging.error(f"Ошибка обработки выбора: {str(e)}")
        await bot.send_message(
            chat_id=callback.message.chat.id,
            text="⚠️ Ошибка обработки выбора"
        )

# Обработчик для продолжения игры (состояние Reg.cont) – если понадобится отдельно
@dp.message(Reg.cont)
async def handle_continue(message: Message, state: FSMContext):
    user_id = message.from_user.id
    cursor.execute("SELECT thread_id, progress, story FROM user_progress WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    if row is None or row[2] == "":
        await message.answer("❌ У вас нет сохранённых игр. Начните новую!", reply_markup=start_keyboard)
        await state.set_state(Reg.start)
    else:
        await state.update_data(thread_id=row[0], progress=row[1], story=row[2])
        await message.answer(f"📜 Ваша сохранённая история:\n{row[2]}\n\nПродолжайте игру!")
        await state.set_state(Reg.dialog)
        await handle_dialog(message, state)

async def main():
    logging.info("Starting bot...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())