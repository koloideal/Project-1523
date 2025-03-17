import asyncio
import logging
import sqlite3
import time
import os

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandStart
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, Message
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

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
        # [KeyboardButton(text="Продолжить игру")],
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
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Функция для разделения длинного текста (если нужно)
def split_text(text, chunk_size=4096):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Обработчик команды /start
@dp.message(CommandStart())
async def start_command(message: Message, state: FSMContext):
    await state.set_state(Reg.start)
    await message.reply(
        "Добро пожаловать в игру! Выберите действие:\n\n"
        "🔹 'Начать игру' – начать новую игру.\n"
        "🔹 'Продолжить игру' – загрузить сохранённый прогресс.\n"
        "🔹 'Новая игра' – стереть старую историю и начать заново.",
        reply_markup=start_keyboard
    )

# Обработчик выбора действия (Reg.start)
@dp.message(Reg.start)
async def start_story(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if message.text == "Начать игру":
        # При выборе "Начать игру" удаляем любые существующие данные пользователя
        cursor.execute("DELETE FROM user_progress WHERE user_id = ?", (user_id,))
        conn.commit()
        # Генерируем новый уникальный thread_id
        new_thread_id = f"{message.chat.id}_{int(time.time())}"
        await state.update_data(thread_id=new_thread_id)
        # Вставляем новую запись
        cursor.execute("INSERT INTO user_progress (user_id, progress, story, thread_id) VALUES (?, 1, '', ?)",
                       (user_id, new_thread_id))
        conn.commit()
        await state.set_state(Reg.dialog)
        await handle_dialog(message, state)
        
    elif message.text == "Продолжить игру":
        cursor.execute("SELECT thread_id, progress, story FROM user_progress WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row is None or row[2] == "":
            await message.answer("❌ У вас нет сохранённых игр. Начните новую игру!", reply_markup=start_keyboard)
            await state.set_state(Reg.start)
        else:
            # Загружаем сохранённые данные и извлекаем последний ответ (story)
            await state.update_data(thread_id=row[0], progress=row[1], story=row[2])
            await message.answer("📜 Сохранённая история загружена. Продолжайте игру, напишите новое сообщение.")
            await state.set_state(Reg.dialog)
            # Не вызываем handle_dialog сразу, ждём нового ввода от пользователя
        
    elif message.text == "Новая игра":
        cursor.execute("DELETE FROM user_progress WHERE user_id = ?", (user_id,))
        conn.commit()
        await message.answer("🔄 Ваша старая история удалена. Начните новую игру!", reply_markup=start_keyboard)
        await state.set_state(Reg.start)

# Основной обработчик диалога (Reg.dialog)
@dp.message(Reg.dialog)
async def handle_dialog(message: Message, state: FSMContext):
    global retriever  # Гарантируем, что retriever доступен в функции
    user_id = message.from_user.id
    data = await state.get_data()
    thread_id = data.get("thread_id", str(message.chat.id))
    saved_story = data.get("story", "")
    
    # Получаем контекст из документа по тексту сообщения
    retrieved_docs = retriever.invoke(message.text)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    formatted_system_prompt = system_prompt.format(context=context)
    
    # Формируем сообщения для LLM:
    cursor.execute("SELECT progress, story FROM user_progress WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    if row is None or row[0] == 1:
        messages_chain = [
            SystemMessage(content=formatted_system_prompt),
            HumanMessage(content=message.text)
        ]
    else:
        # Передаем только последнее сообщение пользователя (без объединения со старой историей)
        messages_chain = [
            HumanMessage(content=message.text)
        ]
    
    # Обновляем прогресс
    if row is None:
        progress = 1
    else:
        progress = row[0] + 1
    cursor.execute("UPDATE user_progress SET progress = ? WHERE user_id = ?", (progress, user_id))
    conn.commit()
    
    config = {"configurable": {"thread_id": thread_id, "progress": progress}}
    time.sleep(2)
    
    results = app.invoke({"messages": messages_chain}, config)
    bot_response = results["messages"][-1].content
    
    # Обновляем историю: сохраняем только последний ответ нейросети
    cursor.execute("UPDATE user_progress SET story = ? WHERE user_id = ?", (bot_response, user_id))
    conn.commit()
    await state.update_data(story=bot_response)
    
    await message.answer(bot_response)

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
