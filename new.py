import asyncio
import logging
import sqlite3
import time
import os
from datetime import datetime

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

os.environ["MISTRAL_API_KEY"] = "XSFzvyw9LNYEjKYPFYFhYCzerqjeAr7Y"
llm = ChatMistralAI(model="mistral-small-latest")

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
cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_stats (
        user_id INTEGER PRIMARY KEY,
        total_decisions INTEGER DEFAULT 0,
        deaths INTEGER DEFAULT 0,
        endings_unlocked INTEGER DEFAULT 0,
        last_played TEXT DEFAULT '',
        threat_scheduled INTEGER DEFAULT 0
    )
""")
conn.commit()

class Reg(StatesGroup):
    start = State()
    dialog = State()
    cont = State()
    new = State()

start_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="✨ Начать квест ✅")],
        [KeyboardButton(text="📜 Продолжить ▶️")], 
        [KeyboardButton(text="🔄 Новая игра 🌍")]
    ],
    resize_keyboard=True,
    input_field_placeholder="🎮 Выбери действие..."
)

final_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🔄 Начать новую игру"), KeyboardButton(text="📊 Показать статистику")],
        [KeyboardButton(text="💀 Смертельные сценарии"), KeyboardButton(text="🌟 Секретные концовки")]
    ],
    resize_keyboard=True,
    input_field_placeholder="Выбери действие после финала..."
)

BOT_TOKEN = '7602719591:AAER_dkEQXD9x0O4RNnya5nzWss3RAnPqGE'
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

logging.basicConfig(level=logging.INFO)

loader = Docx2txtLoader("story.docx")
data = loader.load()
embeddings = MistralAIEmbeddings(model="mistral-embed")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

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

def split_text(text, chunk_size=4096):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


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

@dp.message(Reg.start)
async def start_story(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if message.text == "✨ Начать квест ✅":
        cursor.execute("DELETE FROM user_progress WHERE user_id = ?", (user_id,))
        conn.commit()
        new_thread_id = f"{message.chat.id}_{int(time.time())}"
        await state.update_data(thread_id=new_thread_id)
        cursor.execute("INSERT INTO user_progress (user_id, progress, story, thread_id) VALUES (?, 1, '', ?)",
                      (user_id, new_thread_id))
        cursor.execute("INSERT OR IGNORE INTO user_stats (user_id) VALUES (?)", (user_id,))
        conn.commit()
        await state.set_state(Reg.dialog)
        await handle_dialog(message, state)
        
    elif message.text == "📜 Продолжить ▶️":
        cursor.execute("SELECT thread_id, progress, story FROM user_progress WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row is None or row[2] == "":
            await message.answer("❌ У вас нет сохранённых игр. Начните новую игру!", reply_markup=start_keyboard)
            await state.set_state(Reg.start)
        else:
            await state.update_data(thread_id=row[0], progress=row[1], story=row[2])
            await message.answer("📜 Сохранённая история загружена. Продолжайте игру, напишите новое сообщение.")
            await state.set_state(Reg.dialog)
            
    elif message.text == "🔄 Новая игра 🌍":
        cursor.execute("DELETE FROM user_progress WHERE user_id = ?", (user_id,))
        conn.commit()
        await message.answer("🔄 Ваша старая история удалена. Начните новую игру!", reply_markup=start_keyboard)
        await state.set_state(Reg.start)

@dp.message(Reg.dialog)
async def handle_dialog(message: Message, state: FSMContext):
    user_id = message.from_user.id
    data = await state.get_data()
    
    cursor.execute("SELECT progress FROM user_progress WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    current_progress = row[0] if row else 1

    if current_progress >= 15:
        await generate_final_message(message, state, user_id)
        return
        
    cursor.execute("""
        INSERT OR IGNORE INTO user_stats (user_id) VALUES (?);
        UPDATE user_stats SET total_decisions = total_decisions + 1 WHERE user_id = ?;
    """, (user_id, user_id))
    conn.commit()
    
    retrieved_docs = retriever.invoke(message.text)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    formatted_system_prompt = system_prompt.format(context=context)
    
    messages_chain = [
        SystemMessage(content=formatted_system_prompt),
        HumanMessage(content=message.text)
    ] if current_progress == 1 else [HumanMessage(content=message.text)]
    
    new_progress = current_progress + 1
    cursor.execute("UPDATE user_progress SET progress = ? WHERE user_id = ?", (new_progress, user_id))
    conn.commit()
    
    config = {"configurable": {"thread_id": data.get("thread_id", str(message.chat.id))}}
    results = app.invoke({"messages": messages_chain}, config)
    bot_response = results["messages"][-1].content
    
    if any(word in bot_response.lower() for word in ["погиб", "умер", "смерть"]):
        cursor.execute("UPDATE user_stats SET deaths = deaths + 1 WHERE user_id = ?", (user_id,))
        conn.commit()
    
    cursor.execute("UPDATE user_progress SET story = ? WHERE user_id = ?", (bot_response, user_id))
    conn.commit()
    await message.answer(bot_response)


async def generate_final_message(message: Message, state: FSMContext, user_id: int):
    cursor.execute("SELECT story FROM user_progress WHERE user_id = ?", (user_id,))
    story = cursor.fetchone()[0]
    
    final_prompt = (
        f"Игрок завершил историю со следующими ключевыми моментами:\n{story}\n"
        "Сгенерируй эпичный финал с учётом выбранного пути. "
        "Упомяни 2-3 главных решения игрока. "
        "Добавь философское заключение о последствиях выбора. "
        "В конце добавь секретное пророчество о возможном будущем (1 предложение)."
    )
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=final_prompt)])
    final_text = response.content
    
    cursor.execute("""
        UPDATE user_stats 
        SET endings_unlocked = endings_unlocked + 1, 
            last_played = datetime('now')
        WHERE user_id = ?
    """, (user_id,))
    conn.commit()
    
    final_message = (
        f"🎭 *ФИНАЛЬНАЯ СЦЕНА* 🎭\n\n"
        f"{final_text}\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "Ты достиг конца этой истории, но мир Crystals of Fate продолжает жить...\n"
        "Что ты хочешь сделать дальше?"
    )
    
    await message.answer(final_message, reply_markup=final_keyboard, parse_mode="Markdown")
    await state.set_state(Reg.start)
    
    asyncio.create_task(schedule_threat_notification(user_id))

async def schedule_threat_notification(user_id: int):
    await asyncio.sleep(86400)
    try:
        cursor.execute("SELECT threat_scheduled FROM user_stats WHERE user_id = ?", (user_id,))
        if cursor.fetchone()[0] == 0:
            await bot.send_message(
                user_id,
                "🌑 *Тревожное предупреждение*\n\n"
                "Прошло ровно 24 часа с момента завершения твоего квеста...\n"
                "В Бездне пробудилось нечто древнее. Кристалл Судьбы снова зовет тебя!\n\n"
                "Напиши /start чтобы начать новое приключение!",
                parse_mode="Markdown"
            )
            cursor.execute("UPDATE user_stats SET threat_scheduled = 1 WHERE user_id = ?", (user_id,))
            conn.commit()
    except Exception as e:
        logging.error(f"Не удалось отправить уведомление: {e}")


@dp.message(lambda message: message.text == "📊 Показать статистику")
async def show_stats(message: Message):
    user_id = message.from_user.id
    cursor.execute("""
        SELECT us.total_decisions, us.deaths, us.endings_unlocked, 
               up.progress, up.story
        FROM user_stats us
        LEFT JOIN user_progress up ON us.user_id = up.user_id
        WHERE us.user_id = ?
    """, (user_id,))
    stats = cursor.fetchone()
    
    if stats and stats[0] is not None:
        total_decisions, deaths, endings, progress, story = stats
        decisions_in_story = story.count("1.") + story.count("2.") + story.count("3.") if story else 0
        
        response = (
            f"📜 *Твоя статистика в Crystals of Fate*:\n\n"
            f"• Всего решений: {total_decisions + decisions_in_story}\n"
            f"• Смертельных исходов: {deaths}\n"
            f"• Открыто концовок: {endings}\n"
            f"• Макс. прогресс: {progress if progress else 0}\n\n"
            f"🔮 *Текущая история*:\n{story[:300]}..." if story else ""
        )
    else:
        response = "У тебя пока нет статистики. Пройди квест хотя бы один раз!"
    
    await message.answer(response, parse_mode="Markdown")

@dp.message(lambda message: message.text == "💀 Смертельные сценарии")
async def show_deaths(message: Message):
    user_id = message.from_user.id
    cursor.execute("SELECT deaths FROM user_stats WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    deaths = result[0] if result else 0
    
    death_messages = [
        "Ты ещё не знаешь вкус поражения...",
        "1 смерть - это только начало",
        f"{deaths} раз ты смотрел в бездну...",
        "Мастер смерти! Ты умер {deaths} раз!"
    ]
    
    msg = death_messages[min(deaths, 3)].format(deaths=deaths)
    await message.answer(msg)

@dp.message(lambda message: message.text == "🔄 Начать новую игру")
async def new_game_with_threat(message: Message, state: FSMContext):
    user_id = message.from_user.id
    cursor.execute("DELETE FROM user_progress WHERE user_id = ?", (user_id,))
    cursor.execute("""
        UPDATE user_stats 
        SET last_played = datetime('now'), 
            threat_scheduled = 0
        WHERE user_id = ?
    """, (user_id,))
    conn.commit()
    
    await message.answer(
        "🌌 *Новая игра началась!*\n\n"
        "Но помни - в этом мире ничто не исчезает бесследно...\n"
        "Твои прошлые решения могут повлиять на новую реальность!\n\n"
        "Через 24 часа тебя ждёт сюрприз...",
        reply_markup=start_keyboard,
        parse_mode="Markdown"
    )
    await state.set_state(Reg.start)

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
