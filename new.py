import asyncio
import logging
import sqlite3
import time
import os
import re
import httpx
from datetime import datetime

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandStart
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, Message, ReplyKeyboardRemove
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

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
BOT_TOKEN = '7602719591:AAER_dkEQXD9x0O4RNnya5nzWss3RAnPqGE'
RATE_LIMIT_DELAY = 3.0
MAX_RETRIES = 3

class SafeMistralAI(ChatMistralAI):
    _timestamps = {}
    
    async def safe_invoke(self, messages, retries=MAX_RETRIES):
        model_id = f"{self.model}-{id(self)}"
        if model_id in self._timestamps:
            elapsed = (datetime.now() - self._timestamps[model_id]).total_seconds()
            if elapsed < RATE_LIMIT_DELAY:
                await asyncio.sleep(RATE_LIMIT_DELAY - elapsed)
        
        try:
            self._timestamps[model_id] = datetime.now()
            return self.invoke(messages)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and retries > 0:
                wait_time = 5 * (MAX_RETRIES - retries + 1)
                logging.warning(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                return await self.safe_invoke(messages, retries - 1)
            raise
        except Exception as e:
            logging.error(f"Error in Mistral API: {str(e)}")
            raise

llm = SafeMistralAI(model="mistral-small-latest")

conn = sqlite3.connect("users_data.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_progress (
        user_id INTEGER PRIMARY KEY,
        progress INTEGER DEFAULT 0,
        story TEXT DEFAULT '',
        thread_id TEXT DEFAULT '',
        options TEXT DEFAULT '[]'
    )
""")
conn.commit()

class Reg(StatesGroup):
    start = State()
    dialog = State()
    custom_choice = State()
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

options_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="1-ый вариант")],
        [KeyboardButton(text="2-ой вариант")],
        [KeyboardButton(text="3-ий вариант")],
        [KeyboardButton(text="🎭 Другой вариант")]
    ],
    resize_keyboard=True,
    input_field_placeholder="🎲 Выбери вариант..."
)

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

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def parse_response(text: str) -> tuple[str, list[str]]:
    parts = re.split(r"Какое действие ты сделаешь\??", text, flags=re.IGNORECASE)
    main_text = parts[0].strip()
    options = []
    
    if len(parts) > 1:
        options_block = parts[1]
        option_matches = re.findall(r"\d+\.\s*(.*)", options_block)
        options = [match.strip() for match in option_matches[:3]]
    
    return main_text, options

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
            await message.answer("📜 Сохранённая история загружена. Выберите действие:", reply_markup=options_keyboard)
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
    current_options = data.get("current_options", [])
    
    if message.text in ["1-ый вариант", "2-ой вариант", "3-ий вариант"]:
        option_index = int(message.text[0]) - 1
        if 0 <= option_index < len(current_options):
            user_input = current_options[option_index]
        else:
            await message.answer("❌ Выберите вариант из списка", reply_markup=options_keyboard)
            return
    elif message.text == "🎭 Другой вариант":
        await message.answer("✍️ Напишите свой вариант действия:", reply_markup=ReplyKeyboardRemove())
        await state.set_state(Reg.custom_choice)
        return
    else:
        cursor.execute("SELECT progress FROM user_progress WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row and row[0] == 1:
            user_input = "Начать игру"
        else:
            await message.answer("ℹ️ Используйте кнопки для выбора", reply_markup=options_keyboard)
            return
    
    await process_user_choice(user_input, message, state)

@dp.message(Reg.custom_choice)
async def handle_custom_choice(message: Message, state: FSMContext):
    await process_user_choice(message.text, message, state)
    await state.set_state(Reg.dialog)

async def process_user_choice(user_input: str, message: Message, state: FSMContext):
    user_id = message.from_user.id
    data = await state.get_data()
    
    try:
        retrieved_docs = retriever.invoke(user_input)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        cursor.execute("SELECT progress FROM user_progress WHERE user_id = ?", (user_id,))
        progress = cursor.fetchone()[0] + 1
        
        messages = [
            SystemMessage(content=system_prompt.format(context=context)),
            HumanMessage(content=user_input)
        ] if progress == 1 else [HumanMessage(content=user_input)]
        
        config = {"configurable": {"thread_id": data["thread_id"], "progress": progress}}
        
        await asyncio.sleep(RATE_LIMIT_DELAY)
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: app.invoke({"messages": messages}, config)
        )
        
        bot_response = results["messages"][-1].content
        
        main_text, options = parse_response(bot_response)
        
        await state.update_data(
            story=bot_response,
            current_options=options,
            progress=progress
        )
        
        cursor.execute(
            "UPDATE user_progress SET story = ?, options = ?, progress = ? WHERE user_id = ?",
            (bot_response, str(options), progress, user_id)
        )
        conn.commit()
        
        if progress >= 15:
            await message.answer(main_text, reply_markup=start_keyboard)
            await state.set_state(Reg.start)
        else:
            await message.answer(main_text)
            if options:
                await message.answer("🎲 Выберите действие:", reply_markup=options_keyboard)
            else:
                await message.answer("➡️ Продолжайте...", reply_markup=ReplyKeyboardRemove())
                
    except Exception as e:
        logging.error(f"Error in process_user_choice: {str(e)}")
        await message.answer("⚠️ Произошла ошибка. Пожалуйста, попробуйте позже.", reply_markup=start_keyboard)
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
        await message.answer(f"📜 Ваша сохранённая история:\n{row[2]}\n\nВыберите действие:", reply_markup=options_keyboard)
        await state.set_state(Reg.dialog)

async def main():
    logging.info("Starting bot...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
