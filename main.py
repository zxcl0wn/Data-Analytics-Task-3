import os
import asyncio
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(
        "Отправь текст или CSV/Excel файл — я сделаю анализ 📊"
    )


@dp.message(lambda message: message.text)
async def handle_text(message: Message):
    await message.answer("Обрабатываю текст...")

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=f"Сделай краткое саммари и выдели ключевые идеи: {message.text}"
    )

    await message.answer(response.output_text)


@dp.message(lambda message: message.document is not None)
async def handle_file(message: Message):
    await message.answer("Файл получен, обрабатываю...")

    file = await bot.get_file(message.document.file_id)
    file_path = file.file_path

    downloaded_file = await bot.download_file(file_path)
    filename = message.document.file_name

    with open(filename, "wb") as f:
        f.write(downloaded_file.read())

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(filename)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(filename)
        else:
            await message.answer("Поддерживаются только CSV и Excel файлы")
            return
    except Exception as e:
        await message.answer(f"Ошибка чтения файла: {e}")
        return

    if df.empty:
        await message.answer("Файл пустой")
        return

    sample = df.head(5).to_string()

    await message.answer(f"Пример данных:\n{sample}")

    prompt = f"""
        Проанализируй таблицу и дай:
        1. Краткое описание
        2. Основные зависимости
        3. Интересные наблюдения
        
        Данные:
        {sample}
    """

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )

    await message.answer(response.output_text)


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
