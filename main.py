import os
import asyncio
import traceback
import io
import sys
import base64
import ast
import json
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message, BufferedInputFile
from dotenv import load_dotenv
matplotlib.use("Agg")
load_dotenv()


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

groq = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "llama-3.3-70b-versatile"


def ensure_print(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code

    lines = code.splitlines()
    last = tree.body[-1]
    expr_src = "\n".join(lines[last.lineno - 1:last.end_lineno]).strip()

    return "\n".join(lines[:last.lineno - 1] + [f"print({expr_src})"])


def run_python(code: str, df: pd.DataFrame) -> dict:
    code = ensure_print(code)
    namespace = {"df": df, "pd": pd, "plt": plt, "io": io}
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    images = []
    error = None

    try:
        exec(code, namespace)  # noqa: S102
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    if error is None:
        try:
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                try:
                    plt.tight_layout()
                except Exception:
                    pass
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", dpi=120)
                img_buf.seek(0)
                images.append(base64.b64encode(img_buf.read()).decode())
        except Exception as e:
            error = f"Figure capture error: {e}"
        finally:
            plt.close("all")

    return {"stdout": buf.getvalue(), "error": error, "images_base64": images}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code against the dataset. "
                "The dataframe is pre-loaded as `df`. "
                "Always use print() to output results — bare expressions produce no output. "
                "matplotlib is available as `plt`, figures are captured automatically. "
                "On error: call this tool again with fixed code. Never write code in text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code. Use print() for every value you want to see.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

SYSTEM = """Ты — агент анализа данных. Датасет доступен как pandas DataFrame `df`.

СТРОГИЕ ПРАВИЛА:
1. Весь код выполняй ТОЛЬКО через инструмент run_python. Никогда не пиши код в тексте.
2. Всегда используй print() для вывода:
   ПРАВИЛЬНО: print(df['col'].value_counts())
   НЕПРАВИЛЬНО: df['col'].value_counts()
3. Если вернулась ошибка — немедленно вызови инструмент снова с исправленным кодом.
4. Если stdout пустой — ты забыл print(). Повтори с print().
5. Финальный ответ — только после завершения всего анализа, на русском, с конкретными числами."""


def analyse_sync(df: pd.DataFrame) -> tuple[str, list[bytes]]:
    info = (
        f"Shape: {df.shape}\n"
        f"Columns: {list(df.columns)}\n"
        f"Dtypes:\n{df.dtypes.to_string()}\n"
        f"First 3 rows:\n{df.head(3).to_string()}"
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Датасет загружен.\n\n{info}\n\n"
                "Выполни EDA через run_python:\n"
                "1. print(df.columns.tolist()) и print(df.dtypes)\n"
                "2. print(df.describe(include='all'))\n"
                "3. print(df.isnull().sum())\n"
                "4. Распределения / корреляции (с print)\n"
                "5. 1-2 графика через plt\n"
                "6. Финальный текст с инсайтами на русском"
            ),
        },
    ]

    collected_images: list[bytes] = []
    final_text = ""
    max_iterations = 20

    for _ in range(max_iterations):
        response = groq.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=4096,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # accumulate text
        if msg.content:
            final_text += msg.content

        if finish_reason == "stop" or not msg.tool_calls:
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            if tc.function.name != "run_python":
                continue

            try:
                args = json.loads(tc.function.arguments)
                code = args.get("code", "")
            except Exception:
                code = tc.function.arguments  # fallback

            result = run_python(code, df)

            for b64 in result["images_base64"]:
                collected_images.append(base64.b64decode(b64))

            parts = []
            if result["stdout"]:
                parts.append(f"STDOUT:\n{result['stdout'].strip()}")
            else:
                parts.append("STDOUT: (пусто — забыл print(). Повтори с print().)")
            if result["error"]:
                parts.append(f"ERROR (исправь и вызови run_python снова):\n{result['error'].strip()}")
            if result["images_base64"]:
                parts.append(f"[{len(result['images_base64'])} график(а) сохранено]")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "\n".join(parts),
            })

    return final_text.strip(), collected_images


async def analyse_with_agent(df: pd.DataFrame) -> tuple[str, list[bytes]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, analyse_sync, df)


@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(
        "Привет, отправь CSV или Excel файл — агент проведёт EDA:\n"
        "• статистики и метрики\n"
        "• пропуски и зависимости\n"
        "• графики\n"
        "• инсайты и выводы"
    )


@dp.message(lambda m: m.document is not None)
async def handle_file(message: Message):
    filename = message.document.file_name or ""
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        await message.answer("⚠️ Поддерживаются только .csv и .xlsx файлы.")
        return

    status = await message.answer("📥 Файл получен, загружаю...")

    file = await bot.get_file(message.document.file_id)
    downloaded = await bot.download_file(file.file_path)
    raw_bytes = downloaded.read()

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes)) if filename.endswith(".csv") else pd.read_excel(io.BytesIO(raw_bytes))
    except Exception as e:
        await status.edit_text(f"❌ Ошибка чтения файла: {e}")
        return

    if df.empty:
        await status.edit_text("❌ Файл пустой.")
        return

    await status.edit_text(
        f"✅ Датасет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов.\n"
        "🤖 Запускаю анализ..."
    )

    try:
        summary, images = await analyse_with_agent(df)
    except Exception as e:
        await message.answer(f"❌ Ошибка анализа: {e}")
        return

    if summary:
        for chunk in _split(summary, 4000):
            await message.answer(chunk)

    for i, img_bytes in enumerate(images, 1):
        await message.answer_photo(
            BufferedInputFile(img_bytes, filename=f"chart_{i}.png"),
            caption=f"График {i}"
        )

    if not summary and not images:
        await message.answer("⚠️ Агент не вернул результатов.")


@dp.message(lambda m: m.text and not m.text.startswith("/"))
async def handle_text(message: Message):
    await message.answer("Пожалуйста, отправь CSV или Excel файл для анализа")


def _split(text: str, limit: int) -> list[str]:
    chunks = []
    while len(text) > limit:
        chunks.append(text[:limit])
        text = text[limit:]
    if text:
        chunks.append(text)
    return chunks


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
