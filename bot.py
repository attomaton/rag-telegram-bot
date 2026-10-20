"""Telegram-бот, который отвечает на вопросы по каталогу через RAG.

    python bot.py
"""
import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from minirag import config
from minirag.rag import RAG, format_hit, format_scores, threshold_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

dp = Dispatcher()
rag: RAG
debug_chats: set[int] = set()

HELP = (
    "Я консультант магазина — ищу товары в каталоге и отвечаю по нему.\n\n"
    "Примеры:\n"
    "• наушники для бега до 10 тысяч\n"
    "• чем убирать шерсть кота?\n"
    "• ноутбук для монтажа видео в наличии\n"
    "• что подарить любителю кофе до 10к\n\n"
    "/debug — показывать найденные товары и их близость к запросу"
)


@dp.message(CommandStart())
@dp.message(Command("help"))
async def cmd_start(message: Message) -> None:
    await message.answer(HELP)


@dp.message(Command("debug"))
async def cmd_debug(message: Message) -> None:
    chat_id = message.chat.id
    if chat_id in debug_chats:
        debug_chats.discard(chat_id)
        await message.answer("Отладка выключена.")
    else:
        debug_chats.add(chat_id)
        await message.answer("Отладка включена: покажу, что достал из векторной БД.")


@dp.message(F.text)
async def on_question(message: Message) -> None:
    await message.bot.send_chat_action(message.chat.id, "typing")
    answer = await rag.answer(message.text)
    await message.answer(answer.text)

    if message.chat.id in debug_chats:
        lines = [f"🔎 {config.VECTOR_STORE}: {threshold_label()}", f"Фильтры: {answer.filters}"]
        lines += [f"{format_scores(h)}  {format_hit(h)}" for h in answer.hits] or ["(ничего выше порога)"]
        await message.answer("\n".join(lines))


async def main() -> None:
    global rag
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("Не задан TELEGRAM_BOT_TOKEN — создайте бота у @BotFather и пропишите токен в .env")
    rag = RAG()  # здесь же загружается модель эмбеддингов, чтобы первый ответ не тормозил
    if rag.reranker:
        rag.reranker.model  # и реранкер тоже
    logging.info("Векторная БД %s: %d товаров", config.VECTOR_STORE, rag.store.count())
    await dp.start_polling(Bot(config.TELEGRAM_BOT_TOKEN))


if __name__ == "__main__":
    asyncio.run(main())
