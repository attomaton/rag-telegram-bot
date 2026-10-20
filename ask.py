"""Задать вопрос из терминала, без Telegram — удобно для отладки ретрива.

    python ask.py "нужны наушники для бега до 10 тысяч"
    python ask.py --no-llm "робот-пылесос для квартиры с котом"
    python ask.py --no-llm --no-rerank "…"   # сравнить с чистым векторным поиском
"""
import argparse
import asyncio
import sys

from minirag import config


async def run(question: str) -> None:
    from minirag.rag import RAG, format_hit, format_scores, threshold_label

    rag = RAG()
    answer = await rag.answer(question)
    print(f"Фильтры: {answer.filters}")
    print(f"Найдено ({threshold_label()}):")
    for h in answer.hits:
        print(f"  {format_scores(h)}  {format_hit(h)}")
    print("\nОтвет:\n" + answer.text)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--no-llm", action="store_true", help="только поиск, без Claude")
    parser.add_argument("--no-rerank", action="store_true", help="без реранкера, только векторный поиск")
    args = parser.parse_args()
    if args.no_llm:
        config.USE_LLM = False
    if args.no_rerank:
        config.RERANKER_MODEL = ""
    asyncio.run(run(args.question))


if __name__ == "__main__":
    main()
