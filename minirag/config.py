"""Настройки из переменных окружения (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Кеш моделей HuggingFace — в папке проекта, а не в профиле пользователя на C:
# (реранкер весит ~2,2 ГБ). Должно выполниться до импорта sentence_transformers.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

CATALOG_PATH = Path(os.getenv("CATALOG_PATH", ROOT / "data" / "catalog.csv"))

# Эмбеддинги: локальная мультиязычная модель (русский поддерживается, ключи не нужны).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# Векторное хранилище: "chroma" (встроенная, без сервера) или "pgvector" (Postgres).
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma").lower()
CHROMA_PATH = os.getenv("CHROMA_PATH", str(ROOT / "chroma_db"))
PG_DSN = os.getenv("PG_DSN", "postgresql://rag:rag@localhost:5432/rag")
COLLECTION = os.getenv("COLLECTION", "products")

# Ретрив: сколько товаров достаём и минимальная косинусная близость, ниже которой
# считаем, что в каталоге ничего подходящего нет.
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.81"))

# Реранкер (cross-encoder): пересортировывает RERANK_CANDIDATES кандидатов из векторного
# поиска и оставляет до TOP_K лучших. Пустой RERANKER_MODEL — реранкер выключен,
# тогда работает порог MIN_SCORE по косинусной близости.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
# Если даже лучший кандидат ниже RERANK_MIN_SCORE (0..1) — в каталоге ничего подходящего.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.01"))
# Остальные остаются, если их оценка ≥ RERANK_MIN_SCORE или ≥ RERANK_RELATIVE × оценка лучшего.
RERANK_RELATIVE = float(os.getenv("RERANK_RELATIVE", "0.1"))

# Генерация ответа через Claude. USE_LLM=false — бот просто показывает найденные товары.
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "low")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
