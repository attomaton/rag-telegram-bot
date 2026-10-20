"""RAG-пайплайн: вопрос → фильтры + эмбеддинг → поиск в векторной БД (top-20)
→ реранкер (top-K) → ответ Claude по найденному."""
import asyncio
import logging
import re
from dataclasses import dataclass

import anthropic

from minirag import config
from minirag.embeddings import Embedder
from minirag.reranker import Reranker
from minirag.stores import Filters, SearchHit, get_store

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — консультант интернет-магазина электроники и товаров для активного отдыха.
Отвечай на вопрос покупателя, опираясь ТОЛЬКО на товары из блока <catalog>.

Правила:
- Не придумывай товары, цены и характеристики, которых нет в каталоге.
- Если подходящего товара нет, честно скажи об этом и предложи ближайшую альтернативу из каталога, если она есть.
- Называй товар точным названием и ценой в рублях; если товара нет в наличии — предупреди.
- Пиши коротко и по делу, на русском, простым текстом без Markdown: ответ уйдёт в Telegram как обычный текст."""

NOTHING_FOUND = (
    "Не нашёл в каталоге ничего подходящего 🤷\n"
    "Попробуйте описать товар иначе: для чего он нужен, бюджет, важные характеристики."
)

_NUM = r"(\d[\d\s]*(?:[.,]\d+)?)\s*(к|k|тыс\w*\.?|т\.?р\.?|₽|р\.?|руб\w*\.?)?"
_MAX_RE = re.compile(rf"(?:до|дешевле|не\s+дороже|меньше|максимум|бюджет)\s*{_NUM}", re.I)
_MIN_RE = re.compile(rf"(?:от|дороже|больше|не\s+дешевле)\s*{_NUM}", re.I)
_IN_STOCK_RE = re.compile(r"в\s+наличии|есть\s+в\s+продаже|можно\s+купить\s+сейчас", re.I)


def _parse_amount(number: str, suffix: str | None) -> int | None:
    value = float(number.replace(" ", "").replace(",", "."))
    suffix = (suffix or "").lower()
    if suffix.startswith(("к", "k", "тыс", "т")):
        value *= 1000
    # "до 14 часов" — не цена: без валюты/множителя считаем ценой только суммы от 500.
    if not suffix and value < 500:
        return None
    return int(value)


def extract_filters(question: str) -> Filters:
    """Структурные условия («до 10 тысяч», «в наличии») надёжнее извлечь правилами и
    применить как фильтр в БД: эмбеддинги плохо понимают числа и сравнения."""
    filters = Filters(in_stock_only=bool(_IN_STOCK_RE.search(question)))
    if m := _MAX_RE.search(question):
        filters.max_price = _parse_amount(*m.groups())
    if m := _MIN_RE.search(question):
        filters.min_price = _parse_amount(*m.groups())
    return filters


@dataclass
class Answer:
    text: str
    hits: list[SearchHit]
    filters: Filters


def format_hit(hit: SearchHit) -> str:
    m = hit.metadata
    stock = "в наличии" if m["in_stock"] else "нет в наличии"
    return f"{m['name']} — {m['price']:,} ₽ ({stock})".replace(",", " ")


def format_scores(hit: SearchHit) -> str:
    vec = f"vec {hit.score:.3f}"
    return vec if hit.rerank_score is None else f"rr {hit.rerank_score:.3f} | {vec}"


def threshold_label() -> str:
    if config.RERANKER_MODEL:
        return (f"вектор top-{config.RERANK_CANDIDATES} → реранкер top-{config.TOP_K}, "
                f"лучший rr ≥ {config.RERANK_MIN_SCORE}, остальные ≥ {config.RERANK_MIN_SCORE} "
                f"или ≥ {config.RERANK_RELATIVE}×лучший")
    return f"вектор top-{config.TOP_K}, vec ≥ {config.MIN_SCORE}"


class RAG:
    def __init__(self):
        self.embedder = Embedder(config.EMBEDDING_MODEL)
        self.store = get_store(self.embedder.dim)
        self.reranker = Reranker(config.RERANKER_MODEL) if config.RERANKER_MODEL else None
        self.client = anthropic.AsyncAnthropic() if config.USE_LLM else None

    def retrieve(self, question: str) -> tuple[list[SearchHit], Filters]:
        filters = extract_filters(question)
        query_emb = self.embedder.embed_query(question)
        if self.reranker is None:
            hits = self.store.search(query_emb, k=config.TOP_K, filters=filters)
            relevant = [h for h in hits if h.score >= config.MIN_SCORE]
        else:
            # Двухэтапный поиск: векторный индекс быстро даёт широкий список кандидатов
            # (recall), cross-encoder точно сортирует их и отсекает лишнее (precision).
            candidates = self.store.search(query_emb, k=config.RERANK_CANDIDATES, filters=filters)
            hits = self.reranker.rerank(question, candidates)[:config.TOP_K]
            # Уверенность реранкера сильно зависит от запроса: на «монитор для фото» лучший
            # товар получает 0.97, на «подарок любителю кофе» — 0.01. Поэтому:
            # если даже лучший ниже абсолютного порога — по теме ничего нет;
            # иначе товар остаётся, если прошёл абсолютный порог ИЛИ близок к лучшему.
            if hits and hits[0].rerank_score >= config.RERANK_MIN_SCORE:
                cutoff = min(config.RERANK_MIN_SCORE, hits[0].rerank_score * config.RERANK_RELATIVE)
                relevant = [h for h in hits if h.rerank_score >= cutoff]
            else:
                relevant = []
        log.info("q=%r filters=%s hits=%s", question, filters,
                 [(h.id, format_scores(h)) for h in hits])
        return relevant, filters

    async def answer(self, question: str) -> Answer:
        # Эмбеддинг и поиск — синхронный CPU/IO-код, уводим из event loop бота.
        hits, filters = await asyncio.to_thread(self.retrieve, question)
        if not hits:
            return Answer(NOTHING_FOUND, hits, filters)
        if self.client is None:
            return Answer(self._list_answer(hits), hits, filters)
        try:
            text = await self._generate(question, hits)
        # TypeError SDK бросает, когда не найден ни ключ, ни профиль `ant auth login`.
        except (anthropic.APIError, TypeError) as e:
            log.warning("Claude недоступен (%s), отвечаю списком товаров", e)
            text = self._list_answer(hits)
        return Answer(text, hits, filters)

    async def _generate(self, question: str, hits: list[SearchHit]) -> str:
        context = "\n".join(
            f'<product id="{h.id}" price_rub="{h.metadata["price"]}" '
            f'in_stock="{str(h.metadata["in_stock"]).lower()}">{h.document}</product>'
            for h in hits
        )
        response = await self.client.beta.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": config.CLAUDE_EFFORT},
            # Серверный фолбэк: если модель отклонит запрос, API сам повторит его
            # на подходящей модели в рамках того же вызова.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{
                "role": "user",
                "content": f"<catalog>\n{context}\n</catalog>\n\nВопрос покупателя: {question}",
            }],
        )
        if response.stop_reason == "refusal":
            return self._list_answer(hits)
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text or self._list_answer(hits)

    @staticmethod
    def _list_answer(hits: list[SearchHit]) -> str:
        return "Вот что нашлось в каталоге:\n" + "\n".join(f"• {format_hit(h)}" for h in hits)
