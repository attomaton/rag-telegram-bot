"""Общий интерфейс векторного хранилища — чтобы Chroma и pgvector были взаимозаменяемы."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from minirag.catalog import Product


@dataclass
class Filters:
    """Метаданные-фильтры: применяются в БД вместе с векторным поиском
    (pre-filtering), а не после него — иначе из top-k может ничего не остаться."""
    max_price: int | None = None
    min_price: int | None = None
    in_stock_only: bool = False

    def is_empty(self) -> bool:
        return self.max_price is None and self.min_price is None and not self.in_stock_only


@dataclass
class SearchHit:
    id: str
    score: float  # косинусная близость, 1.0 = идентичны
    document: str
    metadata: dict
    rerank_score: float | None = None  # релевантность от cross-encoder, 0..1


class VectorStore(Protocol):
    def upsert(self, products: list[Product], embeddings: np.ndarray) -> None: ...

    def search(self, query_embedding: np.ndarray, k: int, filters: Filters) -> list[SearchHit]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...
