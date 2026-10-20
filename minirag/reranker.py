"""Реранкер (cross-encoder) — второй этап поиска.

Bi-encoder (эмбеддинги) кодирует запрос и документ ПО ОТДЕЛЬНОСТИ, поэтому быстрый:
векторы документов посчитаны заранее, поиск идёт по индексу. Но он сравнивает только
«общий смысл» и путается в деталях.

Cross-encoder читает пару (запрос, документ) ЦЕЛИКОМ, одним проходом трансформера,
и видит, как слова запроса соотносятся со словами документа. Точнее, но ему нужен
отдельный прогон на каждую пару, поэтому по всему каталогу его не запускают:
векторный поиск отбирает ~20 кандидатов, реранкер их пересортировывает.
"""
from dataclasses import replace
from functools import cached_property

import torch
from sentence_transformers import CrossEncoder

from minirag.stores import SearchHit


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def model(self) -> CrossEncoder:
        return CrossEncoder(self.model_name)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        # Sigmoid переводит логит модели в 0..1 — с такой шкалой проще задавать порог.
        scores = self.model.predict(
            [(query, h.document) for h in hits], activation_fn=torch.nn.Sigmoid()
        )
        reranked = [replace(h, rerank_score=float(s)) for h, s in zip(hits, scores)]
        return sorted(reranked, key=lambda h: h.rerank_score, reverse=True)
