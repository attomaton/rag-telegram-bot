"""Эмбеддинги через sentence-transformers.

Модели семейства E5 обучены с префиксами: документы кодируются как "passage: ...",
запросы — как "query: ...". Без префиксов качество поиска заметно падает.
Векторы нормализуются, поэтому косинусная близость = скалярное произведение.
"""
from functools import cached_property

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._is_e5 = "e5" in model_name.lower()

    @cached_property
    def model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    @property
    def dim(self) -> int:
        # В sentence-transformers 6 метод переименован, старое имя объявлено устаревшим.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return get_dim()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32)

    def embed_query(self, text: str) -> np.ndarray:
        if self._is_e5:
            text = f"query: {text}"
        return self.model.encode(text, normalize_embeddings=True)
