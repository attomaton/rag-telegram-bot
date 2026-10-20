"""Chroma: встроенная векторная БД, хранит данные в папке, сервер не нужен."""
import chromadb
import numpy as np

from minirag.catalog import Product
from minirag.stores.base import Filters, SearchHit


class ChromaStore:
    def __init__(self, path: str, collection: str):
        self.client = chromadb.PersistentClient(path=path)
        self.name = collection
        self.collection = self._get_collection()

    def _get_collection(self):
        # Эмбеддинги считаем сами, поэтому embedding_function не задаём.
        # space=cosine: Chroma вернёт косинусное расстояние (1 - близость).
        # Индекс — HNSW (приближённый поиск ближайших соседей).
        return self.client.get_or_create_collection(
            name=self.name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def upsert(self, products: list[Product], embeddings: np.ndarray) -> None:
        self.collection.upsert(
            ids=[p.id for p in products],
            embeddings=embeddings.tolist(),
            documents=[p.to_document() for p in products],
            metadatas=[p.to_metadata() for p in products],
        )

    def search(self, query_embedding: np.ndarray, k: int, filters: Filters) -> list[SearchHit]:
        res = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=_to_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchHit(id=id_, score=1 - dist, document=doc, metadata=meta)
            for id_, dist, doc, meta in zip(
                res["ids"][0], res["distances"][0], res["documents"][0], res["metadatas"][0]
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.collection = self._get_collection()


def _to_where(filters: Filters) -> dict | None:
    conditions = []
    if filters.max_price is not None:
        conditions.append({"price": {"$lte": filters.max_price}})
    if filters.min_price is not None:
        conditions.append({"price": {"$gte": filters.min_price}})
    if filters.in_stock_only:
        conditions.append({"in_stock": {"$eq": True}})
    if not conditions:
        return None
    # Chroma требует минимум два условия внутри $and.
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}
