from minirag import config
from minirag.stores.base import Filters, SearchHit, VectorStore


def get_store(dim: int) -> VectorStore:
    if config.VECTOR_STORE == "chroma":
        from minirag.stores.chroma_store import ChromaStore
        return ChromaStore(config.CHROMA_PATH, config.COLLECTION)
    if config.VECTOR_STORE == "pgvector":
        from minirag.stores.pgvector_store import PgVectorStore
        return PgVectorStore(config.PG_DSN, config.COLLECTION, dim)
    raise ValueError(f"Неизвестный VECTOR_STORE={config.VECTOR_STORE!r}: ожидается chroma или pgvector")


__all__ = ["Filters", "SearchHit", "VectorStore", "get_store"]
