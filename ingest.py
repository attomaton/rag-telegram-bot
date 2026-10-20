"""Индексация каталога: CSV → эмбеддинги → векторная БД.

    python ingest.py            # upsert (идемпотентно: повторный запуск обновит товары по id)
    python ingest.py --reset    # пересоздать коллекцию/таблицу с нуля
"""
import argparse
import time

from minirag import config
from minirag.catalog import load_catalog
from minirag.embeddings import Embedder
from minirag.stores import get_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="удалить старые данные перед загрузкой")
    args = parser.parse_args()

    products = load_catalog(config.CATALOG_PATH)
    print(f"Каталог: {len(products)} товаров из {config.CATALOG_PATH}")

    embedder = Embedder(config.EMBEDDING_MODEL)
    started = time.perf_counter()
    embeddings = embedder.embed_documents([p.to_document() for p in products])
    print(f"Эмбеддинги: {config.EMBEDDING_MODEL}, размерность {embeddings.shape[1]}, "
          f"{time.perf_counter() - started:.1f} c")

    store = get_store(embedder.dim)
    if args.reset:
        store.reset()
    store.upsert(products, embeddings)
    print(f"Хранилище: {config.VECTOR_STORE}, записей в коллекции: {store.count()}")


if __name__ == "__main__":
    main()
