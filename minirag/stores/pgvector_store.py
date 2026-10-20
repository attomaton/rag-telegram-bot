"""pgvector: векторы как обычная колонка в Postgres.

Плюс подхода — векторный поиск и SQL-фильтры/джойны в одном запросе и одной
транзакции, без отдельной БД для векторов.
"""
import json

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from minirag.catalog import Product
from minirag.stores.base import Filters, SearchHit


class PgVectorStore:
    def __init__(self, dsn: str, table: str, dim: int):
        self.table = table
        self.dim = dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self._create_table()

    def _create_table(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id        text PRIMARY KEY,
                name      text NOT NULL,
                category  text NOT NULL,
                brand     text NOT NULL,
                price     integer NOT NULL,
                in_stock  boolean NOT NULL,
                document  text NOT NULL,
                metadata  jsonb NOT NULL,
                embedding vector({self.dim}) NOT NULL
            )
            """
        )
        # HNSW-индекс по косинусному расстоянию (оператор <=>).
        # Без индекса Postgres делает точный полный перебор — на тысячах строк это ок,
        # на миллионах нужен ANN-индекс (HNSW или IVFFlat).
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_hnsw "
            f"ON {self.table} USING hnsw (embedding vector_cosine_ops)"
        )

    def upsert(self, products: list[Product], embeddings: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table}
                    (id, name, category, brand, price, in_stock, document, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name, category = EXCLUDED.category,
                    brand = EXCLUDED.brand, price = EXCLUDED.price,
                    in_stock = EXCLUDED.in_stock, document = EXCLUDED.document,
                    metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding
                """,
                [
                    (
                        p.id, p.name, p.category, p.brand, p.price, p.in_stock,
                        p.to_document(), json.dumps(p.to_metadata(), ensure_ascii=False), emb,
                    )
                    for p, emb in zip(products, embeddings)
                ],
            )

    def search(self, query_embedding: np.ndarray, k: int, filters: Filters) -> list[SearchHit]:
        where, params = [], []
        if filters.max_price is not None:
            where.append("price <= %s")
            params.append(filters.max_price)
        if filters.min_price is not None:
            where.append("price >= %s")
            params.append(filters.min_price)
        if filters.in_stock_only:
            where.append("in_stock")
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        rows = self.conn.execute(
            f"""
            SELECT id, 1 - (embedding <=> %s) AS score, document, metadata
            FROM {self.table}
            {where_sql}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            [query_embedding, *params, query_embedding, k],
        ).fetchall()
        return [SearchHit(id=r[0], score=float(r[1]), document=r[2], metadata=r[3]) for r in rows]

    def count(self) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def reset(self) -> None:
        self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
        self._create_table()
