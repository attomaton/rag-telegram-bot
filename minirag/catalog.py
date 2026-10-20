"""Загрузка каталога и превращение товара в документ для индексации."""
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Product:
    id: str
    name: str
    category: str
    brand: str
    price: int
    in_stock: bool
    description: str

    def to_document(self) -> str:
        """Текст, который эмбеддится. Один товар = один чанк: описания короткие,
        дробить их не нужно. Для длинных описаний (инструкции, отзывы) понадобился бы
        сплиттер на куски по ~300–500 токенов с перекрытием."""
        return (
            f"{self.name}. Категория: {self.category}. Бренд: {self.brand}. "
            f"{self.description}"
        )

    def to_metadata(self) -> dict:
        """Структурированные поля — по ним фильтруем, а не ищем семантически."""
        return {
            "name": self.name,
            "category": self.category,
            "brand": self.brand,
            "price": self.price,
            "in_stock": self.in_stock,
        }


def load_catalog(path: Path) -> list[Product]:
    with open(path, encoding="utf-8", newline="") as f:
        return [
            Product(
                id=row["id"],
                name=row["name"],
                category=row["category"],
                brand=row["brand"],
                price=int(row["price"]),
                in_stock=row["in_stock"].strip().lower() == "true",
                description=row["description"],
            )
            for row in csv.DictReader(f)
        ]
