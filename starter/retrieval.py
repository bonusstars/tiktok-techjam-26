from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# Regex patterns for deterministic buying signals and constraint extraction
PRICE_RE = re.compile(r"(?:under|below|less than|\$)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
COLOR_RE = re.compile(r"\b(black|white|red|blue|green|yellow|pink|purple|grey|gray|cyan|magenta)\b", re.IGNORECASE)
SIZE_RE = re.compile(r"\b(size\s+[0-9a-z]+|small|medium|large|xl|xxl|\d{1,2}(?:\.\d)?\s*(?:oz|inch|cm|mm|gb|tb|kg))\b", re.IGNORECASE)
PRICE_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")

def _parse_price(value: object) -> float:
    """Safely extracts a numeric float price from dirty strings, floats, ints, or missing values."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Strip commas and find the first valid numeric pattern
        cleaned = value.replace(",", "").strip()
        match = PRICE_NUM_RE.search(cleaned)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 0.0
    return 0.0


def normalize_text(value: object) -> str:
    """Flatten heterogeneous JSON values (dict, list, primitive) into a clean string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def extract_terms(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric keywords, filtering stopwords."""
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class DualTrackRouter:
    """Manages index creation, intent classification, and dual-track search execution."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.encoder = SentenceTransformer(embedding_model_name)

        self.doc_asins: list[str] = []
        self.doc_embeddings: np.ndarray | None = None

        # Pre-computed semantic centroid anchors for hybrid intent classification
        self._intent_anchors = {
            "BUYING": self.encoder.encode([
                "buy exact model",
                "replacement part for brand",
                "cheap price under 20",
                "specific product item code",
            ], normalize_embeddings=True).mean(axis=0),
            "BROWSING": self.encoder.encode([
                "gift ideas for friend",
                "aesthetic vibe for bedroom",
                "things to help relax",
                "what should I get for a trip",
            ], normalize_embeddings=True).mean(axis=0),
        }

        self._build_indices()

    def _build_indices(self) -> None:
        cursor = self.connection.cursor()

        # Structured table for relational attribute filtering
        cursor.execute(
            "CREATE TABLE product_meta ("
            "parent_asin TEXT PRIMARY KEY, title TEXT, price REAL, store TEXT, categories TEXT)"
        )

        # FTS5 full-text index for precision keyword retrieval
        cursor.execute(
            "CREATE VIRTUAL TABLE products_fts USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        fts_batch: list[tuple[str, str, str, str, str, str, str]] = []
        meta_batch: list[tuple[str, str, float, str, str]] = []
        texts_to_embed: list[str] = []

        if not self.catalog_path.exists():
            return

        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product.get("parent_asin", ""))
                title = normalize_text(product.get("title"))
                categories = normalize_text(product.get("categories"))
                features = normalize_text(product.get("features"))
                details = normalize_text(product.get("details"))
                store = normalize_text(product.get("store"))
                description = normalize_text(product.get("description"))
                price = _parse_price(product.get("price"))

                self.doc_asins.append(asin)
                texts_to_embed.append(f"{title}. Category: {categories}. Features: {features} {description}")

                meta_batch.append((asin, title, price, store, categories))
                fts_batch.append((asin, title, categories, features, details, store, description))

                if len(fts_batch) >= 1000:
                    cursor.executemany("INSERT INTO products_fts VALUES (?, ?, ?, ?, ?, ?, ?)", fts_batch)
                    cursor.executemany("INSERT INTO product_meta VALUES (?, ?, ?, ?, ?)", meta_batch)
                    fts_batch.clear()
                    meta_batch.clear()

        if fts_batch:
            cursor.executemany("INSERT INTO products_fts VALUES (?, ?, ?, ?, ?, ?, ?)", fts_batch)
            cursor.executemany("INSERT INTO product_meta VALUES (?, ?, ?, ?, ?)", meta_batch)

        self.connection.commit()

        if texts_to_embed:
            self.doc_embeddings = self.encoder.encode(
                texts_to_embed,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

    def classify_intent(self, query: str) -> tuple[str, dict[str, Any]]:
        """Hybrid classifier: Fast regex hard-rules with semantic vector fallback."""
        constraints: dict[str, Any] = {}

        price_match = PRICE_RE.search(query)
        if price_match:
            constraints["max_price"] = float(price_match.group(1))

        color_match = COLOR_RE.search(query)
        if color_match:
            constraints["color"] = color_match.group(1).lower()

        size_match = SIZE_RE.search(query)
        if size_match:
            constraints["size"] = size_match.group(1).lower()

        # Hard constraints directly flag high-intent buying
        if constraints:
            return "BUYING", constraints

        query_vec = self.encoder.encode(query, normalize_embeddings=True)
        buying_score = float(np.dot(query_vec, self._intent_anchors["BUYING"]))
        browsing_score = float(np.dot(query_vec, self._intent_anchors["BROWSING"]))

        intent = "BUYING" if buying_score > browsing_score else "BROWSING"
        return intent, constraints

    def search_buying_track(self, query: str, constraints: dict[str, Any], top_k: int) -> list[dict]:
        """Track 1: BM25 keyword matching with relational constraint enforcement."""
        terms = list(dict.fromkeys(extract_terms(query)))[:40]
        fts_expression = " AND ".join(f'"{term}"' for term in terms) if terms else ""

        sql_parts = ["SELECT p.parent_asin FROM products_fts p JOIN product_meta m ON p.parent_asin = m.parent_asin"]
        where_clauses: list[str] = []
        params: list[Any] = []

        if fts_expression:
            where_clauses.append("products_fts MATCH ?")
            params.append(fts_expression)

        if "max_price" in constraints:
            where_clauses.append("m.price <= ? AND m.price > 0")
            params.append(constraints["max_price"])

        if where_clauses:
            sql_parts.append("WHERE " + " AND ".join(where_clauses))

        sql_parts.append("ORDER BY bm25(products_fts, 0.0, 8.0, 4.0, 2.0, 2.0, 1.0, 1.0) LIMIT ?")
        params.append(top_k)

        rows = self.connection.execute(" ".join(sql_parts), params).fetchall()

        # Fallback to OR expression if AND conjunction is overly restrictive
        if not rows and terms:
            loose_expression = " OR ".join(f'"{term}"' for term in terms)
            rows = self.connection.execute(
                "SELECT parent_asin FROM products_fts WHERE products_fts MATCH ? "
                "ORDER BY bm25(products_fts, 0.0, 8.0, 4.0, 2.0, 2.0, 1.0, 1.0) LIMIT ?",
                (loose_expression, top_k),
            ).fetchall()

        return [{"parent_asin": str(row[0]), "track": "buying"} for row in rows]

    def search_browsing_track(self, query: str, top_k: int) -> list[dict]:
        """Track 2: Dense cosine similarity vector search for scenario matching."""
        if self.doc_embeddings is None or len(self.doc_asins) == 0:
            return []

        query_vec = self.encoder.encode(query, normalize_embeddings=True)
        similarities = np.dot(self.doc_embeddings, query_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {"parent_asin": self.doc_asins[idx], "track": "browsing", "score": float(similarities[idx])}
            for idx in top_indices
        ]


