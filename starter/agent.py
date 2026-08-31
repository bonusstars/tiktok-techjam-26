from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from starter.dialog_state import SessionState


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    """Weak baseline retrieval + Person B's dialog-state layer wired in."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._states: dict[str, SessionState] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._states[session_id] = SessionState(session_id, user_profile)

    def _search(self, query_text: str, top_k: int) -> tuple[list[dict], int]:
        """Returns (top_k recommendations, total_match_count).

        total_match_count is the FULL number of matches BEFORE truncation to
        top_k — this is what tells us whether the candidate pool is actually
        large (over-generality) vs. narrow, since `recommendations` itself is
        always capped at top_k (10) and can never reflect pool size on its own.
        """
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return [], 0

        total = self.connection.execute(
            "SELECT COUNT(*) FROM products WHERE products MATCH ?",
            (expression,),
        ).fetchone()[0]

        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, top_k),
        ).fetchall()
        recommendations = [{"parent_asin": str(row[0])} for row in rows]
        return recommendations, total

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._states:
            raise RuntimeError("reset must be called before respond")

        state = self._states[session_id]
        state.ingest(user_message)

        query_terms = state.as_query_terms()
        query_text = " ".join(query_terms) if query_terms else user_message
        recommendations, total_matches = self._search(query_text, top_k)

        # NEW: use total_matches (the real pool size), not len(recommendations)
        # (which is always <= top_k and can never signal "pool is too big").
        ask_attribute = None
        if state.should_ask(total_matches, turn):
            ask_attribute = state.next_attribute_to_ask()

        if ask_attribute:
            message = f"Here are some options so far — could you tell me more about your preferred {ask_attribute}?"
        else:
            message = "Here are the closest matches I found."

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }