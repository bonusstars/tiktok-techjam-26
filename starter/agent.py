from __future__ import annotations

import sys
from pathlib import Path

# Ensure absolute and relative imports work consistently regardless of invocation root
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from retrieval import DualTrackRouter
from starter.dialog_state import SessionState


class Agent:
    """Dual-Track Retrieval Search Engine with Multi-Turn Dialog-State Management."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.router = DualTrackRouter(catalog_path=catalog_path)
        self._states: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._states[session_id] = SessionState(session_id, user_profile)

    def _search(
        self,
        query_text: str,
        intent: str,
        constraints: dict,
        top_k: int,
    ) -> tuple[list[dict], int]:
        """Routes search into Buying vs. Browsing tracks and computes pool matches.

        Returns:
            (top_k recommendations, total_match_count)
        """
        if intent == "BUYING":
            # Track 1: Precision SQL + BM25 filtering
            recommendations = self.router.search_buying_track(
                query=query_text,
                constraints=constraints,
                top_k=top_k,
            )
            # Pool size for buying track: Count matching FTS items with constraints
            terms = list(dict.fromkeys(self.router.extract_terms(query_text) if hasattr(self.router, "extract_terms") else query_text.split()))[:40]
            if terms:
                fts_expr = " OR ".join(f'"{t}"' for t in terms)
                total_matches = self.router.connection.execute(
                    "SELECT COUNT(*) FROM products_fts WHERE products_fts MATCH ?",
                    (fts_expr,),
                ).fetchone()[0]
            else:
                total_matches = len(recommendations)
        else:
            # Track 2: Open-ended dense semantic retrieval
            recommendations = self.router.search_browsing_track(
                query=query_text,
                top_k=top_k,
            )
            # For dense embeddings, candidate pool is the entire embedding space
            total_matches = len(self.router.doc_asins) if self.router.doc_asins else len(recommendations)

        return recommendations, total_matches

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int = 1,
        top_k: int = 10,
    ) -> dict:
        if session_id not in self._states:
            raise RuntimeError("reset must be called before respond")

        # 1. Dialog state ingestion
        state = self._states[session_id]
        state.ingest(user_message)

        # 2. Dual-Track Intent & Hard-Constraint Classification
        intent, constraints = self.router.classify_intent(user_message)

        # 3. Query term accumulation from dialog state
        query_terms = state.as_query_terms()
        query_text = " ".join(query_terms) if query_terms else user_message

        # 4. Search execution across the selected track
        recommendations, total_matches = self._search(
            query_text=query_text,
            intent=intent,
            constraints=constraints,
            top_k=top_k,
        )

        # 5. Clarification attribute logic using total_matches
        ask_attribute = None
        if state.should_ask(total_matches, turn):
            ask_attribute = state.next_attribute_to_ask()

        # 6. Response generation
        if ask_attribute:
            message = f"Here are some options so far — could you tell me more about your preferred {ask_attribute}?"
        elif intent == "BUYING":
            message = f"Applied precise filters for your search (Constraints: {constraints or 'keyword match'})."
        else:
            message = "Surfacing scenario and vibe-aligned recommendations."

        return {
            "message": message,
            "detected_intent": intent,
            "extracted_constraints": constraints,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
