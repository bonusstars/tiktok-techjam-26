from __future__ import annotations

from pathlib import Path
from .retrieval import DualTrackRouter


class Agent:
    """Dual-track agent routing queries into precision buying or exploratory browsing."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.router = DualTrackRouter(catalog_path=catalog_path)
        self._sessions: set[str] = set()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int = 1,
        top_k: int = 10,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        # Step 1: Detect intent and extract hard filters
        intent, constraints = self.router.classify_intent(user_message)

        # Step 2: Route dynamically based on classified intent
        if intent == "BUYING":
            recommendations = self.router.search_buying_track(
                query=user_message,
                constraints=constraints,
                top_k=top_k,
            )
            message = f"Applied precise filters for your search (Constraints: {constraints or 'keyword match'})."
        else:
            recommendations = self.router.search_browsing_track(
                query=user_message,
                top_k=top_k,
            )
            message = "Surfacing scenario and vibe-aligned recommendations."

        return {
            "message": message,
            "detected_intent": intent,
            "extracted_constraints": constraints,
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
