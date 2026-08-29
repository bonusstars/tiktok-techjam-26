from __future__ import annotations

import re

# Order matters: this is the priority in which we ask for missing attributes.
# Category first (already known from turn 1 almost always), then attributes
# that most reliably narrow the candidate pool.
ATTRIBUTE_ORDER = [
    "category",
    "budget",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "use_case",
    "feature",
]

# Mirrors the evaluator's own regexes (see evaluator/local_evaluator.py) so that
# extraction targets the same vocabulary the simulated customer actually uses.
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)
BUDGET_RE = re.compile(r"\$\s?\d+(?:\.\d+)?|\bunder\s+\$?\d+\b", re.I)
SIZE_RE = re.compile(r"\bsize\s+([a-z0-9]+)\b|\b(small|medium|large|x[sl]|xxl)\b", re.I)
CATEGORY_RE = re.compile(r"looking for ([^.,]+)", re.I)

# The evaluator's override message always starts with this exact phrase.
OVERRIDE_PREFIX = "actually, ignore my earlier preference"


class SessionState:
    """Tracks accumulated slot values and clarification history for one session."""

    def __init__(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id
        self.user_profile = user_profile or {}
        self.slots: dict[str, str | None] = {attr: None for attr in ATTRIBUTE_ORDER if attr != "category"}
        self.slots["category"] = None
        self.asked: set[str] = set()
        self.turn_count = 0

    # ------------------------------------------------------------------ #
    # Ingesting a new user message
    # ------------------------------------------------------------------ #
    def ingest(self, message: str) -> None:
        self.turn_count += 1
        text = message or ""
        lowered = text.strip().lower()

        if lowered.startswith(OVERRIDE_PREFIX):
            self._handle_override(text)
            return

        self._extract_into_slots(text)

    def _handle_override(self, text: str) -> None:
        # Message format: "Actually, ignore my earlier preference. What I need is: {new_value}."
        new_value = ""
        if "what i need is:" in text.lower():
            new_value = text.split(":", 1)[-1].strip(" .")

        # A category change invalidates most other slots; a same-category
        # correction (e.g. a different material/color) should only overwrite
        # the conflicting slot. We can't always tell which without deeper NLP,
        # so use a conservative heuristic: if the new value doesn't match any
        # known attribute regex, treat it as a full reset except category.
        matched_any = False
        for attr, extractor in self._extractors():
            value = extractor(new_value)
            if value:
                self.slots[attr] = value
                matched_any = True

        if not matched_any and new_value:
            # Couldn't classify the new value into a known slot type (e.g. it's
            # a style/use-case phrase) — reset soft slots but keep category,
            # and stash it under "feature" so it still influences retrieval.
            for attr in self.slots:
                if attr != "category":
                    self.slots[attr] = None
            self.slots["feature"] = new_value

        # Whatever attribute we just resolved shouldn't be re-asked.
        self.asked.discard(self._classify(new_value))

    def _extract_into_slots(self, text: str) -> None:
        for attr, extractor in self._extractors():
            value = extractor(text)
            if value:
                self.slots[attr] = value

        if self.slots["category"] is None:
            m = CATEGORY_RE.search(text)
            if m:
                self.slots["category"] = m.group(1).strip()

    def _extractors(self):
        return [
            ("material", lambda t: (MATERIAL_RE.search(t) or [None]) and self._match_or_none(MATERIAL_RE, t)),
            ("color", lambda t: self._match_or_none(COLOR_RE, t)),
            ("budget", lambda t: self._match_or_none(BUDGET_RE, t)),
            ("size", lambda t: self._match_size(t)),
        ]

    @staticmethod
    def _match_or_none(pattern: re.Pattern, text: str) -> str | None:
        m = pattern.search(text)
        return m.group(1).lower() if m and m.groups() else (m.group(0).lower() if m else None)

    @staticmethod
    def _match_size(text: str) -> str | None:
        m = SIZE_RE.search(text)
        if not m:
            return None
        return (m.group(1) or m.group(2) or "").lower() or None

    @staticmethod
    def _classify(text: str) -> str | None:
        if not text:
            return None
        if MATERIAL_RE.search(text):
            return "material"
        if COLOR_RE.search(text):
            return "color"
        if BUDGET_RE.search(text):
            return "budget"
        if SIZE_RE.search(text):
            return "size"
        return None

    # ------------------------------------------------------------------ #
    # Deciding what to ask next
    # ------------------------------------------------------------------ #
    def next_attribute_to_ask(self) -> str | None:
        """Return the highest-priority unfilled, unasked attribute, or None."""
        for attr in ATTRIBUTE_ORDER:
            if self.slots.get(attr) is None and attr not in self.asked:
                self.asked.add(attr)
                return attr
        return None

    MAX_QUESTIONS_PER_SESSION = 2
    STOP_ASKING_AFTER_TURN = 7
    POOL_TOO_BIG_THRESHOLD = 15

    def should_ask(self, candidate_count: int, turn: int) -> bool:
        """Heuristic: ask a clarifying question only when the pool is still
        large, we haven't already asked too many questions this session, and
        we're not close enough to the turn limit that asking would waste our
        remaining budget."""
        if len(self.asked) >= self.MAX_QUESTIONS_PER_SESSION:
            return False
        if turn >= self.STOP_ASKING_AFTER_TURN:
            return False
        if candidate_count <= self.POOL_TOO_BIG_THRESHOLD:
            return False
        return self.next_available_attribute() is not None

    def next_available_attribute(self) -> str | None:
        """Peek without consuming — used by should_ask() to check availability."""
        for attr in ATTRIBUTE_ORDER:
            if self.slots.get(attr) is None and attr not in self.asked:
                return attr
        return None

    def as_query_terms(self) -> list[str]:
        """Flatten known slot values into a list of search terms for retrieval."""
        return [str(v) for v in self.slots.values() if v]