# TechJam 2026: Conversational Shopping Agent

A multi-turn conversational shopping agent for an Amazon Clothing, Shoes & Jewelry
catalog (50,000 products). The agent routes user intent into "Buying" (precision,
constraint-filtered search) and "Browsing" (open-ended semantic search) tracks,
tracks conversational state across turns, and asks targeted clarifying questions
when the candidate pool is too broad to confidently recommend a product.

## Overview

Traditional keyword search struggles to distinguish a shopper who knows exactly
what they want from one who's still exploring. This agent addresses that by:

- **Classifying intent per turn** into a Buying track (hard constraints like
  price/color/size → precision SQL + BM25 filtering) or a Browsing track (dense
  semantic embedding search over the full catalog).
- **Tracking conversational state** across up to 10 turns — accumulating slot
  values (category, material, color, size, etc.) turn by turn, and handling
  abrupt intent changes ("Actually, ignore my earlier preference...").
- **Detecting over-generality** — when the candidate pool is still too large,
  the agent asks one targeted clarifying question instead of dumping results,
  prioritizing attributes most likely to narrow the pool based on what's
  actually present in this catalog's data.

## Architecture

```
starter/
├── agent.py           # Entry point (Agent class: reset() / respond()).
│                       # Wires dialog state + retrieval together per the
│                       # evaluator's API contract.
├── dialog_state.py     # SessionState: slot tracking, intent-override
│                       # handling, over-generality detection, and
│                       # clarification-attribute selection.
└── retrieval.py        # DualTrackRouter: intent classification, BM25
                        # keyword search (Buying track), and dense
                        # sentence-embedding search (Browsing track).
```

**Flow per turn:** `user_message` → `SessionState.ingest()` (update slots) →
`DualTrackRouter.classify_intent()` (Buying vs. Browsing) → track-specific
search → `SessionState.should_ask()` (ask a question or return results) →
response.

## Setup & Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/bonusstars/tiktokTechJam26.git
cd tiktokTechJam26

# 2. Install dependencies
pip install -r requirements.txt --break-system-packages

# 3. Unpack the catalog (if not already present in data/)
gzip -dk catalog.jsonl.gz
mkdir -p data
mv catalog.jsonl data/catalog.jsonl
```

> **Note:** the Browsing track uses `sentence-transformers` for dense
> embeddings (model: `all-MiniLM-L6-v2`, downloaded automatically on first
> run). This requires an internet connection the first time the agent is
> initialized.

## Reproducing Results

```bash
python3 -m evaluator.local_evaluator
```

This runs all 200 public dev sessions and writes full per-session results to
`results.json` (gitignored — regenerated locally). A summary prints to stdout:

```json
{
  "hit_rate_at_10": ...,
  "mrr": ...,
  "mttc": ...,
  "recommended_technical_score": ...
}
```

## Results

| Metric | Starter Baseline (weak BM25) | Our Agent |
|---|---|---|
| Hit Rate@10 | 0.125 | **0.65** |
| MRR | 0.068 | **0.309** |
| MTTC | 9.81 | **7.03** |
| Technical Score | ~0.107 | **0.497** |

Per-scenario breakdown (our agent):

| Scenario | Hit Rate@10 | MRR | MTTC |
|---|---|---|---|
| Buying | 0.66 | 0.34 | 6.58 |
| Browsing | 0.68 | 0.28 | 7.06 |
| Intent Override | 0.57 | 0.30 | 7.87 |
| Boundary | 0.60 | 0.29 | 7.80 |

## Limitations & Future Work

- **Personalization (Pillar III) is partially unrealized.** We store each
  user's `preference_tags` and `rating_style` from `user_profile`, but two
  concrete attempts to use them — (1) injecting `preference_tags` directly
  into the search query, and (2) adjusting clarification-question thresholds
  based on `rating_style` — both **measurably regressed** Hit Rate/MRR (0.65 →
  ~0.45) when tested against the evaluator, and were reverted. We believe
  personalization is still valuable here, but naively blending past
  preferences into a BM25 query dilutes precision the same way generic words
  (e.g. "Imported") do. A more promising direction we'd pursue with more time:
  using `preference_tags` as a **reranking signal** on top of retrieved
  candidates, rather than as raw search terms.
- **Intent-override handling is functional but not deeply validated.** Our
  override detection relies on an exact-prefix match specific to how the
  local evaluator's simulated customer phrases a change of mind. This is
  reliable against the public dataset but may not generalize to more varied,
  naturally-phrased override language.
- **Slot extraction is regex-based**, not learned. It performs well against
  this dataset's vocabulary (validated against real catalog term frequencies)
  but is unlikely to generalize to phrasing outside the patterns we tested
  for (colors, materials, sizes, budget, plus a fallback that captures
  unrecognized attribute answers verbatim).
- **No LLM is currently used anywhere in the pipeline** — intent
  classification, slot extraction, and reranking are all rule-based or
  embedding-based. This keeps the system fast, free, and fully reproducible,
  but an LLM-assisted reranking step (using retrieved candidates + accumulated
  slots) is a natural next step for squeezing more precision out of the
  Browsing track in particular.
- Given more time, we would build a proper turn-by-turn regression test suite
  (rather than manual tracing via our debug script) to catch behavior changes
  earlier when either half of the system changes.

## Team Contributions

- **Wenya** — Intent classification (Buying/Browsing routing), dual-track
  retrieval (BM25 keyword search + dense embedding search via
  sentence-transformers), catalog indexing, evaluation infrastructure.
- **Ernest** — Dialog state tracking (slot schema, accumulation, and
  intent-override handling), over-generality detection and clarification
  question generation, personalization experiments (tested and documented,
  see Limitations), debugging tooling (`debug_browsing.py`).

## Acknowledgments

Built on the TechJam 2026 participant kit, using a frozen 50,000-product
subset of the Amazon Reviews 2023 (Clothing, Shoes & Jewelry) dataset.
