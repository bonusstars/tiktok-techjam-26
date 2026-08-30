"""
Debug script: trace through a handful of FAILED browsing sessions turn by turn,
so you can see exactly what the agent extracted, asked, and got back at each step.

Run from your repo root:
    python3 debug_browsing.py
"""
from __future__ import annotations

import json
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
    MAX_TURNS,
    TOP_K,
)
from starter.agent import Agent


def trace_session(agent: Agent, sample: dict, catalog_ids: set, categories: dict, products: dict) -> None:
    print(f"\n{'='*70}")
    print(f"sample_id: {sample['sample_id']}  |  scenario: {sample['scenario_type']}  |  difficulty: {sample.get('difficulty_bucket')}")
    target = str(sample["ground_truth"]["parent_asin"])
    print(f"target parent_asin: {target}")
    print(f"target title: {products.get(target, {}).get('title', '(not found)')}")

    session_id = f"debug_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])

    effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
    print(f"hidden hard_constraints: {effective_intent_card.get('hard_constraints')}")
    print(f"hidden soft_preferences: {effective_intent_card.get('soft_preferences')}")

    disclosed: set = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- turn {turn} ---")
        print(f"  USER: {user_message}")
        response = agent.respond(session_id, user_message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        print(f"  AGENT message: {response.get('message')}")
        print(f"  AGENT ask_attribute: {response.get('ask_attribute')}")
        print(f"  AGENT top recs: {ranked[:5]}{'...' if len(ranked) > 5 else ''}")

        if target in ranked:
            print(f"  >>> HIT at rank {ranked.index(target) + 1} <<<")
            return

        if turn == MAX_TURNS:
            print("  >>> NO HIT — turn limit reached <<<")
            return

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used
            )


def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
    agent = Agent("data/catalog.jsonl")

    # First, find which browsing sessions currently fail, by quickly running
    # them without printing anything.
    browsing = [s for s in samples if s["scenario_type"] == "browsing"]
    failing = []
    for sample in browsing:
        session_id = f"scan_{sample['sample_id']}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
        disclosed: set = set()
        boundary_used = False
        user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
        hit = False
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if target in ranked:
                hit = True
                break
            if turn == MAX_TURNS:
                break
            user_message, boundary_used = customer_reply(
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used
            )
        if not hit:
            failing.append(sample)

    print(f"\n{len(failing)} / {len(browsing)} browsing sessions currently FAIL.")
    print("Tracing the first 5 failures in detail:\n")

    for sample in failing[:5]:
        trace_session(agent, sample, catalog_ids, categories, products)


if __name__ == "__main__":
    main()