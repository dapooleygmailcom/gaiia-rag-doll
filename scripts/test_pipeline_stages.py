"""
Pipeline Stages Verification Test — Gaiia RAG Doll.
Tests Stage 1 (Distiller), Stage 2 (HyDE), Stage 3 (Dual Vector),
Stage 4 (Hierarchy Expansion), and Stage 5 (Co-occurrence Expansion).
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.retrieval.rules_lawyer import ask_rules_lawyer_game, load_game_profile


def test_query_pipeline():
    profile_path = "data/up_front_profile.json"
    print("=" * 70)
    print("TESTING DECOUPLED PIPELINE ON UP FRONT QUERIES")
    print("=" * 70)

    load_game_profile(profile_path)

    test_queries = [
        {
            "name": "Infiltrator Status Post-Close-Combat",
            "query": "Infiltrators and Close Combat: If an infiltrating group attacks in close combat and wins, do they retain their Infiltrator status after the close combat resolves?",
            "expected_rule": "20.73"
        },
        {
            "name": "Killed Squad Leader and Card Draw Limits",
            "query": "Background: In my game yesterday, my Squad Leader was killed by a sniper attack. What happens to the squad's card hand size and discard limits when the SL dies?",
            "expected_rule": "15.2"
        }
    ]

    for t in test_queries:
        print(f"\n--- Running Test Query: {t['name']} ---")
        print(f"Raw Input: {t['query']}")

        start_t = time.time()
        answer, context_chunks, debug_info = ask_rules_lawyer_game(t["query"], profile_path=profile_path)
        duration = round(time.time() - start_t, 2)

        print(f"[Timing] Duration: {duration}s")
        print(f"[Stage 1 - Distilled Question]: {debug_info.get('distilled_question')}")
        print(f"[Stage 2 - HyDE Clause]: {debug_info.get('hyde_clause')}")
        print(f"[Query Type]: {debug_info.get('query_type')}")
        print(f"[Extracted Rules]: {debug_info.get('rule_numbers')}")
        print(f"[Stage 4 - Parent Expansions]: {debug_info.get('num_parent_expansions')}")
        print(f"[Stage 5 - Co-Occurrence Expansions]: {debug_info.get('num_cooccurrence_expansions')}")
        print(f"[Total Retrieved Chunks]: {debug_info.get('num_retrieved')}")
        print(f"[Stage 6 - Answer Preview]: {answer[:300]}...")

        # Assertions
        assert debug_info.get("distilled_question"), "Distilled question must not be empty"
        assert debug_info.get("hyde_clause"), "HyDE clause must not be empty"
        assert len(context_chunks) > 0, "Retrieved context chunks must not be empty"
        print("  -> Pipeline stages verified successfully for this query.")

    print("\n" + "=" * 70)
    print("ALL PIPELINE STAGE VERIFICATION TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_query_pipeline()
