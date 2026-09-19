import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.retrieval.rules_lawyer import ask_rules_lawyer_game, load_game_profile
from engine.evaluation.adjudicator import adjudicate_answer

load_game_profile("data/asl_profile.json")

def retest(tc_num, tc_id, query, ground_truth, expected_rules):
    print("=" * 80)
    print(f"RETESTING TC #{tc_num} (ID: {tc_id})")
    print(f"Query: {query}")
    print(f"Expected Rules: {expected_rules}")
    print("=" * 80)

    answer, context_chunks, debug = ask_rules_lawyer_game(query)

    print("\n--- GENERATED ANSWER ---")
    print(answer)

    print("\n--- RUNNING ADJUDICATION ---")
    adj = adjudicate_answer(
        query=query,
        ground_truth=ground_truth,
        candidate_answer=answer,
        judge_model="llama3.1:8b"
    )

    print(f"\nRuling: {adj.get('ruling')} | Substantive Accuracy: {adj.get('substantive_accuracy')}")
    print(f"Rationale: {adj.get('rationale')}")
    print("=" * 80 + "\n")
    return {
        "id": tc_id,
        "tc_num": tc_num,
        "query": query,
        "answer": answer,
        "adjudication": adj
    }

if __name__ == "__main__":
    gt34 = (
        "Can a squad advance into an adjacent hex and into a foxhole in the same move?\n"
        "As previous, yes. Outside and inside the foxhole are the same Location for movement purposes [B27.13], "
        "so you aren't moving more than one Location. You can even move from IN a foxhole out, into the next hex "
        "then IN a foxhole in that hex. A downside is that if the cost to move between hexes is two or greater "
        "(e.g. brush, woods) or if the unit possesses more than its IPC it may be an Advance vs. Difficult Terrain [A4.72]."
    )
    res34 = retest(
        34,
        "bgg_asl_3682996",
        "Advance phase and foxholes: Can a squad advance into an adjacent hex and into a foxhole in the same move?",
        gt34,
        ["A4.7", "B27.13"]
    )

    gt55 = (
        "So my opponent and I just completed a Close Combat. No ambush was obtained. I had 2:1 odds and rolled a 5, "
        "eliminating him, he rolled Snakes. He has the option now of withdrawing from the hex and no elimination occurs "
        "to either side, or he can stay and both participants are eliminated, correct?\n"
        "A11.22, \"Provided it has not already been eliminated/captured/pinned, any Infantry/Cavalry unit which rolls "
        "an Original 2 CC DR may withdraw from CC/Melee immediately thereafter in the same CCPh without being attacked, "
        "even if it did not eliminate the defenders.\"\n"
        "In your case the unit was already eliminated, and that result stands. This is one reason why it is very important "
        "to follow the mechanics of A11.12 exactly and not just start throwing dice in the CCPh willy-nilly."
    )
    res55 = retest(
        55,
        "bgg_asl_3657611",
        "Close Combat Sequence and Infiltration: So my opponent and I just completed a Close Combat. No ambush was obtained. I had 2:1 odds and rolled a 5, eliminating him, he rolled Snakes. He has the option now of withdrawing from the hex and no elimination occurs to either side, or he can stay and both participants are eliminated, correct?",
        gt55,
        ["A11.12", "A11.22"]
    )
