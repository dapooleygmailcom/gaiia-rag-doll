import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.retrieval.rules_lawyer import ask_rules_lawyer_game, load_game_profile

def inspect_query(q_id, query):
    print("=" * 80)
    print(f"QUERY {q_id}: {query}")
    print("=" * 80)
    
    answer, context_chunks, debug = ask_rules_lawyer_game(query, profile_path="data/asl_profile.json")
    
    print("\nDEBUG INFO:")
    for k, v in debug.items():
        print(f"  {k}: {v}")
        
    print(f"\nTOP 18 CONTEXT CHUNKS GIVEN TO GENERATOR ({len(context_chunks)} total):")
    for i, (doc, meta) in enumerate(context_chunks[:18]):
        rn = meta.get('rule_number') or meta.get('rule_id', '')
        src = meta.get('source_doc') or meta.get('source_file', '')
        p = meta.get('priority', 9)
        print(f"  [{i+1}] Rule: {rn} | P{p} | Doc: {src}")
        print(f"      Snippet: {doc[:160].strip()}...\n")
        
    print("=" * 40)
    print("GENERATED ANSWER:")
    print(answer)
    print("=" * 80 + "\n")

if __name__ == "__main__":
    inspect_query("34", "Advance phase and foxholes: Can a squad advance into an adjacent hex and into a foxhole in the same move?")
    inspect_query("55", "Close Combat Sequence and Infiltration: So my opponent and I just completed a Close Combat. No ambush was obtained. I had 2:1 odds and rolled a 5, eliminating him, he rolled Snakes. He has the option now of withdrawing from the hex and no elimination occurs to either side, or he can stay and both participants are eliminated, correct?")
