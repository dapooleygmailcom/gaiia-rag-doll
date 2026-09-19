import os
import sys
import time
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import engine.retrieval.rules_lawyer as rules_lawyer

def run_test():
    print("Loading ASL Profile (data/asl_profile.json)...")
    rules_lawyer.load_game_profile("data/asl_profile.json")

    query = (
        "Hi,\n"
        "when resolving Defensive First Fire against a moving unit (unit A) that spends 2 MF to enter a hex in LOS of 2 enemy units (B and C), what is the correct resolution?\n"
        "- Unit B uses First Fire, then Unit C uses First Fire, then both Unit can use Subsequent First Fire to fire on the 2nd MF spent\n"
        "- Unit B uses First Fire and then its Subsequent First Fire, then Unit C uses First Fire and then its Subsequent First Fire\n"
        "- a mix of the 2"
    )

    print("=" * 70)
    print("RUNNING ASL BGG TEST CASE: Thread 3759227")
    print(f"URL: https://boardgamegeek.com/thread/3759227/first-fire-and-subsequent-first-fire-sequence")
    print("=" * 70)
    print(f"QUERY:\n{query}\n")
    print("=" * 70)

    start_t = time.time()
    ans, ctx, dbg = rules_lawyer.ask_rules_lawyer_game(query)
    elapsed = round(time.time() - start_t, 2)

    print(f"\nAGENT ANSWER ({elapsed}s):")
    print("-" * 70)
    print(ans)
    print("-" * 70)

    print("\nDEBUG INFO:")
    print(f"Query Type: {dbg.get('query_type')}")
    print(f"Retrieved Chunks: {dbg.get('num_retrieved')}")
    print(f"Cross Refs: {dbg.get('num_cross_refs')}")
    print(f"Extracted Rules: {dbg.get('extracted_rules')}")

    print("\nTOP RETRIEVED SOURCES:")
    for idx, (doc, meta) in enumerate(ctx[:10], 1):
        rn = meta.get("rule_number") or meta.get("rule_id", "N/A")
        src = meta.get("source_file", "unknown")
        p = meta.get("priority", "N/A")
        header = doc.split("\n")[0] if isinstance(doc, str) else ""
        print(f"[{idx}] Rule: {rn} | Source: {src} | Priority: {p} | Header: {header[:60]}")

if __name__ == "__main__":
    run_test()
