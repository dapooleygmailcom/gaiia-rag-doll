import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.retrieval.rules_lawyer import (
    load_game_profile,
    classify_query,
    extract_keywords,
    _get_collection_and_index
)

load_game_profile("data/asl_profile.json")
collection, rule_index = _get_collection_and_index()

print("=" * 80)
print("INVESTIGATING TC #34: Advance Phase and Foxholes")
print("=" * 80)

q34 = "Advance phase and foxholes: Can a squad advance into an adjacent hex and into a foxhole in the same move?"
cls34 = classify_query(q34)
print("Classifier output:", cls34)

# Check if B27.13 exists in rule index or chroma
r_b27_13 = collection.get(where={"rule_number": "B27.13"})
print(f"Chroma lookup for 'B27.13': {len(r_b27_13['ids'])} chunks")
if r_b27_13['ids']:
    for d, m in zip(r_b27_13['documents'], r_b27_13['metadatas']):
        print(f"  Doc: {m.get('source_doc')} | P{m.get('priority')}: {d[:200]}...")

# Check B27.1, 27.13
for rn in ["27.13", "B27.13", "B27.1", "27.1", "A4.7", "4.7"]:
    res = collection.get(where={"rule_number": rn})
    print(f"Rule '{rn}' in Chroma: {len(res['ids'])} chunks")

print("\n" + "=" * 80)
print("INVESTIGATING TC #55: Close Combat Sequence and Infiltration")
print("=" * 80)

q55 = "Close Combat Sequence and Infiltration: So my opponent and I just completed a Close Combat. No ambush was obtained. I had 2:1 odds and rolled a 5, eliminating him, he rolled Snakes. He has the option now of withdrawing from the hex and no elimination occurs to either side, or he can stay and both participants are eliminated, correct?"
cls55 = classify_query(q55)
print("Classifier output:", cls55)

# Check A11.22 in Chroma
for rn in ["A11.22", "11.22", "A11.12", "11.12"]:
    res = collection.get(where={"rule_number": rn})
    print(f"Rule '{rn}' in Chroma: {len(res['ids'])} chunks")
    if res['ids']:
        for d, m in zip(res['documents'], res['metadatas']):
            print(f"  [{rn}] Doc: {m.get('source_doc')} | P{m.get('priority')}: {d[:200]}...")
