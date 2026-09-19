import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("data/eval/asl_bgg_eval_checkpoint.json", "r", encoding="utf-8") as f:
    chk = json.load(f)

results = chk.get("results", [])

for target_id in ["bgg_asl_3682996", "bgg_asl_3657611"]:
    item = next((r for r in results if r["id"] == target_id), None)
    if not item:
        print(f"ID {target_id} not found!")
        continue
    print("=" * 80)
    print(f"TC ID: {item.get('id')} | Overall Index: {item.get('overall_index')}")
    print(f"Title: {item.get('title')}")
    print(f"Expected Rules: {item.get('expected_rules')}")
    print(f"Retrieved Rules: {item.get('retrieved_rules')}")
    print(f"Hits: {item.get('hits')}")
    print(f"Rule Recall: {item.get('rule_recall')}")
    print(f"Adjudication Ruling: {(item.get('adjudication') or {}).get('ruling')}")
    print(f"Adjudication Rationale: {(item.get('adjudication') or {}).get('rationale')}")
    print("-" * 40)
    print("QUERY:")
    print(item.get("query"))
    print("-" * 40)
    print("GENERATED ANSWER:")
    print(item.get("generated_answer"))
    print("-" * 40)
    print("GROUND TRUTH ANSWER:")
    print(item.get("ground_truth_answer"))
    print("=" * 80 + "\n")
