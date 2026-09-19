import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def summarize():
    path = "data/eval/upfront_cloud_eval_61_100.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    results = data.get("results", [])
    print(f"Total results recorded: {len(results)}")
    
    hits = [r for r in results if r.get("rule_hit")]
    misses = [r for r in results if r.get("rule_hit") is False]
    general = [r for r in results if not r.get("expected_rules")]
    
    print(f"Hits: {len(hits)}")
    print(f"Misses: {len(misses)}")
    print(f"General/Unlabeled: {len(general)}")
    print("\n--- SAMPLE HITS ---")
    for r in hits[:6]:
        print(f"ID: {r['id']} ({r.get('title', '')[:35]})")
        print(f"  Expected: {r['expected_rules']}")
        print(f"  Retrieved: {r['retrieved_rules'][:5]}")
        print(f"  Recall: {r.get('rule_recall')}")
        print(f"  Verdict: {r.get('debug_info', {}).get('verdict')}")
        print()

    print("--- SAMPLE MISSES ---")
    for r in misses[:4]:
        print(f"ID: {r['id']} ({r.get('title', '')[:35]})")
        print(f"  Expected: {r['expected_rules']}")
        print(f"  Retrieved: {r['retrieved_rules'][:5]}")
        print()

if __name__ == "__main__":
    summarize()
