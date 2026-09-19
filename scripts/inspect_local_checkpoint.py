import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def inspect_local():
    path = os.path.join("data", "eval", "upfront_bgg_eval_checkpoint.json")
    if not os.path.exists(path):
        print("Checkpoint does not exist:", path)
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    print(f"Total results in local checkpoint: {len(results)}")
    
    with open(os.path.join("data", "eval", "upfront_bgg_eval_benchmark.json"), "r", encoding="utf-8") as f:
        benchmark = json.load(f)
        
    first_60_ids = [item["id"] for item in benchmark[:60]]
    completed_ids = set(r["id"] for r in results)
    
    matching_1_60 = [r for r in results if r["id"] in set(first_60_ids)]
    print(f"Matching items from tests 1-60 in local checkpoint: {len(matching_1_60)} / 60")
    
    hits = [r for r in matching_1_60 if r.get("rule_hit")]
    recalls = [r["rule_recall"] for r in matching_1_60 if r.get("rule_recall") is not None]
    print(f"Local hits in 1-60: {len(hits)}")
    if recalls:
        print(f"Local avg recall in 1-60: {sum(recalls)/len(recalls):.2f}")
    lats = [r.get("latency_seconds", 0) for r in matching_1_60]
    if lats:
        print(f"Local avg latency in 1-60: {sum(lats)/len(lats):.2f}s")

if __name__ == "__main__":
    inspect_local()
