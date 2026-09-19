import json
import os

def inspect_eval():
    print("=== INSPECTING ASL EVAL FILES ===")
    eval_dir = "data/eval"
    files = os.listdir(eval_dir)
    for f in files:
        if "asl" in f:
            path = os.path.join(eval_dir, f)
            size = os.path.getsize(path)
            print(f"File: {f} ({size} bytes)")
    
    # Check benchmark
    with open("data/eval/asl_bgg_eval_benchmark.json", "r", encoding="utf-8") as f:
        bench = json.load(f)
    print(f"\nTotal benchmark items: {len(bench)}")
    
    # Check checkpoints
    for chk_file in ["asl_bgg_eval_checkpoint.json", "asl_bgg_eval_checkpoint.json.bak_clean_reingest"]:
        p = os.path.join(eval_dir, chk_file)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                chk = json.load(f)
            print(f"\nCheckpoint '{chk_file}':")
            print(f"  Last update: {chk.get('last_update')}")
            print(f"  Total in suite: {chk.get('total_in_suite')}")
            results = chk.get("results", [])
            print(f"  Results count: {len(results)}")
            for idx, r in enumerate(results):
                print(f"    [{idx+1}] ID: {r.get('id')} | Overall Index: {r.get('overall_index')} | Title: {r.get('title')}")
                print(f"        Expected Rules: {r.get('expected_rules')}")
                print(f"        Retrieved Rules: {r.get('retrieved_rules')}")
                print(f"        Hits: {r.get('hits')}")
                print(f"        Rule Recall: {r.get('rule_recall')}")
                print(f"        Rule Precision: {r.get('rule_precision')}")
                print(f"        Rule Hit: {r.get('rule_hit')}")
                adj = r.get("adjudication")
                if adj:
                    print(f"        Adjudication: Ruling={adj.get('ruling')}, Accuracy={adj.get('substantive_accuracy')}, Conf={adj.get('confidence')}")
                    print(f"        Rationale: {adj.get('rationale')}")
                else:
                    print("        Adjudication: None")

if __name__ == "__main__":
    inspect_eval()
