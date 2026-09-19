import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def analyze():
    chk_path = "data/eval/asl_bgg_eval_checkpoint.json.bak_clean_reingest"
    with open(chk_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    print(f"Loaded {len(results)} test cases from {chk_path}\n")

    print(f"{'TC':<4} | {'ID':<18} | {'Title':<45} | {'Exp':<4} | {'Hits':<4} | {'Base Rec':<8} | {'Adj Rec':<8} | {'Ruling':<8}")
    print("-" * 110)

    for i, r in enumerate(results):
        tc_id = r.get("id", "")
        title = r.get("title", "")[:45]
        exp = len(r.get("expected_rules", []))
        hits = len(r.get("hits", []))
        b_rec = f"{r.get('rule_recall', 0.0):.2f}" if r.get('rule_recall') is not None else "N/A"
        a_rec = f"{r.get('adjudicated_recall', 0.0):.2f}" if r.get('adjudicated_recall') is not None else "N/A"
        adj = r.get("adjudication") or {}
        ruling = adj.get("ruling", "None")
        print(f"{i+1:<4} | {tc_id:<18} | {title:<45} | {exp:<4} | {hits:<4} | {b_rec:<8} | {a_rec:<8} | {ruling:<8}")

    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN BY TEST CASE:")
    print("=" * 80)

    for i, r in enumerate(results):
        print(f"\n--- [TC {i+1}] {r.get('title')} ({r.get('id')}) ---")
        exp = r.get("expected_rules", [])
        ret = r.get("retrieved_rules", [])
        hits = r.get("hits", [])
        print(f"Expected Rules ({len(exp)}): {exp}")
        print(f"Retrieved Rules ({len(ret)}): {ret}")
        print(f"Hits ({len(hits)}): {hits}")
        print(f"Baseline Recall: {r.get('rule_recall')} | Precision: {r.get('rule_precision')}")
        
        # Check prefix matches
        unprefixed_matches = []
        for e in exp:
            clean_e = e[1:] if e and e[0].isalpha() else e
            if clean_e in ret and e not in ret:
                unprefixed_matches.append((e, clean_e))
        if unprefixed_matches:
            print(f"  ⚠️ Prefix Mismatch Found: {unprefixed_matches} (expected chapter prefix, but retrieved un-prefixed decimal)")

        adj = r.get("adjudication")
        if adj:
            print(f"Adjudication: Ruling={adj.get('ruling')} | Substantive Accuracy={adj.get('substantive_accuracy')}")
            print(f"Rationale: {adj.get('rationale')}")
        else:
            print("Adjudication: None")

if __name__ == "__main__":
    analyze()
