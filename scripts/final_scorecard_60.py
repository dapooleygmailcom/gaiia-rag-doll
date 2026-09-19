import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("data/eval/asl_bgg_eval_checkpoint.json", "r", encoding="utf-8") as f:
    chk = json.load(f)

results = chk.get("results", [])
print(f"Total evaluated test cases: {len(results)}\n")

active = [r for r in results if not r.get("skipped")]
with_rules = [r for r in active if r.get("expected_rules")]
non_rule = [r for r in active if not r.get("expected_rules")]

exact_hits = [r for r in with_rules if r.get("rule_recall") == 1.0]
agreed_hits = [r for r in with_rules if (r.get("adjudication") or {}).get("ruling") == "AGREE" or r.get("rule_recall") == 1.0]
disagreed = [r for r in with_rules if (r.get("adjudication") or {}).get("ruling") == "DISAGREE"]

base_recalls = [r["rule_recall"] for r in with_rules if r.get("rule_recall") is not None]
adj_recalls = [r.get("adjudicated_recall", r.get("rule_recall", 0.0)) for r in with_rules if r.get("adjudicated_recall") is not None]

avg_base_rec = sum(base_recalls) / len(base_recalls) if base_recalls else 0.0
avg_adj_rec = sum(adj_recalls) / len(adj_recalls) if adj_recalls else 0.0

print(f"Summary Statistics (Items 1 to {len(results)}):")
print(f"  • Total Evaluated: {len(active)}")
print(f"  • Active Rule-Bearing Queries: {len(with_rules)}")
print(f"  • Non-Citation Queries (Scenario/Errata): {len(non_rule)}")
print(f"  • Strict Exact Matches (Recall = 1.0): {len(exact_hits)} / {len(with_rules)} ({len(exact_hits)/len(with_rules)*100:.1f}%)")
print(f"  • Adjudicated Substantive Accuracy: {len(agreed_hits)} / {len(with_rules)} ({len(agreed_hits)/len(with_rules)*100:.1f}%)")
print(f"  • Average Strict Recall: {avg_base_rec:.2f}")
print(f"  • Average Adjudicated Recall: {avg_adj_rec:.2f}")
print(f"  • Mechanical Failures (DISAGREE): {len(disagreed)} / {len(with_rules)} ({len(disagreed)/len(with_rules)*100:.1f}%)\n")

print("Disagreed Items Breakdown:")
for d in disagreed:
    adj = d.get("adjudication") or {}
    print(f"  • TC #{d.get('overall_index', '?')} ({d.get('id')}): '{d.get('title')}'")
    print(f"    Expected: {d.get('expected_rules')}")
    print(f"    Retrieved: {d.get('retrieved_rules')[:10]}")
    print(f"    Rationale: {adj.get('rationale')}\n")
