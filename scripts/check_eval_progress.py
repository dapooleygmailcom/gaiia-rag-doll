import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("data/eval/asl_bgg_eval_checkpoint.json", "r", encoding="utf-8") as f:
    chk = json.load(f)

results = chk.get("results", [])
print(f"Total evaluated items in checkpoint: {len(results)}\n")
print(f"{'#':<3} | {'ID':<18} | {'Title':<40} | {'Base Rec':<9} | {'Adj Rec':<8} | {'Ruling':<9}")
print("-" * 95)

for idx, r in enumerate(results):
    adj = r.get("adjudication") or {}
    b_rec = f"{r.get('rule_recall', 0.0):.2f}" if r.get('rule_recall') is not None else "N/A"
    a_rec = f"{r.get('adjudicated_recall', 0.0):.2f}" if r.get('adjudicated_recall') is not None else "N/A"
    ruling = adj.get("ruling", "EXACT" if r.get("rule_recall") == 1.0 else "-")
    title = r.get("title", "")[:40]
    print(f"{idx+1:<3} | {r.get('id', ''):<18} | {title:<40} | {b_rec:<9} | {a_rec:<8} | {ruling:<9}")
