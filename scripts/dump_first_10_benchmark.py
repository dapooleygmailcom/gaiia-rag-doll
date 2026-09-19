import json

with open("data/eval/asl_bgg_eval_benchmark.json", "r", encoding="utf-8") as f:
    bench = json.load(f)

with open("first_10_benchmark_utf8.txt", "w", encoding="utf-8") as out:
    out.write(f"Total benchmark items: {len(bench)}\n\n")

    for i in range(min(10, len(bench))):
        item = bench[i]
        out.write("=" * 80 + "\n")
        out.write(f"TC #{i+1} | ID: {item.get('id')} | Thread: {item.get('thread_id')}\n")
        out.write(f"Title: {item.get('title')}\n")
        out.write(f"Expected Rules: {item.get('expected_rules') or item.get('expected_rule_citations')}\n")
        out.write(f"Query Intent: {item.get('query_intent')} | Concepts: {item.get('concepts')}\n")
        out.write(f"Query:\n{item.get('query')}\n")
        out.write("-" * 40 + "\n")
        out.write(f"Ground Truth Answer:\n{item.get('ground_truth_answer')}\n")
        out.write("=" * 80 + "\n\n")

print("Wrote UTF-8 first_10_benchmark_utf8.txt successfully.")
