"""
analyze_cloud_gaps.py

Analyzes retrieval misses and gap with local baseline across:
- Tests 1-60: The 4 local-only hits and 1 common miss
- Tests 61-100: The 5 cloud misses

For each case, analyzes:
1. Expected rule(s) and question text
2. Ingestion: Is the expected rule in DynamoDB rule index / profile?
3. Retrieval: Where did it rank in candidate retrieval? What stages found what?
4. Model: Did query distillation or HyDE shift focus away? Did Nova Pro see it?
"""

import os
import sys
import json

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_CHECKPOINT = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_bgg_eval_checkpoint.json")
CLOUD_100_CHECKPOINT = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_cloud_eval_1_100.json")
BENCHMARK_FILE = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_bgg_eval_benchmark.json")

# Cloud rule index and profile
RULE_INDEX_FILE = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "gaiia-rag-doll-cloud", "backend", "data", "up_front_rule_index.json"))
PROFILE_FILE = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "gaiia-rag-doll-cloud", "backend", "data", "up_front_profile.json"))
SECTION_TREE_FILE = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "gaiia-rag-doll-cloud", "backend", "data", "up_front_section_tree.json"))
GRAPH_FILE = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "gaiia-rag-doll-cloud", "backend", "data", "up_front_cooccurrence_graph.json"))

def main():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    with open(CLOUD_100_CHECKPOINT, "r", encoding="utf-8") as f:
        cld_data = json.load(f)
    cld_results = cld_data.get("results", [])
    cld_dict = {r["id"]: r for r in cld_results}

    with open(LOCAL_CHECKPOINT, "r", encoding="utf-8") as f:
        loc_data = json.load(f)
    loc_dict = {r["id"]: r for r in loc_data.get("results", [])}

    with open(RULE_INDEX_FILE, "r", encoding="utf-8") as f:
        rule_index = json.load(f)

    with open(PROFILE_FILE, "r", encoding="utf-8") as f:
        profile = json.load(f)

    with open(SECTION_TREE_FILE, "r", encoding="utf-8") as f:
        section_tree = json.load(f)

    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        graph = json.load(f)

    print("=== DATASETS LOADED ===")
    print(f"Benchmark: {len(benchmark)} items")
    print(f"Cloud 100 results: {len(cld_results)} items")
    print(f"Local results: {len(loc_dict)} items")
    print(f"Rule Index items: {len(rule_index)}")
    print(f"Profile rules: {len(profile.get('rules', {}))}")

    # Identify misses in 1-60
    loc_only_60 = []
    both_miss_60 = []
    for b in benchmark[:60]:
        cid = b["id"]
        exp = b.get("expected_rule_citations", [])
        if not exp:
            continue
        l_hit = loc_dict.get(cid, {}).get("rule_hit", False)
        c_hit = cld_dict.get(cid, {}).get("rule_hit", False)
        if l_hit and not c_hit:
            loc_only_60.append((b, loc_dict.get(cid), cld_dict.get(cid)))
        elif not l_hit and not c_hit:
            both_miss_60.append((b, loc_dict.get(cid), cld_dict.get(cid)))

    # Identify misses in 61-100
    misses_61_100 = []
    for b in benchmark[60:100]:
        cid = b["id"]
        exp = b.get("expected_rule_citations", [])
        if not exp:
            continue
        c_item = cld_dict.get(cid)
        if c_item and not c_item.get("rule_hit", False):
            misses_61_100.append((b, c_item))

    print(f"\nDiscrepancies in 1-60 (Local Hit, Cloud Miss): {len(loc_only_60)}")
    print(f"Both Missed in 1-60: {len(both_miss_60)}")
    print(f"Misses in 61-100: {len(misses_61_100)}")

    def check_ingestion(rule_num):
        in_idx = rule_num in rule_index
        in_prof = rule_num in profile.get("rules", {})
        matches_idx = [k for k in rule_index.keys() if k == rule_num or k.startswith(rule_num + ".") or rule_num.startswith(k + ".")]
        matches_prof = [k for k in profile.get("rules", {}).keys() if k == rule_num or k.startswith(rule_num + ".") or rule_num.startswith(k + ".")]
        return {
            "exact_rule_index": in_idx,
            "exact_profile": in_prof,
            "related_index": matches_idx[:5],
            "related_profile": matches_prof[:5]
        }

    print("\n" + "="*80)
    print("DETAILED ANALYSIS: 1-60 LOCAL HITS vs CLOUD MISSES")
    print("="*80)
    for b, l_item, c_item in loc_only_60:
        idx = benchmark.index(b) + 1
        cid = b["id"]
        title = b.get("title", "")
        exp = b.get("expected_rule_citations", [])
        print(f"\n--- [#{idx}] {cid}: {title} ---")
        print(f"Query: {b['query'][:120]}...")
        print(f"Expected Rules: {exp}")
        print(f"Local Retrieved: {l_item.get('retrieved_rules', [])[:10]}")
        print(f"Cloud Retrieved: {c_item.get('retrieved_rules', [])[:10]}")
        print(f"Cloud Debug: Distilled='{c_item.get('debug', {}).get('distilled_question')}', QueryType='{c_item.get('debug', {}).get('query_type')}'")
        for r in exp:
            ing_stat = check_ingestion(r)
            print(f"  Ingestion check for Rule '{r}': Index={ing_stat['exact_rule_index']} (Related: {ing_stat['related_index']}), Profile={ing_stat['exact_profile']} (Related: {ing_stat['related_profile']})")

    print("\n" + "="*80)
    print("DETAILED ANALYSIS: 1-60 BOTH MISSED")
    print("="*80)
    for b, l_item, c_item in both_miss_60:
        idx = benchmark.index(b) + 1
        cid = b["id"]
        title = b.get("title", "")
        exp = b.get("expected_rule_citations", [])
        print(f"\n--- [#{idx}] {cid}: {title} ---")
        print(f"Query: {b['query'][:120]}...")
        print(f"Expected Rules: {exp}")
        print(f"Local Retrieved: {l_item.get('retrieved_rules', [])[:10] if l_item else 'None'}")
        print(f"Cloud Retrieved: {c_item.get('retrieved_rules', [])[:10] if c_item else 'None'}")
        for r in exp:
            ing_stat = check_ingestion(r)
            print(f"  Ingestion check for Rule '{r}': Index={ing_stat['exact_rule_index']}, Profile={ing_stat['exact_profile']}")

    print("\n" + "="*80)
    print("DETAILED ANALYSIS: 61-100 CLOUD MISSES")
    print("="*80)
    for b, c_item in misses_61_100:
        idx = benchmark.index(b) + 1
        cid = b["id"]
        title = b.get("title", "")
        exp = b.get("expected_rule_citations", [])
        print(f"\n--- [#{idx}] {cid}: {title} ---")
        print(f"Query: {b['query'][:120]}...")
        print(f"Expected Rules: {exp}")
        print(f"Cloud Retrieved: {c_item.get('retrieved_rules', [])[:10]}")
        print(f"Cloud Debug: Distilled='{c_item.get('debug', {}).get('distilled_question')}', QueryType='{c_item.get('debug', {}).get('query_type')}'")
        for r in exp:
            ing_stat = check_ingestion(r)
            print(f"  Ingestion check for Rule '{r}': Index={ing_stat['exact_rule_index']} (Related: {ing_stat['related_index']}), Profile={ing_stat['exact_profile']} (Related: {ing_stat['related_profile']})")

if __name__ == "__main__":
    main()
