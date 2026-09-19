"""
deep_gap_analysis.py

Performs deep structural analysis across the 1-60 and 61-100 gaps:
1. Ingestion: Analyzes rule representation in up_front_rule_index.json, DynamoDB, and cooccurrence graph.
2. Retrieval: Compares Local vs Cloud retrieval pipeline, rank positions, candidate list depth, and graph traversal.
3. Model: Compares Nova Pro vs Ollama query synthesis, distillation, HyDE, and citation extraction.
"""

import os
import sys
import json

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_CHECKPOINT = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_bgg_eval_checkpoint.json")
CLOUD_CHECKPOINT = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_cloud_eval_1_100.json")
BENCHMARK_FILE = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_bgg_eval_benchmark.json")

RULE_INDEX_FILE = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "gaiia-rag-doll-cloud", "backend", "data", "up_front_rule_index.json"))
SECTION_TREE_FILE = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "gaiia-rag-doll-cloud", "backend", "data", "up_front_section_tree.json"))
GRAPH_FILE = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "gaiia-rag-doll-cloud", "backend", "data", "up_front_cooccurrence_graph.json"))

def main():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    with open(LOCAL_CHECKPOINT, "r", encoding="utf-8") as f:
        loc_results = json.load(f).get("results", [])
    with open(CLOUD_CHECKPOINT, "r", encoding="utf-8") as f:
        cld_results = json.load(f).get("results", [])
    with open(RULE_INDEX_FILE, "r", encoding="utf-8") as f:
        rule_index = json.load(f)
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        graph = json.load(f)

    loc_map = {r["id"]: r for r in loc_results}
    cld_map = {r["id"]: r for r in cld_results}
    bench_map = {b["id"]: b for b in benchmark}

    # Case lists
    cases_1_60 = ["bgg_uf_3589681", "bgg_uf_3371642", "bgg_uf_3199417", "bgg_uf_2984341", "bgg_uf_3206922"]
    cases_61_100 = ["bgg_uf_2688478", "bgg_uf_2633076", "bgg_uf_2479530", "bgg_uf_2475533", "bgg_uf_2324650"]

    print("=" * 90)
    print("SECTION 1: GAP ANALYSIS ON 1-60 (LOCAL vs CLOUD)")
    print("=" * 90)

    for cid in cases_1_60:
        b = bench_map.get(cid)
        idx = benchmark.index(b) + 1
        l = loc_map.get(cid)
        c = cld_map.get(cid)

        print(f"\n--- [Case #{idx}] {cid}: '{b.get('title')}' ---")
        print(f"Query: {b.get('query')}")
        print(f"Expected Rules: {b.get('expected_rule_citations')}")

        # Local analysis
        if l:
            l_rules = l.get("retrieved_rules", [])
            l_hits = l.get("hits", [])
            print(f"Local Hit: {l.get('rule_hit')} | Recall: {l.get('rule_recall')} | Hits: {l_hits}")
            print(f"Local Total Rules Retrieved: {len(l_rules)}")
            for exp_r in b.get("expected_rule_citations", []):
                if exp_r in l_rules:
                    rank = l_rules.index(exp_r) + 1
                    print(f"  -> Local found '{exp_r}' at rank #{rank} of {len(l_rules)}")
                else:
                    # check partial
                    partials = [r for r in l_rules if r.startswith(exp_r) or exp_r.startswith(r)]
                    print(f"  -> Local did NOT find exact '{exp_r}'. Partials: {partials}")

        # Cloud analysis
        if c:
            c_rules = c.get("retrieved_rules", [])
            c_hits = c.get("hits", [])
            c_dbg = c.get("debug", {})
            print(f"Cloud Hit: {c.get('rule_hit')} | Recall: {c.get('rule_recall')} | Hits: {c_hits}")
            print(f"Cloud Total Rules Retrieved (Top Citations): {len(c_rules)}")
            print(f"Cloud Citations: {c_rules}")
            print(f"Cloud Distilled Question: '{c_dbg.get('distilled_question')}'")
            print(f"Cloud HyDE: '{c_dbg.get('hyde_clause')}'")
            print(f"Cloud Query Type: '{c_dbg.get('query_type')}'")
            print(f"Cloud Expansions: Parents={c_dbg.get('num_parent_expansions')}, Cooc={c_dbg.get('num_cooccurrence_expansions')}, TotalRetrieved={c_dbg.get('num_retrieved')}")

        # Ingestion rule inspection
        for exp_r in b.get("expected_rule_citations", []):
            r_info = rule_index.get(exp_r)
            if isinstance(r_info, list):
                print(f"  [Ingestion for {exp_r}] Entries: {len(r_info)} chunks: {[e.get('chunk_id') for e in r_info if isinstance(e, dict)]}")
                if r_info and isinstance(r_info[0], dict):
                    print(f"  [Rule Content Preview]: {r_info[0].get('text', r_info[0].get('content', ''))[:200]}...")
            elif isinstance(r_info, dict):
                print(f"  [Ingestion for {exp_r}] Title: '{r_info.get('title')}' | Chunks: {r_info.get('chunk_ids')}")
                print(f"  [Rule Content Preview]: {r_info.get('content', '')[:200]}...")
            else:
                print(f"  [Ingestion for {exp_r}] NOT FOUND in rule_index.json!")

            node = graph.get("nodes", {}).get(exp_r)
            edges = [e for e in graph.get("edges", []) if e.get("source") == exp_r or e.get("target") == exp_r]
            print(f"  [Graph for {exp_r}] Node Exists: {bool(node)} | Connected Edges: {len(edges)}")

    print("\n" + "=" * 90)
    print("SECTION 2: GAP ANALYSIS ON 61-100 (CLOUD MISSES)")
    print("=" * 90)

    for cid in cases_61_100:
        b = bench_map.get(cid)
        idx = benchmark.index(b) + 1
        c = cld_map.get(cid)

        print(f"\n--- [Case #{idx}] {cid}: '{b.get('title')}' ---")
        print(f"Query: {b.get('query')}")
        print(f"Expected Rules: {b.get('expected_rule_citations')}")

        if c:
            c_rules = c.get("retrieved_rules", [])
            c_dbg = c.get("debug", {})
            print(f"Cloud Hit: {c.get('rule_hit')} | Recall: {c.get('rule_recall')}")
            print(f"Cloud Citations: {c_rules}")
            print(f"Cloud Distilled Question: '{c_dbg.get('distilled_question')}'")
            print(f"Cloud HyDE: '{c_dbg.get('hyde_clause')}'")
            print(f"Cloud Query Type: '{c_dbg.get('query_type')}'")
            print(f"Cloud Expansions: Parents={c_dbg.get('num_parent_expansions')}, Cooc={c_dbg.get('num_cooccurrence_expansions')}, TotalRetrieved={c_dbg.get('num_retrieved')}")

        for exp_r in b.get("expected_rule_citations", []):
            r_info = rule_index.get(exp_r)
            if isinstance(r_info, list):
                print(f"  [Ingestion for {exp_r}] Entries: {len(r_info)} chunks: {[e.get('chunk_id') for e in r_info if isinstance(e, dict)]}")
                if r_info and isinstance(r_info[0], dict):
                    print(f"  [Rule Content Preview]: {r_info[0].get('text', r_info[0].get('content', ''))[:160]}...")
            elif isinstance(r_info, dict):
                print(f"  [Ingestion for {exp_r}] Title: '{r_info.get('title')}' | Chunks: {r_info.get('chunk_ids')}")
                print(f"  [Rule Content Preview]: {r_info.get('content', '')[:160]}...")
            else:
                print(f"  [Ingestion for {exp_r}] NOT FOUND in rule_index.json!")

            node = graph.get("nodes", {}).get(exp_r)
            edges = [e for e in graph.get("edges", []) if e.get("source") == exp_r or e.get("target") == exp_r]
            print(f"  [Graph for {exp_r}] Node Exists: {bool(node)} | Connected Edges: {len(edges)}")

if __name__ == "__main__":
    main()
