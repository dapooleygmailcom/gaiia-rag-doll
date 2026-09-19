"""
retest_failures.py

Retests ONLY the 10 previously failing test cases against the updated production Cloud endpoint:
- Verifies impact of:
  1. Section Root Rule Bundling / Promotion
  2. Nova Lite fast model (clean distillation & domain comprehension)
  3. Double citation aperture (top 50 chunks)
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
import re

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_FILE = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_bgg_eval_benchmark.json")
API_URL = "https://8ifjmds7mk.execute-api.ap-southeast-2.amazonaws.com/api/ask"

TARGET_FAILURE_IDS = [
    "bgg_uf_3589681", # TC 14 (Expected 28.46)
    "bgg_uf_3371642", # TC 25 (Expected 19.2)
    "bgg_uf_3206922", # TC 34 (Expected 23.7)
    "bgg_uf_3199417", # TC 37 (Expected 4.2)
    "bgg_uf_2984341", # TC 54 (Expected 20.6)
    "bgg_uf_2688478", # TC 79 (Expected 16.3, 16.5, 20.52...)
    "bgg_uf_2633076", # TC 83 (Expected 25.8)
    "bgg_uf_2479530", # TC 89 (Expected 19.11)
    "bgg_uf_2475533", # TC 90 (Expected 20.51)
    "bgg_uf_2324650", # TC 98 (Expected 10.12)
]

def query_api(query):
    payload = {
        "tenantSlug": "internal-test",
        "titleSlug": "up-front",
        "query": query,
        "captchaToken": "test-eval-suite-token"
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST"
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=40) as response:
        body = response.read().decode("utf-8")
        rtt = round(time.time() - t0, 2)
        data = json.loads(body)
        ruling = data.get("ruling", {})
        raw_citations = data.get("citations") or ruling.get("citations", [])
        answer = data.get("answerMarkdown") or ruling.get("answer", "")
        debug_info = data.get("debug", {})
        exec_ms = data.get("executionTimeMs")

        retrieved_rules = []
        for c in raw_citations:
            rn = c.get("ruleNumber")
            if rn:
                retrieved_rules.append(str(rn).strip())

        # Also extract citations from text
        text_citations = re.findall(r'\[(?:UP FRONT|Rule|Section)?\s*([\d\.]+)\]', answer, re.IGNORECASE)
        for tc in text_citations:
            tc_clean = tc.strip()
            if tc_clean not in retrieved_rules:
                retrieved_rules.append(tc_clean)

        return {
            "rtt": rtt,
            "exec_ms": exec_ms,
            "citations_count": len(raw_citations),
            "retrieved_rules": retrieved_rules,
            "answer_preview": answer[:200],
            "verdict": data.get("verdict") or ruling.get("verdict"),
            "debug": debug_info
        }

def main():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    bench_map = {b["id"]: b for b in benchmark}

    print("=" * 85)
    print("🎯 RETESTING 10 PREVIOUS RETRIEVAL FAILURES ON LIVE CLOUD PRODUCTION")
    print("=" * 85)

    results = []
    for idx_target, cid in enumerate(TARGET_FAILURE_IDS, 1):
        b = bench_map[cid]
        overall_idx = benchmark.index(b) + 1
        exp_rules = b.get("expected_rule_citations", [])
        print(f"\n[{idx_target}/10] Testing TC #{overall_idx} ({cid}): '{b.get('title')[:35]}'...")
        print(f"       Expected: {exp_rules}")

        try:
            res = query_api(b["query"])
            ret_rules = res["retrieved_rules"]

            hits = [r for r in exp_rules if any(r == ret or r.startswith(ret) or ret.startswith(r) for ret in ret_rules)]
            is_hit = len(hits) > 0
            recall = round(len(hits) / len(exp_rules), 2) if exp_rules else 0.0

            rank_info = []
            for h in hits:
                exact_rank = [i+1 for i, r in enumerate(ret_rules) if r == h or r.startswith(h) or h.startswith(r)]
                rank_info.append(f"{h}@rank#{exact_rank[0]}" if exact_rank else h)

            hit_symbol = "✅ HIT" if is_hit else "❌ MISS"
            print(f"       Outcome : {hit_symbol} (Recall: {recall}) | Latency: {res['rtt']}s (Server: {res['exec_ms']}ms)")
            print(f"       Citations Count: {res['citations_count']} | Hits: {rank_info}")
            print(f"       Top 10 Citations: {ret_rules[:10]}")

            results.append({
                "overall_idx": overall_idx,
                "id": cid,
                "title": b.get("title"),
                "expected": exp_rules,
                "is_hit": is_hit,
                "recall": recall,
                "hits": hits,
                "rank_info": rank_info,
                "latency": res["rtt"],
                "exec_ms": res["exec_ms"],
                "citations_count": res["citations_count"],
                "top_citations": ret_rules[:10],
                "ret_rules": ret_rules
            })
        except Exception as e:
            print(f"       Error: {e}")
            results.append({
                "overall_idx": overall_idx,
                "id": cid,
                "title": b.get("title"),
                "expected": exp_rules,
                "is_hit": False,
                "error": str(e)
            })

    # Summary table
    print("\n" + "=" * 85)
    print("📊 BEFORE vs AFTER RETEST SCORECARD (10 FAILURES ONLY)")
    print("=" * 85)
    print(f"{'#':<4} | {'ID':<16} | {'Expected':<12} | {'Prev':<6} | {'New':<6} | {'New Hits (Rank)':<20} | {'Latency':<8}")
    print("-" * 85)
    new_hits_count = 0
    for r in results:
        prev_status = "MISS"
        new_status = "HIT ✅" if r.get("is_hit") else "MISS ❌"
        if r.get("is_hit"):
            new_hits_count += 1
        exp_str = ", ".join(r.get("expected", []))[:12]
        hits_str = ", ".join(r.get("rank_info", []))[:20] if r.get("is_hit") else "-"
        lat_str = f"{r.get('latency', 0)}s"
        print(f"{r.get('overall_idx'):<4} | {r.get('id'):<16} | {exp_str:<12} | {prev_status:<6} | {new_status:<6} | {hits_str:<20} | {lat_str:<8}")
    print("=" * 85)
    print(f"🎯 SUMMARY: {new_hits_count} of 10 previously failing cases are now HITS! (+{new_hits_count*10}% recovery rate)")
    print("=" * 85)

    out_file = os.path.join(PROJECT_ROOT, "data", "eval", "retest_failures_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {out_file}")

if __name__ == "__main__":
    main()
