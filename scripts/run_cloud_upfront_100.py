"""
run_cloud_upfront_100.py

Authoritative 100-Case Up Front BGG Benchmark Runner against Production Gaiia RAG Doll Cloud.
Executes the first 100 test cases from the official BGG Up Front benchmark against live AWS API Gateway,
Bedrock Nova Pro, and DynamoDB backend.

Features:
- Adaptive contract parsing (supports ruling object payload and top-level fields)
- Concurrent worker support (default 2 workers for optimal throughput and Bedrock quota safety)
- Robust retry with exponential backoff on network errors
- Thread-safe atomic checkpointing (resumable at any point)
- Comprehensive metrics: Hit Rate, Recall, RTT, Cloud Execution Time, P50/P90/P95 latency
- Automatically produces a publication-grade Markdown scorecard and JSON dataset
"""

import os
import sys
import json
import time
import re
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Directory paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_BENCHMARK = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_bgg_eval_benchmark.json")
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_cloud_eval_1_100.json")
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_cloud_100_report.md")
DEFAULT_API_URL = "https://8ifjmds7mk.execute-api.ap-southeast-2.amazonaws.com/api/ask"


def query_cloud_api(query: str,
                    api_url: str = DEFAULT_API_URL,
                    tenant_slug: str = "internal-test",
                    title_slug: str = "up-front",
                    timeout: int = 40,
                    retries: int = 3):
    """Execute evaluation query against deployed AWS API Gateway endpoint with retry."""
    payload = {
        "tenantSlug": tenant_slug,
        "titleSlug": title_slug,
        "query": query,
        "captchaToken": "test-eval-suite-token"
    }

    req_data = json.dumps(payload).encode("utf-8")
    last_err = None

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            api_url,
            data=req_data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST"
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                rtt = round(time.time() - t0, 2)
                data = json.loads(body)

                ruling = data.get("ruling", {})
                verdict = data.get("verdict") or ruling.get("verdict", "UNKNOWN")
                confidence = data.get("confidenceScore") or ruling.get("confidence")
                answer = data.get("answerMarkdown") or ruling.get("answer") or ruling.get("summary", "")
                raw_citations = data.get("citations") or ruling.get("citations", [])
                exec_time_ms = data.get("executionTimeMs")
                debug_info = data.get("debug", {})

                # Extract rule numbers from structured citations
                retrieved_rules = []
                for c in raw_citations:
                    rn = c.get("ruleNumber")
                    if rn:
                        retrieved_rules.append(str(rn).strip())

                # Extract citations from answer text [12.4] or [UP FRONT 12.4] or [Rule 12.4]
                text_citations = re.findall(r'\[(?:UP FRONT|Rule|Section)?\s*([\d\.]+)\]', answer, re.IGNORECASE)
                for tc in text_citations:
                    tc_clean = tc.strip()
                    if tc_clean not in retrieved_rules:
                        retrieved_rules.append(tc_clean)

                dedup_rules = sorted(list(set(retrieved_rules)))

                parsed_debug = {
                    "executionTimeMs": exec_time_ms,
                    "verdict": verdict,
                    "confidenceScore": confidence,
                    "roundTripTimeSec": rtt,
                    "num_citations": len(raw_citations),
                    "query_type": debug_info.get("query_type"),
                    "distilled_question": debug_info.get("distilled_question"),
                    "hyde_clause": debug_info.get("hyde_clause"),
                    "num_retrieved": debug_info.get("num_retrieved"),
                    "attempt": attempt
                }

                return answer, raw_citations, dedup_rules, parsed_debug, None

        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.reason}"
            if e.code in [502, 503, 504] and attempt < retries:
                time.sleep(2 * attempt)
                continue
            break
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            break

    rtt = round(time.time() - t0, 2)
    return f"ERROR: {last_err}", [], [], {"roundTripTimeSec": rtt, "error": last_err}, last_err


def evaluate_single_item(item: dict,
                         overall_idx: int,
                         total_items: int,
                         api_url: str,
                         tenant_slug: str,
                         title_slug: str,
                         timeout: int,
                         retries: int):
    """Run cloud query and evaluate retrieval metrics for one test item."""
    item_id = item["id"]
    title = item.get("title", "")
    query = item["query"]
    intent = item.get("intent", "clarification")
    expected_rules = item.get("expected_rule_citations", [])
    gt_answer = item.get("ground_truth_answer", "")

    t_start = time.time()
    answer, raw_citations, retrieved_rules, debug_info, err = query_cloud_api(
        query=query,
        api_url=api_url,
        tenant_slug=tenant_slug,
        title_slug=title_slug,
        timeout=timeout,
        retries=retries
    )
    latency = round(time.time() - t_start, 2)

    # Calculate retrieval hits and recall
    if expected_rules:
        hits = [r for r in expected_rules if any(r == ret or r.startswith(ret) or ret.startswith(r) for ret in retrieved_rules)]
        rule_hit = len(hits) > 0
        recall = round(len(hits) / len(expected_rules), 2)
    else:
        hits = []
        rule_hit = None
        recall = None

    record = {
        "id": item_id,
        "overall_index": overall_idx,
        "target": "cloud",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "title": title,
        "query": query,
        "intent": intent,
        "expected_rules": expected_rules,
        "retrieved_rules": retrieved_rules,
        "rule_hit": rule_hit,
        "hits": hits,
        "rule_recall": recall,
        "latency_seconds": latency,
        "cloud_exec_ms": debug_info.get("executionTimeMs"),
        "verdict": debug_info.get("verdict", "UNKNOWN"),
        "confidence": debug_info.get("confidenceScore"),
        "generated_answer": answer,
        "ground_truth_answer": gt_answer,
        "citations_count": len(raw_citations),
        "debug": debug_info,
        "error": err
    }

    return record


def save_checkpoint_atomic(checkpoint_path: str, completed_results: list, total_in_suite: int):
    """Atomically save checkpoint file."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    temp_file = f"{checkpoint_path}.tmp"
    payload = {
        "last_updated": datetime.now(timezone.utc).isoformat() + "Z",
        "target": "cloud",
        "total_evaluated": len(completed_results),
        "total_in_suite": total_in_suite,
        "results": sorted(completed_results, key=lambda x: x.get("overall_index", 0))
    }
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_file, checkpoint_path)


def compute_statistics(results: list, total_in_slice: int):
    """Compute detailed benchmark statistics across completed results."""
    eval_with_rules = [r for r in results if r.get("expected_rules")]
    num_with_rules = len(eval_with_rules)

    if eval_with_rules:
        hits = sum(1 for r in eval_with_rules if r.get("rule_hit"))
        hit_rate = round((hits / num_with_rules) * 100, 1)
        recalls = [r["rule_recall"] for r in eval_with_rules if r.get("rule_recall") is not None]
        avg_recall = round(sum(recalls) / len(recalls), 2) if recalls else 0.0
    else:
        hits, hit_rate, avg_recall = 0, 0.0, 0.0

    # Latencies
    rtt_list = sorted([r.get("latency_seconds", 0) for r in results if r.get("latency_seconds") is not None])
    exec_list = sorted([r.get("cloud_exec_ms", 0) for r in results if r.get("cloud_exec_ms") is not None])

    def calc_percentiles(vals):
        if not vals:
            return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        n = len(vals)
        mean_val = round(sum(vals) / n, 2)
        p50 = vals[int(n * 0.50)]
        p90 = vals[min(n - 1, int(n * 0.90))]
        p95 = vals[min(n - 1, int(n * 0.95))]
        return {
            "mean": mean_val,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "min": vals[0],
            "max": vals[-1]
        }

    rtt_stats = calc_percentiles(rtt_list)
    exec_stats = calc_percentiles(exec_list)

    # Verdicts
    verdicts = {}
    for r in results:
        v = r.get("verdict") or "UNKNOWN"
        verdicts[v] = verdicts.get(v, 0) + 1

    # Intent breakdown
    intents = {}
    for r in results:
        it = r.get("intent", "other")
        if it not in intents:
            intents[it] = {"total": 0, "with_rules": 0, "hits": 0, "recalls": [], "latencies": []}
        intents[it]["total"] += 1
        intents[it]["latencies"].append(r.get("latency_seconds", 0))
        if r.get("expected_rules"):
            intents[it]["with_rules"] += 1
            if r.get("rule_hit"):
                intents[it]["hits"] += 1
            if r.get("rule_recall") is not None:
                intents[it]["recalls"].append(r["rule_recall"])

    intent_breakdown = {}
    for it, d in sorted(intents.items()):
        w = d["with_rules"]
        hr = round((d["hits"] / w) * 100, 1) if w > 0 else 0.0
        ar = round(sum(d["recalls"]) / len(d["recalls"]), 2) if d["recalls"] else 0.0
        al = round(sum(d["latencies"]) / len(d["latencies"]), 2) if d["latencies"] else 0.0
        intent_breakdown[it] = {
            "total": d["total"],
            "with_rules": w,
            "hits": d["hits"],
            "hit_rate": hr,
            "avg_recall": ar,
            "avg_latency": al
        }

    # Errors
    errors = [r for r in results if r.get("error")]

    # Misses (items with expected rules but 0 hits)
    misses = [r for r in eval_with_rules if not r.get("rule_hit")]

    return {
        "total_evaluated": len(results),
        "total_in_slice": total_in_slice,
        "num_with_rules": num_with_rules,
        "total_hits": hits,
        "hit_rate": hit_rate,
        "avg_recall": avg_recall,
        "rtt_stats": rtt_stats,
        "exec_stats": exec_stats,
        "verdicts": verdicts,
        "intent_breakdown": intent_breakdown,
        "error_count": len(errors),
        "misses": misses
    }


def print_scorecard(stats: dict):
    """Display clean terminal scorecard."""
    print("\n" + "=" * 75)
    print("🎯 UP FRONT BGG BENCHMARK SCORECARD — LIVE CLOUD PRODUCTION")
    print("=" * 75)
    print(f"  • Completed Evaluated  : {stats['total_evaluated']} / {stats['total_in_slice']} ({(stats['total_evaluated']/stats['total_in_slice'])*100:.1f}%)")
    print(f"  • Cases with Rule Citations : {stats['num_with_rules']}")
    print(f"  • Errors / Failures    : {stats['error_count']}")
    print("-" * 75)
    print(f"  🏷️  RETRIEVAL CITATION ACCURACY:")
    print(f"      - Rule Hit Rate    : {stats['hit_rate']}% ({stats['total_hits']}/{stats['num_with_rules']})")
    print(f"      - Average Recall   : {stats['avg_recall']}")
    print("-" * 75)
    print(f"  ⏱️  LATENCY PROFILE:")
    rtt = stats['rtt_stats']
    print(f"      - Round-Trip Latency (RTT) : Mean {rtt['mean']}s | P50: {rtt['p50']}s | P90: {rtt['p90']}s | Max: {rtt['max']}s")
    ex = stats['exec_stats']
    print(f"      - Server Execution Time    : Mean {ex['mean']}ms | P50: {ex['p50']}ms | P90: {ex['p90']}ms | Max: {ex['max']}ms")
    print("-" * 75)
    print(f"  ⚖️  VERDICT DISTRIBUTION:")
    for v, cnt in sorted(stats['verdicts'].items(), key=lambda x: -x[1]):
        print(f"      - {v:<14}: {cnt} ({cnt/stats['total_evaluated']*100:.1f}%)")
    print("=" * 75)


def generate_markdown_report(report_path: str, stats: dict, results: list, start_idx: int, end_idx: int, api_url: str):
    """Generate comprehensive executive Markdown report."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    rtt = stats['rtt_stats']
    ex = stats['exec_stats']

    md = []
    md.append(f"# Gaiia RAG Doll Cloud — Up Front 100-Benchmark Scorecard\n")
    md.append(f"**Execution Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    md.append(f"**Production Endpoint**: `{api_url}`  ")
    md.append(f"**Test Cases Evaluated**: #{start_idx} to #{end_idx} (Total: {stats['total_evaluated']})  ")
    md.append(f"**Engine / LLM**: AWS Bedrock Nova Pro + DynamoDB Rule Index\n")

    md.append("## 1. Executive Summary\n")
    md.append("| Metric | Production Cloud Value | Target / Benchmark |")
    md.append("| :--- | :---: | :---: |")
    md.append(f"| **Rule Hit Rate** | **{stats['hit_rate']}%** ({stats['total_hits']}/{stats['num_with_rules']}) | > 85.0% |")
    md.append(f"| **Average Rule Recall** | **{stats['avg_recall']}** | > 0.70 |")
    md.append(f"| **Average RTT Latency** | **{rtt['mean']}s** | < 6.0s |")
    md.append(f"| **P50 RTT Latency** | **{rtt['p50']}s** | < 5.0s |")
    md.append(f"| **P90 RTT Latency** | **{rtt['p90']}s** | < 8.0s |")
    md.append(f"| **Average Server Exec Time** | **{ex['mean']}ms** | < 3500ms |")
    md.append(f"| **HTTP / Pipeline Errors** | **{stats['error_count']}** | 0 |\n")

    md.append("## 2. Latency Profile\n")
    md.append("| Measurement | Mean | P50 (Median) | P90 | P95 | Min | Max |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| **Round-Trip Time (RTT)** | {rtt['mean']}s | {rtt['p50']}s | {rtt['p90']}s | {rtt['p95']}s | {rtt['min']}s | {rtt['max']}s |")
    md.append(f"| **Cloud Lambda Exec Time** | {ex['mean']}ms | {ex['p50']}ms | {ex['p90']}ms | {ex['p95']}ms | {ex['min']}ms | {ex['max']}ms |\n")

    md.append("## 3. Query Intent Breakdown\n")
    md.append("| Intent Category | Total Inquiries | Inquiries w/ Rules | Hit Rate (%) | Avg Recall | Avg RTT Latency |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for it, d in stats['intent_breakdown'].items():
        md.append(f"| `{it}` | {d['total']} | {d['with_rules']} | **{d['hit_rate']}%** | {d['avg_recall']} | {d['avg_latency']}s |")
    md.append("")

    md.append("## 4. Adjudication Verdict Distribution\n")
    md.append("| Verdict | Count | Percentage |")
    md.append("| :--- | :---: | :---: |")
    for v, cnt in sorted(stats['verdicts'].items(), key=lambda x: -x[1]):
        pct = (cnt / stats['total_evaluated']) * 100
        md.append(f"| `{v}` | {cnt} | {pct:.1f}% |")
    md.append("")

    # Section 5: Misses
    md.append("## 5. Retrieval Misses Deep Dive\n")
    if stats['misses']:
        md.append(f"The following **{len(stats['misses'])} questions** had expected rule citations where no strict match was found:\n")
        md.append("| # | ID | Title | Expected Rules | Retrieved Rules | Verdict |")
        md.append("| :---: | :--- | :--- | :--- | :--- | :---: |")
        for m in stats['misses']:
            exp = ", ".join(m.get("expected_rules", []))
            ret = ", ".join(m.get("retrieved_rules", [])[:5]) or "None"
            md.append(f"| {m.get('overall_index')} | `{m['id']}` | {m.get('title', '')[:35]} | `{exp}` | `{ret}` | `{m.get('verdict')}` |")
    else:
        md.append("🎉 **Zero misses!** 100% of questions with expected rules had at least one rule retrieved.\n")
    md.append("")

    # Section 6: Itemized Results
    md.append("## 6. Itemized Test Results (First 100 Cases)\n")
    md.append("| # | ID | Intent | Verdict | Expected Rules | Hits | Recall | Latency | Exec (ms) |")
    md.append("| :---: | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: |")
    for r in sorted(results, key=lambda x: x.get("overall_index", 0)):
        idx = r.get("overall_index")
        rid = r.get("id")
        intent = r.get("intent")
        verdict = r.get("verdict", "UNKNOWN")
        exp = ", ".join(r.get("expected_rules", [])) or "-"
        hits = ", ".join(r.get("hits", [])) or ("-" if not r.get("expected_rules") else "❌")
        rec = f"{r.get('rule_recall'):.2f}" if r.get("rule_recall") is not None else "-"
        lat = f"{r.get('latency_seconds', 0):.2f}s"
        ems = f"{r.get('cloud_exec_ms', 0)}ms" if r.get("cloud_exec_ms") is not None else "-"
        md.append(f"| {idx} | `{rid}` | `{intent}` | `{verdict}` | {exp} | {hits} | {rec} | {lat} | {ems} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n📄 Comprehensive Markdown scorecard exported to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Authoritative Up Front 100 Benchmark Runner for Cloud Production")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help=f"Path to benchmark JSON (default: {DEFAULT_BENCHMARK})")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help=f"Path to output checkpoint JSON (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--report", default=DEFAULT_REPORT, help=f"Path to output report Markdown (default: {DEFAULT_REPORT})")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"Production API Gateway URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--tenant-slug", default="internal-test", help="Tenant slug")
    parser.add_argument("--title-slug", default="up-front", help="Game title slug")
    parser.add_argument("--start", type=int, default=1, help="Start index 1-based (default: 1)")
    parser.add_argument("--end", type=int, default=100, help="End index 1-based (default: 100)")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent workers (default: 2)")
    parser.add_argument("--timeout", type=int, default=40, help="HTTP timeout per query in seconds (default: 40)")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries on transient errors (default: 3)")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and execute from scratch")
    args = parser.parse_args()

    print("=" * 75)
    print("🚀 GAIIA RAG DOLL CLOUD — 100-BENCHMARK TEST ORCHESTRATOR")
    print(f"Target: {args.api_url}")
    print(f"Range : Tests #{args.start} to #{args.end}")
    print(f"Workers: {args.workers} | Timeout: {args.timeout}s | Retries: {args.retries}")
    print("=" * 75)

    if not os.path.exists(args.benchmark):
        raise FileNotFoundError(f"Benchmark file not found at: {args.benchmark}")

    with open(args.benchmark, "r", encoding="utf-8") as f:
        all_items = json.load(f)

    start_0 = max(0, args.start - 1)
    end_0 = min(len(all_items), args.end)
    slice_items = all_items[start_0:end_0]
    total_in_slice = len(slice_items)
    print(f"Total benchmark items in slice: {total_in_slice}")

    # Handle checkpoint
    if args.reset and os.path.exists(args.checkpoint):
        backup_file = args.checkpoint.replace(".json", f"_backup_{int(time.time())}.json")
        os.rename(args.checkpoint, backup_file)
        print(f"📦 Reset requested. Archived previous checkpoint to: {os.path.basename(backup_file)}")

    completed_results = []
    completed_ids = set()

    if os.path.exists(args.checkpoint):
        try:
            with open(args.checkpoint, "r", encoding="utf-8") as f:
                chk = json.load(f)
                completed_results = chk.get("results", [])
                completed_ids = set(r["id"] for r in completed_results)
                print(f"Loaded existing checkpoint with {len(completed_results)} completed items.")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}). Starting fresh.")
            completed_results = []
            completed_ids = set()

    # Determine pending items within the slice
    pending_items = []
    for item in slice_items:
        if item["id"] not in completed_ids:
            overall_idx = all_items.index(item) + 1
            pending_items.append((item, overall_idx))

    print(f"Items remaining to evaluate in slice: {len(pending_items)} / {total_in_slice}")

    if not pending_items:
        print("✅ All items in target range have already been evaluated!")
        slice_completed = [r for r in completed_results if args.start <= r.get("overall_index", 0) <= args.end]
        stats = compute_statistics(slice_completed, total_in_slice)
        print_scorecard(stats)
        generate_markdown_report(args.report, stats, slice_completed, args.start, args.end, args.api_url)
        return

    checkpoint_lock = Lock()
    console_lock = Lock()
    start_time = time.time()
    processed_count = 0

    def task_worker(item, overall_idx):
        nonlocal processed_count
        res = evaluate_single_item(
            item=item,
            overall_idx=overall_idx,
            total_items=len(all_items),
            api_url=args.api_url,
            tenant_slug=args.tenant_slug,
            title_slug=args.title_slug,
            timeout=args.timeout,
            retries=args.retries
        )

        with checkpoint_lock:
            completed_results.append(res)
            completed_ids.add(item["id"])
            save_checkpoint_atomic(args.checkpoint, completed_results, len(all_items))
            processed_count += 1
            curr_done = processed_count
            total_pending = len(pending_items)

        with console_lock:
            exp_rules = res.get("expected_rules", [])
            hit_str = ""
            if exp_rules:
                hit_str = f" | {'✅ HIT' if res.get('rule_hit') else '❌ MISS'} (Rec: {res.get('rule_recall')})"
            print(f"[{curr_done}/{total_pending}] (#{overall_idx}) {res['id']}: {res['title'][:32]} | Lat: {res['latency_seconds']}s | Verdict: {res['verdict']}{hit_str}")

        return res

    print(f"\n🚀 Launching {min(args.workers, len(pending_items))} worker(s)...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(task_worker, item, idx) for item, idx in pending_items]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as exc:
                print(f"Worker exception: {exc}")

    total_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 75)
    print(f"🎉 SUITE EXECUTION FINISHED: {len(pending_items)} items evaluated in {total_time}s ({total_time/len(pending_items):.2f}s/query)")
    print("=" * 75)

    slice_completed = [r for r in completed_results if args.start <= r.get("overall_index", 0) <= args.end]
    stats = compute_statistics(slice_completed, total_in_slice)
    print_scorecard(stats)
    generate_markdown_report(args.report, stats, slice_completed, args.start, args.end, args.api_url)


if __name__ == "__main__":
    main()
