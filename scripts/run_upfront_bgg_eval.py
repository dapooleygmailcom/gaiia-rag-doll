import os
import sys
import json
import time
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from engine.retrieval.rules_lawyer import ask_rules_lawyer_game, load_game_profile
from engine.evaluation.adjudicator import adjudicate_answer

BENCHMARK_FILE = "data/eval/upfront_bgg_eval_benchmark.json"
CHECKPOINT_FILE = "data/eval/upfront_bgg_eval_checkpoint.json"
CLOUD_CHECKPOINT_FILE = "data/eval/upfront_cloud_eval_checkpoint.json"
PROFILE_PATH = "data/up_front_profile.json"
DEFAULT_CLOUD_API_URL = "https://8ifjmds7mk.execute-api.ap-southeast-2.amazonaws.com/api/ask"


def query_cloud_api(query: str, api_url: str = DEFAULT_CLOUD_API_URL, tenant_slug: str = "internal-test", title_slug: str = "up-front", timeout: int = 35):
    """Execute evaluation query against deployed AWS API Gateway / Bedrock Nova Pro endpoint."""
    import urllib.request
    import urllib.error
    import re

    payload = {
        "tenantSlug": tenant_slug,
        "titleSlug": title_slug,
        "query": query,
        "captchaToken": "test-eval-suite-token"
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST"
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
            rtt = round(time.time() - t0, 2)
            answer = data.get("answerMarkdown", "") or data.get("answerText", "")
            citations = data.get("citations", [])
            retrieved_rules = []
            for c in citations:
                rn = c.get("ruleNumber")
                if rn:
                    retrieved_rules.append(str(rn).strip())

            # Extract citations from answer text like [12.4] or [UP FRONT 12.4] or [Rule 12.4]
            text_citations = re.findall(r'\[(?:UP FRONT|Rule|Section)?\s*([\d\.]+)\]', answer, re.IGNORECASE)
            for tc in text_citations:
                if tc.strip() not in retrieved_rules:
                    retrieved_rules.append(tc.strip())

            debug_info = {
                "executionTimeMs": data.get("executionTimeMs"),
                "verdict": data.get("verdict"),
                "confidenceScore": data.get("confidenceScore"),
                "roundTripTimeSec": rtt,
                "api_url": api_url,
                "num_citations": len(citations)
            }
            return answer, citations, sorted(list(set(retrieved_rules))), debug_info, None
    except Exception as e:
        rtt = round(time.time() - t0, 2)
        return f"ERROR: {e}", [], [], {"roundTripTimeSec": rtt, "api_url": api_url}, str(e)


def extract_retrieved_rules(context_chunks):
    """Extract unique rule numbers from retrieved context metadata and text headers."""
    retrieved_rules = set()
    for item in context_chunks:
        if isinstance(item, tuple) and len(item) == 2:
            doc, meta = item
        elif isinstance(item, dict):
            meta = item
            doc = meta.get("text", "")
        else:
            continue

        rn = meta.get("rule_number") or meta.get("rule_id")
        if rn:
            retrieved_rules.add(str(rn).strip())

        # Also check chunk text header for [Rule: XX.XX] or [Section: XX.XX]
        import re
        matches = re.findall(r'\[(?:Rule|Section):\s*([\d\.]+)\]', str(doc))
        for m in matches:
            retrieved_rules.add(m.strip())

    return sorted(list(retrieved_rules))


def print_scorecard(completed_results, total_in_suite):
    """Calculate and display sandboxed baseline vs adjudicated benchmark metrics."""
    eval_with_rules = [r for r in completed_results if r.get("expected_rules")]
    
    total_evaluated = len(completed_results)
    num_with_citations = len(eval_with_rules)
    
    if eval_with_rules:
        # Baseline Citation Metrics
        total_hits_base = sum(1 for r in eval_with_rules if r.get("rule_hit"))
        hit_rate_base = round((total_hits_base / num_with_citations) * 100, 1)
        valid_recalls_base = [r["rule_recall"] for r in eval_with_rules if r.get("rule_recall") is not None]
        avg_recall_base = round(sum(valid_recalls_base) / len(valid_recalls_base), 2) if valid_recalls_base else 0.0

        # Adjudicated Substantive Metrics (Sandboxed)
        total_hits_adj = sum(1 for r in eval_with_rules if r.get("adjudicated_hit", r.get("rule_hit")))
        hit_rate_adj = round((total_hits_adj / num_with_citations) * 100, 1)
        valid_recalls_adj = [r.get("adjudicated_recall", r.get("rule_recall")) for r in eval_with_rules if (r.get("adjudicated_recall") or r.get("rule_recall")) is not None]
        avg_recall_adj = round(sum(valid_recalls_adj) / len(valid_recalls_adj), 2) if valid_recalls_adj else 0.0
    else:
        total_hits_base, hit_rate_base, avg_recall_base = 0, 0.0, 0.0
        total_hits_adj, hit_rate_adj, avg_recall_adj = 0, 0.0, 0.0

    avg_latency = round(sum(r.get("latency_seconds", 0) for r in completed_results) / total_evaluated, 2) if total_evaluated > 0 else 0.0

    print("\n" + "=" * 70)
    print("📊 CUMULATIVE BENCHMARK SCORECARD (SANDBOXED EVALUATION)")
    print("=" * 70)
    print(f"  • Total Evaluated: {total_evaluated} / {total_in_suite} ({(total_evaluated/total_in_suite)*100:.1f}%)")
    print(f"  • Questions with Rule Citations: {num_with_citations}")
    print(f"  • Average Latency per Query: {avg_latency}s")
    print("-" * 70)
    print(f"  🏷️  BASELINE CITATION METRICS (Strict Regex Match):")
    print(f"      - Rule Hit Rate : {hit_rate_base}% ({total_hits_base}/{num_with_citations})")
    print(f"      - Average Recall: {avg_recall_base}")
    print("-" * 70)
    print(f"  ⚖️  ADJUDICATED METRICS (Substantive Accuracy & Completeness):")
    print(f"      - Substantive Hit Rate : {hit_rate_adj}% ({total_hits_adj}/{num_with_citations})")
    print(f"      - Substantive Recall   : {avg_recall_adj}")
    
    adjudicated_items = [r for r in completed_results if r.get("adjudication") and r["adjudication"].get("adjudicated")]
    if adjudicated_items:
        print(f"\n  🔍 Adjudicated Cases ({len(adjudicated_items)} items):")
        for adj_item in adjudicated_items:
            adj = adj_item["adjudication"]
            base_r = adj_item.get("rule_recall", 0.0)
            adj_r = adj_item.get("adjudicated_recall", 0.0)
            status = f"✅ ADJUSTED ({base_r} -> {adj_r})" if adj.get("substantive_accuracy") else f"❌ UNCHANGED ({base_r})"
            print(f"    • ID: {adj_item['id']} ({adj_item.get('title', '')[:38]}) -> {status}")
            print(f"      Ruling: {adj.get('ruling')} | Rationale: {adj.get('rationale')}")
    print("=" * 70)


def rescore_checkpoint_adjudication(checkpoint_path=CHECKPOINT_FILE):
    """Retroactively double-check all full misses and partial citation scores in an existing checkpoint."""
    print("=" * 70)
    print("⚖️ RESCORING CHECKPOINT (FULL MISSES & PARTIAL SCORES < 1.0)")
    print("=" * 70)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)

    results = checkpoint_data.get("results", [])
    total_in_suite = checkpoint_data.get("total_in_suite", 404)
    print(f"Loaded checkpoint with {len(results)} evaluated items.")

    updated_count = 0
    for r in results:
        expected_rules = r.get("expected_rules", [])
        rule_hit = r.get("rule_hit")
        recall = r.get("rule_recall")

        # Double check full misses and partial scores (recall < 1.0)
        if expected_rules and (rule_hit is False or (recall is not None and recall < 1.0)):
            print(f"\nAdjudicating: ID {r['id']} — '{r.get('title')}'")
            print(f"  Expected Rules: {expected_rules} | Hits: {r.get('hits')} (Base Recall: {recall})")
            
            adj = adjudicate_answer(
                query=r.get("query", ""),
                ground_truth=r.get("ground_truth_answer", ""),
                candidate_answer=r.get("generated_answer", ""),
                judge_model="llama3.1:8b"
            )
            
            r["adjudication"] = adj
            if adj.get("substantive_accuracy"):
                r["adjudicated_hit"] = True
                r["adjudicated_recall"] = 1.0
                print(f"  -> Adjudicator: ✅ {adj.get('ruling')} (Recall Adjusted: {recall} -> 1.0)")
                print(f"     Rationale: {adj.get('rationale')}")
            else:
                r["adjudicated_hit"] = rule_hit
                r["adjudicated_recall"] = recall
                print(f"  -> Adjudicator: ❌ {adj.get('ruling')} (Recall Unchanged: {recall})")
                print(f"     Rationale: {adj.get('rationale')}")
            updated_count += 1
        else:
            # Pass through original metrics if already 1.0 or N/A
            if "adjudicated_hit" not in r:
                r["adjudicated_hit"] = r.get("rule_hit")
            if "adjudicated_recall" not in r:
                r["adjudicated_recall"] = r.get("rule_recall")

    checkpoint_data["results"] = results
    checkpoint_data["last_updated"] = datetime.now().isoformat()

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    print(f"\nSuccessfully rescored and updated {updated_count} items in checkpoint.")
    print_scorecard(results, total_in_suite)
    return results


def run_evaluation_batch(benchmark_path=BENCHMARK_FILE,
                         checkpoint_path=None,
                         profile_path=PROFILE_PATH,
                         batch_size=None,
                         max_items=None,
                         start_index=None,
                         end_index=None,
                         target="local",
                         api_url=DEFAULT_CLOUD_API_URL,
                         tenant_slug="internal-test",
                         title_slug="up-front",
                         skip_judge=False,
                         judge_model="llama3.1:8b"):
    print("=" * 70)
    print(f"🎯 UP FRONT BGG RULES EVALUATION RUNNER (TARGET: {target.upper()})")
    print("=" * 70)

    if checkpoint_path is None:
        checkpoint_path = CLOUD_CHECKPOINT_FILE if target == "cloud" else CHECKPOINT_FILE

    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found at: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_items = json.load(f)

    total_in_suite = len(benchmark_items)
    print(f"Total benchmark items available: {total_in_suite}")

    # Only load local profile if target is local
    if target == "local":
        print(f"Loading game profile: {profile_path}...")
        load_game_profile(profile_path)
    else:
        print(f"Targeting Cloud Production Endpoint: {api_url}")
        print(f"Tenant: {tenant_slug} | Title: {title_slug}")

    # Load or initialize checkpoint
    completed_results = []
    completed_ids = set()

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                completed_results = checkpoint_data.get("results", [])
                completed_ids = set(r["id"] for r in completed_results)
                print(f"Loaded existing checkpoint with {len(completed_results)} completed items ({checkpoint_path}).")
        except Exception as e:
            print(f"Warning: Could not read existing checkpoint ({e}). Starting fresh.")
            completed_results = []
            completed_ids = set()

    # Determine candidate items based on start/end range if provided
    if start_index is not None and end_index is not None:
        start_0 = max(0, start_index - 1)
        end_0 = min(total_in_suite, end_index)
        candidate_items = benchmark_items[start_0:end_0]
        print(f"🎯 Evaluating specified range: Tests #{start_index} to #{end_index} ({len(candidate_items)} items in slice)")
    else:
        candidate_items = benchmark_items

    # Filter pending items
    pending_items = [item for item in candidate_items if item["id"] not in completed_ids]
    print(f"Pending items remaining to run: {len(pending_items)}")

    if not pending_items:
        print("All candidate benchmark items in target range have already been evaluated!")
        print_scorecard(completed_results, total_in_suite)
        return completed_results

    # Slice batch to evaluate
    items_to_run = pending_items
    if batch_size is not None:
        items_to_run = items_to_run[:batch_size]
    if max_items is not None:
        items_to_run = items_to_run[:max_items]

    print(f"Executing batch of {len(items_to_run)} items against [{target.upper()}]...")
    print("=" * 70)

    batch_start_time = time.time()
    batch_results = []

    for idx, item in enumerate(items_to_run, 1):
        item_id = item["id"]
        title = item["title"]
        query = item["query"]
        intent = item.get("intent", "clarification")
        expected_rules = item.get("expected_rule_citations", [])
        gt_answer = item.get("ground_truth_answer", "")

        # Find overall 1-based index in benchmark_items
        try:
            overall_idx = benchmark_items.index(item) + 1
        except ValueError:
            overall_idx = len(completed_results) + 1

        print(f"\n[{overall_idx}/{total_in_suite}] (Batch {idx}/{len(items_to_run)}) ID: {item_id}")
        print(f"  Title: {title}")
        print(f"  Intent: {intent} | Expected Rules: {expected_rules if expected_rules else 'None specified'}")

        item_start = time.time()
        if target == "cloud":
            answer, context_chunks, retrieved_rules, debug_info, err = query_cloud_api(
                query=query,
                api_url=api_url,
                tenant_slug=tenant_slug,
                title_slug=title_slug
            )
        else:
            try:
                answer, context_chunks, debug_info = ask_rules_lawyer_game(query, profile_path=profile_path)
                retrieved_rules = extract_retrieved_rules(context_chunks)
                err = None
            except Exception as e:
                print(f"  ❌ Execution error: {e}")
                answer = f"ERROR: {e}"
                context_chunks = []
                debug_info = {}
                retrieved_rules = []
                err = str(e)

        latency = round(time.time() - item_start, 2)

        # Calculate baseline retrieval metrics
        if expected_rules:
            hits = [r for r in expected_rules if any(r == ret or r.startswith(ret) or ret.startswith(r) for ret in retrieved_rules)]
            rule_hit = len(hits) > 0
            recall = round(len(hits) / len(expected_rules), 2)
        else:
            hits = []
            rule_hit = None
            recall = None

        print(f"  ⏱️ Latency: {latency}s | Extracted Rules: {retrieved_rules[:8]}")
        if debug_info.get("executionTimeMs"):
            print(f"  ⚡ Cloud Execution Time: {debug_info['executionTimeMs']}ms | Verdict: {debug_info.get('verdict')}")
        if expected_rules:
            status_icon = "✅ HIT" if rule_hit else "❌ MISS"
            print(f"  Retrieval: {status_icon} (Hits: {hits}/{expected_rules} -> Recall: {recall})")
        print(f"  Generated Answer Preview: {answer[:160].replace(chr(10), ' ')}...")

        # Secondary Referee Adjudication on full citation misses or partial matches (recall < 1.0)
        adjudication_info = None
        adjudicated_hit = rule_hit
        adjudicated_recall = recall

        if expected_rules and (not rule_hit or (recall is not None and recall < 1.0)):
            if skip_judge:
                adjudication_info = {"status": "skipped"}
            else:
                try:
                    print(f"  ⚖️ Triggering Secondary Referee Adjudication (Base Recall: {recall})...")
                    adj = adjudicate_answer(
                        query=query,
                        ground_truth=gt_answer,
                        candidate_answer=answer,
                        judge_model=judge_model
                    )
                    adjudication_info = adj
                    if adj.get("substantive_accuracy"):
                        adjudicated_hit = True
                        adjudicated_recall = 1.0
                        print(f"  ⚖️ Referee Verdict: ✅ {adj.get('ruling')} (Substantively Accurate: True | Conf: {adj.get('confidence')})")
                        print(f"     Rationale: {adj.get('rationale')}")
                    else:
                        adjudicated_hit = rule_hit
                        adjudicated_recall = recall
                        print(f"  ⚖️ Referee Verdict: ❌ {adj.get('ruling')} (Substantively Accurate: False | Conf: {adj.get('confidence')})")
                        print(f"     Rationale: {adj.get('rationale')}")
                except Exception as judge_err:
                    print(f"  ⚠️ Referee adjudication unavailable ({judge_err}). Proceeding without judge.")
                    adjudication_info = {"error": str(judge_err)}

        # Build evaluation record with sandboxed metrics
        eval_record = {
            "id": item_id,
            "overall_index": overall_idx,
            "target": target,
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "query": query,
            "intent": intent,
            "expected_rules": expected_rules,
            "retrieved_rules": retrieved_rules,
            "rule_hit": rule_hit,
            "hits": hits,
            "rule_recall": recall,
            "adjudication": adjudication_info,
            "adjudicated_hit": adjudicated_hit,
            "adjudicated_recall": adjudicated_recall,
            "latency_seconds": latency,
            "generated_answer": answer,
            "ground_truth_answer": gt_answer,
            "debug": debug_info,
            "error": err
        }

        completed_results.append(eval_record)
        batch_results.append(eval_record)
        completed_ids.add(item_id)

        # Save checkpoint atomically after each item
        checkpoint_payload = {
            "last_updated": datetime.now().isoformat(),
            "target": target,
            "total_evaluated": len(completed_results),
            "total_in_suite": total_in_suite,
            "results": completed_results
        }

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        temp_chk = f"{checkpoint_path}.tmp"
        with open(temp_chk, "w", encoding="utf-8") as f:
            json.dump(checkpoint_payload, f, indent=2)
        os.replace(temp_chk, checkpoint_path)

    batch_total_time = round(time.time() - batch_start_time, 2)
    print("\n" + "=" * 70)
    print(f"BATCH EXECUTION COMPLETE ({len(batch_results)} items in {batch_total_time}s)")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print("=" * 70)

    print_scorecard(completed_results, total_in_suite)
    return completed_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Up Front BGG RAG Evaluation Suite (Local or Cloud)")
    parser.add_argument("--target", choices=["local", "cloud"], default="local", help="Execution target: 'local' engine or 'cloud' production API (default: local)")
    parser.add_argument("--api-url", default=DEFAULT_CLOUD_API_URL, help=f"Cloud API endpoint for /api/ask (default: {DEFAULT_CLOUD_API_URL})")
    parser.add_argument("--start-index", type=int, default=None, help="Start index (1-based), e.g. 61")
    parser.add_argument("--end-index", type=int, default=None, help="End index (1-based), e.g. 100")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of test items to execute in this batch")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum items to execute before stopping")
    parser.add_argument("--checkpoint", default=None, help="Custom checkpoint file path")
    parser.add_argument("--skip-judge", action="store_true", help="Skip secondary referee adjudication")
    parser.add_argument("--judge-model", default="llama3.1:8b", help="Judge model name for secondary adjudication")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and re-run from item 1")
    parser.add_argument("--rescore-checkpoint", action="store_true", help="Retroactively adjudicate all misses/partial scores in existing checkpoint and rescore")
    args = parser.parse_args()

    chk_file = args.checkpoint or (CLOUD_CHECKPOINT_FILE if args.target == "cloud" else CHECKPOINT_FILE)

    if args.reset and os.path.exists(chk_file):
        backup_file = f"{chk_file}.bak_{int(time.time())}"
        import shutil
        shutil.copyfile(chk_file, backup_file)
        os.remove(chk_file)
        print(f"Reset checkpoint file. Previous checkpoint backed up to: {backup_file}")

    if args.rescore_checkpoint:
        rescore_checkpoint_adjudication(chk_file)
    else:
        run_evaluation_batch(
            checkpoint_path=chk_file,
            batch_size=args.batch_size,
            max_items=args.max_items,
            start_index=args.start_index,
            end_index=args.end_index,
            target=args.target,
            api_url=args.api_url,
            skip_judge=args.skip_judge,
            judge_model=args.judge_model
        )
