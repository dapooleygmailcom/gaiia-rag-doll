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

BENCHMARK_FILE = "data/eval/asl_bgg_eval_benchmark.json"
CHECKPOINT_FILE = "data/eval/asl_bgg_eval_checkpoint.json"
PROFILE_PATH = "data/asl_profile.json"


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
        matches = re.findall(r'\[(?:Rule|Section):\s*([A-W]?\d+[\.\d\w]*)\]', str(doc))
        for m in matches:
            retrieved_rules.add(m.strip())

    return sorted(list(retrieved_rules))


def check_rule_index_coverage(expected_rules, rule_index):
    """Check which expected rules exist in the rule index. Returns (indexed_rules, missing_rules)."""
    if not expected_rules or not rule_index:
        return list(expected_rules or []), []

    indexed = []
    missing = []
    index_keys = set(k.lower() for k in rule_index.keys() if not k.startswith("__"))

    for r in expected_rules:
        clean_r = r.strip().lower()
        clean_no_ch = clean_r[1:] if clean_r and clean_r[0].isalpha() else clean_r

        found = False
        if clean_r in index_keys:
            found = True
        elif any(k == clean_r or k.endswith(clean_r) or clean_r.endswith(k) or k == clean_no_ch for k in index_keys):
            found = True

        if found:
            indexed.append(r)
        else:
            missing.append(r)

    return indexed, missing


def print_scorecard(completed_results, total_in_suite):
    """Calculate and display sandboxed baseline vs adjudicated benchmark metrics for ASL."""
    active_results = [r for r in completed_results if not r.get("skipped")]
    skipped_results = [r for r in completed_results if r.get("skipped")]

    eval_with_rules = [r for r in active_results if r.get("expected_rules")]
    
    total_evaluated = len(active_results)
    total_skipped = len(skipped_results)
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

    avg_latency = round(sum(r.get("latency_seconds", 0) for r in active_results) / total_evaluated, 2) if total_evaluated > 0 else 0.0

    print("\n" + "=" * 75)
    print("📊 ASL CUMULATIVE BENCHMARK SCORECARD (SANDBOXED EVALUATION)")
    print("=" * 75)
    print(f"  • Total Evaluated: {total_evaluated} / {total_in_suite} ({(total_evaluated/total_in_suite)*100:.1f}%)" if total_in_suite > 0 else f"  • Total Evaluated: {total_evaluated}")
    if total_skipped > 0:
        print(f"  • Skipped (Missing Modules / Unindexed Rules): {total_skipped}")
    print(f"  • Active Questions with Rule Citations: {num_with_citations}")
    print(f"  • Average Latency per Query: {avg_latency}s")
    print("-" * 75)
    print(f"  🏷️  BASELINE CITATION METRICS (Strict Regex Match):")
    print(f"      - Rule Hit Rate : {hit_rate_base}% ({total_hits_base}/{num_with_citations})")
    print(f"      - Average Recall: {avg_recall_base}")
    print("-" * 75)
    print(f"  ⚖️  ADJUDICATED METRICS (Substantive Accuracy & Completeness):")
    print(f"      - Substantive Hit Rate : {hit_rate_adj}% ({total_hits_adj}/{num_with_citations})")
    print(f"      - Substantive Recall   : {avg_recall_adj}")

    if skipped_results:
        print(f"\n  ⚠️  Skipped Cases — Missing Modules ({len(skipped_results)} items):")
        for sk in skipped_results:
            print(f"    • ID: {sk['id']} ({sk.get('title', '')[:38]}) — {sk.get('skip_reason')}")

    adjudicated_items = [r for r in active_results if r.get("adjudication") and r["adjudication"].get("adjudicated")]
    if adjudicated_items:
        print(f"\n  🔍 Adjudicated Cases ({len(adjudicated_items)} items):")
        for adj_item in adjudicated_items:
            adj = adj_item["adjudication"]
            base_r = adj_item.get("rule_recall", 0.0)
            adj_r = adj_item.get("adjudicated_recall", 0.0)
            status = f"✅ ADJUSTED ({base_r} -> {adj_r})" if adj.get("substantive_accuracy") else f"❌ UNCHANGED ({base_r})"
            print(f"    • ID: {adj_item['id']} ({adj_item.get('title', '')[:38]}) -> {status}")
            print(f"      Ruling: {adj.get('ruling')} | Rationale: {adj.get('rationale')}")
    print("=" * 75)


def rescore_checkpoint_adjudication(checkpoint_path=CHECKPOINT_FILE):
    """Retroactively double-check all full misses and partial citation scores in an existing checkpoint."""
    print("=" * 75)
    print("⚖️ RESCORING ASL CHECKPOINT (FULL MISSES & PARTIAL SCORES < 1.0)")
    print("=" * 75)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)

    results = checkpoint_data.get("results", [])
    total_in_suite = checkpoint_data.get("total_in_suite", len(results))
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
            r["adjudicated_hit"] = rule_hit
            r["adjudicated_recall"] = recall

    # Save updated checkpoint
    checkpoint_data["last_rescore_timestamp"] = datetime.now().isoformat()
    checkpoint_data["adjudicator_model"] = "llama3.1:8b"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    print(f"\nSuccessfully rescored {updated_count} items and updated {checkpoint_path}")
    print_scorecard(results, total_in_suite)


def run_benchmark(benchmark_path=BENCHMARK_FILE, checkpoint_path=CHECKPOINT_FILE,
                  profile_path=PROFILE_PATH, sample_size=None, start=1, limit=None,
                  end=None, target_id=None, no_judge=False, dry_run=False):
    """Execute ASL evaluation benchmark."""
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        full_benchmark_items = json.load(f)

    total_in_suite = len(full_benchmark_items)

    if target_id:
        benchmark_items = [b for b in full_benchmark_items if b["id"] == target_id or b.get("thread_id") == target_id]
        if not benchmark_items:
            print(f"Error: Target ID '{target_id}' not found in benchmark.")
            return
        item_indices = [(i + 1, item) for i, item in enumerate(full_benchmark_items) if item["id"] == target_id or item.get("thread_id") == target_id]
    elif sample_size and sample_size < len(full_benchmark_items):
        import random
        random.seed(42)
        sample_indices = sorted(random.sample(range(len(full_benchmark_items)), sample_size))
        item_indices = [(i + 1, full_benchmark_items[i]) for i in sample_indices]
    else:
        start_idx = (start - 1) if start and start > 0 else 0
        end_idx = end if end is not None else limit
        if end_idx is not None:
            selected_slice = full_benchmark_items[start_idx:end_idx]
        else:
            selected_slice = full_benchmark_items[start_idx:]
        item_indices = [(start_idx + 1 + i, item) for i, item in enumerate(selected_slice)]

    print(f"Loaded {len(item_indices)} items to evaluate (Total benchmark size: {total_in_suite}).", flush=True)

    # Load existing checkpoint if available
    completed_results = []
    completed_ids = set()
    if os.path.exists(checkpoint_path) and not target_id:
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                chk = json.load(f)
                completed_results = chk.get("results", [])
                completed_ids = {r["id"] for r in completed_results}
                print(f"Resuming from checkpoint with {len(completed_results)} previously evaluated queries.", flush=True)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}", flush=True)

    # Load rule index for missing module checking
    rule_index = {}
    rule_index_file = "data/asl_rule_index.json"
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as pf:
                prof_data = json.load(pf)
                rule_index_file = prof_data.get("rule_index_file", rule_index_file)
        except Exception:
            pass

    if os.path.exists(rule_index_file):
        try:
            with open(rule_index_file, "r", encoding="utf-8") as rf:
                rule_index = json.load(rf)
                print(f"Loaded rule index with {len(rule_index)} indexed rules for module coverage verification.", flush=True)
        except Exception as e:
            print(f"Warning: Could not load rule index '{rule_index_file}': {e}", flush=True)

    # Process items
    for overall_idx, item in item_indices:
        item_id = item["id"]
        if item_id in completed_ids:
            print(f"\n[{overall_idx}/{total_in_suite}] Skipping (Already in Checkpoint): {item_id} — '{item.get('title', '')[:60]}'", flush=True)
            continue

        title = item.get("title", "")
        query = item.get("query", "")
        expected_rules = item.get("expected_rule_citations", [])
        ground_truth = item.get("ground_truth_answer", "")

        print(f"\n[{overall_idx}/{total_in_suite}] Evaluating: {item_id} — '{title[:60]}'", flush=True)
        print(f"  Expected Rules: {expected_rules}", flush=True)

        # Check for missing module in index
        indexed_rules, missing_rules = check_rule_index_coverage(expected_rules, rule_index)
        is_missing_module = bool(expected_rules and len(indexed_rules) == 0 and rule_index)

        if is_missing_module:
            print(f"  ⚠️  SKIPPED (Missing Module / Unindexed Corpus Rules: {missing_rules})", flush=True)
            res_item = {
                "id": item_id,
                "overall_index": overall_idx,
                "title": title,
                "query": query,
                "expected_rules": expected_rules,
                "indexed_rules": indexed_rules,
                "missing_rules": missing_rules,
                "skipped": True,
                "skip_reason": f"Missing Module / Unindexed Corpus Rules: {missing_rules}",
                "retrieved_rules": [],
                "hits": [],
                "rule_hit": None,
                "rule_recall": None,
                "rule_precision": None,
                "adjudicated_hit": None,
                "adjudicated_recall": None,
                "adjudication": None,
                "generated_answer": "SKIPPED: Target rules not present in indexed corpus.",
                "ground_truth_answer": ground_truth,
                "latency_seconds": 0.0,
                "evaluated_at": datetime.now().isoformat()
            }
            completed_results.append(res_item)
            completed_ids.add(item_id)

            # Save checkpoint atomically
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            temp_chk = f"{checkpoint_path}.tmp"
            with open(temp_chk, "w", encoding="utf-8") as f:
                json.dump({
                    "game": "asl",
                    "last_update": datetime.now().isoformat(),
                    "total_in_suite": total_in_suite,
                    "results": completed_results
                }, f, indent=2)
            os.replace(temp_chk, checkpoint_path)
            continue

        if dry_run:
            print("  [DRY-RUN] Query skipped.", flush=True)
            continue

        t0 = time.time()
        try:
            answer, context_chunks, debug_info = ask_rules_lawyer_game(query, profile_path=profile_path)
            latency = round(time.time() - t0, 2)
        except Exception as e:
            print(f"  Error executing query: {e}", flush=True)
            answer = f"ERROR: {e}"
            context_chunks = []
            debug_info = {}
            latency = round(time.time() - t0, 2)

        retrieved_rules = extract_retrieved_rules(context_chunks)

        # Baseline evaluation
        hits = [r for r in expected_rules if any(r.lower() == ret.lower() or ret.lower().startswith(r.lower()) for ret in retrieved_rules)]
        rule_hit = len(hits) > 0 if expected_rules else None
        rule_recall = round(len(hits) / len(expected_rules), 2) if expected_rules else None
        rule_precision = round(len(hits) / len(retrieved_rules), 2) if retrieved_rules and expected_rules else None

        print(f"  Retrieved Rules: {retrieved_rules[:15]}", flush=True)
        print(f"  Hits: {hits} | Recall: {rule_recall} | Latency: {latency}s", flush=True)

        # LLM Adjudication if needed
        adjudication = None
        adjudicated_hit = rule_hit
        adjudicated_recall = rule_recall

        if not no_judge and expected_rules and (rule_hit is False or (rule_recall is not None and rule_recall < 1.0)):
            print(f"  Running Adjudicator (Ground Truth Comparison)...", flush=True)
            adjudication = adjudicate_answer(
                query=query,
                ground_truth=ground_truth,
                candidate_answer=answer,
                judge_model="llama3.1:8b"
            )
            if adjudication.get("substantive_accuracy"):
                adjudicated_hit = True
                adjudicated_recall = 1.0
                print(f"  -> Adjudicator: ✅ {adjudication.get('ruling')} (Adjusted Recall: {rule_recall} -> 1.0)", flush=True)
            else:
                print(f"  -> Adjudicator: ❌ {adjudication.get('ruling')} (Recall: {rule_recall})", flush=True)

        res_item = {
            "id": item_id,
            "overall_index": overall_idx,
            "title": title,
            "query": query,
            "expected_rules": expected_rules,
            "retrieved_rules": retrieved_rules,
            "hits": hits,
            "rule_hit": rule_hit,
            "rule_recall": rule_recall,
            "rule_precision": rule_precision,
            "adjudicated_hit": adjudicated_hit,
            "adjudicated_recall": adjudicated_recall,
            "adjudication": adjudication,
            "generated_answer": answer,
            "ground_truth_answer": ground_truth,
            "latency_seconds": latency,
            "evaluated_at": datetime.now().isoformat()
        }
        completed_results.append(res_item)
        completed_ids.add(item_id)

        # Save checkpoint atomically after each item
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        temp_chk = f"{checkpoint_path}.tmp"
        with open(temp_chk, "w", encoding="utf-8") as f:
            json.dump({
                "game": "asl",
                "last_update": datetime.now().isoformat(),
                "total_in_suite": total_in_suite,
                "results": completed_results
            }, f, indent=2)
        os.replace(temp_chk, checkpoint_path)

    if completed_results:
        print_scorecard(completed_results, total_in_suite)


def main():
    parser = argparse.ArgumentParser(description="ASL BGG Rules Evaluation Benchmark Runner")
    parser.add_argument("--benchmark", default=BENCHMARK_FILE, help="Path to benchmark JSON")
    parser.add_argument("--checkpoint", default=CHECKPOINT_FILE, help="Path to checkpoint JSON")
    parser.add_argument("--profile", default=PROFILE_PATH, help="Path to ASL game profile JSON")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate random N queries")
    parser.add_argument("--start", type=int, default=1, help="1-indexed starting query index (default: 1)")
    parser.add_argument("--end", type=int, default=None, help="1-indexed ending query index inclusive (e.g. --end 10)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N queries or slice end")
    parser.add_argument("--id", type=str, default=None, help="Target a specific question ID")
    parser.add_argument("--no-judge", action="store_true", help="Skip secondary LLM adjudication")
    parser.add_argument("--rescore", action="store_true", help="Rescore existing checkpoint with Adjudicator")
    parser.add_argument("--summary", action="store_true", help="Print scorecard summary from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Preview benchmark queries without executing")

    args = parser.parse_args()

    if args.summary:
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            return
        with open(args.checkpoint, "r", encoding="utf-8") as f:
            chk = json.load(f)
        print_scorecard(chk.get("results", []), chk.get("total_in_suite", len(chk.get("results", []))))
        return

    if args.rescore:
        rescore_checkpoint_adjudication(args.checkpoint)
        return

    run_benchmark(
        benchmark_path=args.benchmark,
        checkpoint_path=args.checkpoint,
        profile_path=args.profile,
        sample_size=args.sample,
        start=args.start,
        end=args.end,
        limit=args.limit,
        target_id=args.id,
        no_judge=args.no_judge,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
