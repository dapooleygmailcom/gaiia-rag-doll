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

BENCHMARK_FILE = "data/eval/upfront_bgg_eval_benchmark.json"
CHECKPOINT_FILE = "data/eval/upfront_bgg_eval_checkpoint.json"
PROFILE_PATH = "data/up_front_profile.json"

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

def run_evaluation_batch(benchmark_path=BENCHMARK_FILE,
                         checkpoint_path=CHECKPOINT_FILE,
                         profile_path=PROFILE_PATH,
                         batch_size=10,
                         max_items=10):
    print("=" * 70)
    print("🎯 UP FRONT BGG RULES EVALUATION RUNNER (RAG-DOLL)")
    print("=" * 70)
    
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found at: {benchmark_path}")
        
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_items = json.load(f)
        
    total_in_suite = len(benchmark_items)
    print(f"Total benchmark items available: {total_in_suite}")
    
    # Load profile into Rules Lawyer
    print(f"Loading game profile: {profile_path}...")
    load_game_profile(profile_path)
    
    # Load or initialize checkpoint
    completed_results = []
    completed_ids = set()
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                completed_results = checkpoint_data.get("results", [])
                completed_ids = set(r["id"] for r in completed_results)
                print(f"Loaded existing checkpoint with {len(completed_results)} completed items.")
        except Exception as e:
            print(f"Warning: Could not read existing checkpoint ({e}). Starting fresh.")
            completed_results = []
            completed_ids = set()
    
    # Filter pending items
    pending_items = [item for item in benchmark_items if item["id"] not in completed_ids]
    print(f"Pending items remaining: {len(pending_items)}")
    
    if not pending_items:
        print("All benchmark items have already been evaluated!")
        return completed_results
        
    # Slice batch to evaluate
    items_to_run = pending_items[:batch_size]
    if max_items:
        items_to_run = items_to_run[:max_items]
        
    print(f"Executing batch of {len(items_to_run)} items (stopping after this batch)...")
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
        
        overall_idx = len(completed_results) + 1
        print(f"\n[{overall_idx}/{total_in_suite}] (Batch {idx}/{len(items_to_run)}) ID: {item_id}")
        print(f"  Title: {title}")
        print(f"  Intent: {intent} | Expected Rules: {expected_rules if expected_rules else 'None specified'}")
        
        item_start = time.time()
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
        
        # Calculate retrieval metrics
        if expected_rules:
            hits = [r for r in expected_rules if any(r == ret or r.startswith(ret) or ret.startswith(r) for ret in retrieved_rules)]
            rule_hit = len(hits) > 0
            recall = round(len(hits) / len(expected_rules), 2)
        else:
            hits = []
            rule_hit = None # Not applicable if no specific rule was cited
            recall = None
            
        print(f"  ⏱️ Latency: {latency}s | Retrieved Chunks: {len(context_chunks)} | Extracted Rules: {retrieved_rules[:8]}")
        if expected_rules:
            status_icon = "✅ HIT" if rule_hit else "❌ MISS"
            print(f"  Retrieval: {status_icon} (Hits: {hits}/{expected_rules} -> Recall: {recall})")
        print(f"  Generated Answer Preview: {answer[:160].replace(chr(10), ' ')}...")
        
        # Build evaluation record
        eval_record = {
            "id": item_id,
            "overall_index": overall_idx,
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "query": query,
            "intent": intent,
            "expected_rules": expected_rules,
            "retrieved_rules": retrieved_rules,
            "rule_hit": rule_hit,
            "hits": hits,
            "rule_recall": recall,
            "latency_seconds": latency,
            "generated_answer": answer,
            "ground_truth_answer": gt_answer,
            "debug": {
                "distilled_question": debug_info.get("distilled_question"),
                "hyde_clause": debug_info.get("hyde_clause"),
                "query_type": debug_info.get("query_type"),
                "rule_numbers": debug_info.get("rule_numbers"),
                "sub_queries": debug_info.get("sub_queries"),
                "num_retrieved": debug_info.get("num_retrieved", len(context_chunks)),
                "num_parent_expansions": debug_info.get("num_parent_expansions", 0),
                "num_cooccurrence_expansions": debug_info.get("num_cooccurrence_expansions", 0),
                "num_cross_refs": debug_info.get("num_cross_refs", 0)
            },
            "error": err
        }
        
        completed_results.append(eval_record)
        batch_results.append(eval_record)
        completed_ids.add(item_id)
        
        # Save checkpoint atomically after each item
        checkpoint_payload = {
            "last_updated": datetime.now().isoformat(),
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
    
    # Calculate batch summary metrics
    eval_with_rules = [r for r in completed_results if r["expected_rules"]]
    if eval_with_rules:
        total_hits = sum(1 for r in eval_with_rules if r["rule_hit"])
        hit_rate = round((total_hits / len(eval_with_rules)) * 100, 1)
        avg_recall = round(sum(r["rule_recall"] for r in eval_with_rules if r["rule_recall"] is not None) / len(eval_with_rules), 2)
    else:
        total_hits = 0
        hit_rate = 0
        avg_recall = 0
        
    avg_latency = round(sum(r["latency_seconds"] for r in completed_results) / len(completed_results), 2)
    
    print("\nCUMULATIVE BENCHMARK PROGRESS:")
    print(f"  • Total Evaluated: {len(completed_results)} / {total_in_suite} ({(len(completed_results)/total_in_suite)*100:.1f}%)")
    print(f"  • Questions with Rule Citations: {len(eval_with_rules)}")
    print(f"  • Rule Retrieval Hit Rate: {hit_rate}% ({total_hits}/{len(eval_with_rules)})")
    print(f"  • Average Rule Recall: {avg_recall}")
    print(f"  • Average Latency per Query: {avg_latency}s")
    print("=" * 70)
    
    return completed_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Up Front BGG RAG Evaluation Suite in Chunks")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of test items to execute in this batch (default: 10)")
    parser.add_argument("--max-items", type=int, default=10, help="Maximum items to execute before stopping (default: 10)")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and re-run from item 1")
    args = parser.parse_args()

    if args.reset and os.path.exists(CHECKPOINT_FILE):
        backup_file = f"{CHECKPOINT_FILE}.bak_{int(time.time())}"
        import shutil
        shutil.copyfile(CHECKPOINT_FILE, backup_file)
        os.remove(CHECKPOINT_FILE)
        print(f"Reset checkpoint file. Previous checkpoint backed up to: {backup_file}")
    
    run_evaluation_batch(batch_size=args.batch_size, max_items=args.max_items)
