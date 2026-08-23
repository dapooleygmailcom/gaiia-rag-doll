import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from engine.retrieval.rules_lawyer import ask_rules_lawyer_game, load_game_profile
from scripts.run_upfront_bgg_eval import extract_retrieved_rules

BENCHMARK_FILE = "data/eval/upfront_bgg_eval_benchmark.json"
CHECKPOINT_FILE = "data/eval/upfront_bgg_eval_checkpoint.json"
PROFILE_PATH = "data/up_front_profile.json"

TARGET_IDS = [
    "bgg_uf_3645496",  # Item 5 (AFV Stun & Bogging)
    "bgg_uf_3623562",  # Item 8 (Infiltration after CC)
    "bgg_uf_3623559",  # Item 9 (Killed SL & Hand Size)
    "bgg_uf_3610186",  # Item 12 (Desert Flanking Fire)
    "bgg_uf_3573599",  # Item 15 (LMG Crew Reassignment)
    "bgg_uf_3479566",  # Item 20 (Lateral Group Transfer Terrain)
]

PREVIOUS_SCORES = {
    "bgg_uf_3645496": 0.86,
    "bgg_uf_3623562": 0.50,
    "bgg_uf_3623559": 0.67,
    "bgg_uf_3610186": 0.60,
    "bgg_uf_3573599": 0.80,
    "bgg_uf_3479566": 0.33,
}

def run_targeted_eval():
    print("=" * 70)
    print("🎯 TARGETED RETEST OF LOW-SCORING BENCHMARK ITEMS")
    print("=" * 70)
    
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        all_items = json.load(f)
        
    items_by_id = {item["id"]: item for item in all_items}
    targets = [items_by_id[tid] for tid in TARGET_IDS if tid in items_by_id]
    
    print(f"Loaded {len(targets)} target items to re-evaluate.")
    print(f"Loading game profile: {PROFILE_PATH}...")
    load_game_profile(PROFILE_PATH)
    
    # Load existing checkpoint to update it
    checkpoint_data = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
            
    existing_results = {r["id"]: r for r in checkpoint_data.get("results", [])}
    
    retest_summary = []
    
    for i, item in enumerate(targets, 1):
        item_id = item["id"]
        title = item.get("title", "")
        query = item.get("query", "")
        expected = item.get("expected_rules", [])
        prev_recall = PREVIOUS_SCORES.get(item_id, 0.0)
        
        print(f"\n[{i}/{len(targets)}] Retesting ID: {item_id}")
        print(f"  Title: {title}")
        print(f"  Expected Rules: {expected}")
        print(f"  Previous Recall: {prev_recall * 100:.1f}%")
        
        start_t = time.time()
        try:
            answer, retrieved_chunks, debug_info = ask_rules_lawyer_game(query)
            latency = round(time.time() - start_t, 2)
            retrieved_rules = extract_retrieved_rules(retrieved_chunks)
            
            hits = []
            recall = 0.0
            if expected:
                for exp_r in expected:
                    if exp_r in retrieved_rules or any(r.startswith(exp_r) or exp_r.startswith(r) for r in retrieved_rules):
                        hits.append(exp_r)
                recall = round(len(hits) / len(expected), 2)
                
            print(f"  Retrieved Rules ({len(retrieved_rules)}): {retrieved_rules[:15]}")
            print(f"  Hits: {hits}/{expected}")
            print(f"  New Recall: {recall * 100:.1f}% (Previous: {prev_recall * 100:.1f}%) | Latency: {latency}s")
            
            # Update checkpoint entry
            updated_entry = {
                "id": item_id,
                "overall_index": existing_results.get(item_id, {}).get("overall_index", i),
                "timestamp": datetime.now().isoformat(),
                "title": title,
                "query": query,
                "intent": item.get("intent", "clarification"),
                "expected_rules": expected,
                "retrieved_rules": retrieved_rules,
                "rule_hit": len(hits) > 0 if expected else None,
                "hits": hits,
                "rule_recall": recall if expected else None,
                "latency_seconds": latency,
                "generated_answer": answer,
                "ground_truth_answer": item.get("ground_truth_answer", ""),
                "debug": debug_info,
                "error": None
            }
            existing_results[item_id] = updated_entry
            
            retest_summary.append({
                "id": item_id,
                "title": title,
                "expected": expected,
                "hits": hits,
                "prev_recall": prev_recall,
                "new_recall": recall,
                "latency": latency
            })
            
        except Exception as e:
            print(f"  ❌ Retest error: {e}")
            
    # Save updated checkpoint
    checkpoint_data["results"] = list(existing_results.values())
    checkpoint_data["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
        
    print("\n" + "=" * 70)
    print("🎯 RETEST SUMMARY & RECALL GAINS")
    print("=" * 70)
    for s in retest_summary:
        gain = s['new_recall'] - s['prev_recall']
        gain_str = f"+{gain*100:.1f}%" if gain >= 0 else f"{gain*100:.1f}%"
        print(f"• {s['id']} ({s['title'][:40]}...):")
        print(f"    Expected: {s['expected']}")
        print(f"    Hits:     {s['hits']}")
        print(f"    Recall:   {s['prev_recall']*100:.1f}% ➔ {s['new_recall']*100:.1f}% ({gain_str})")
        
    # Calculate new cumulative recall for the whole 20-item set
    all_citation_items = [r for r in existing_results.values() if r.get("rule_recall") is not None]
    if all_citation_items:
        avg_recall = sum(r["rule_recall"] for r in all_citation_items) / len(all_citation_items)
        print("=" * 70)
        print(f"CUMULATIVE BENCHMARK RECALL ACROSS ALL 20 ITEMS: {avg_recall*100:.1f}% ({len(all_citation_items)} questions)")
        print("=" * 70)

if __name__ == "__main__":
    run_targeted_eval()
