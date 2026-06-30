import os
import json
import time
from test_query_asl import ASL_TEST_CASES
import engine.retrieval.rules_lawyer

os.makedirs("data/logs", exist_ok=True)
REPORT_FILE = "data/logs/asl_query_benchmark.md"

def run_suite():
    print("Loading ASL Profile...")
    rules_lawyer.load_game_profile('data/asl_profile.json')
    
    cases_to_run = ASL_TEST_CASES
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# ASL Query Benchmark Results\n\n")
        
    print(f"Running {len(cases_to_run)} benchmark queries...\n")
    
    for tc in cases_to_run:
        start_t = time.time()
        print(f"Running Case {tc['id']}: {tc['category']}...")
        ans, ctx, dbg = rules_lawyer.ask_rules_lawyer_game(tc["query"])
        elapsed = time.time() - start_t
        
        # Write to report
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(f"## Case {tc['id']} [{tc['category']}]\n")
            f.write(f"**Query**: {tc['query']}\n\n")
            f.write(f"**Expected Keywords**: {', '.join(tc['expected_keywords'])}\n")
            f.write(f"**Expected Rule**: {tc['expected_rule']}\n\n")
            f.write(f"### Agent Answer ({elapsed:.1f}s)\n")
            f.write(f"{ans}\n\n")
            f.write(f"**Debug**: Retrieved={dbg.get('num_retrieved')}, XRefs={dbg.get('num_cross_refs')}\n")
            f.write(f"**Top Sources**:\n")
            for doc, meta in ctx[:3]:
                src = meta.get("source_file", "unknown")
                p = meta.get("priority", 9)
                rn = meta.get("rule_number", "")
                f.write(f"- {src} (P{p}) Rule: {rn}\n")
            f.write("\n---\n\n")
            
        print(f"  -> Done in {elapsed:.1f}s")
        
if __name__ == "__main__":
    run_suite()
    print(f"\nBenchmark complete. Results written to {REPORT_FILE}")
