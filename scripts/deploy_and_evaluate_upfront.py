"""
deploy_and_evaluate_upfront.py

Unified End-to-End Orchestrator:
1. Executes `bash deploy.sh` in gaiia-rag-doll-cloud (CDK synth + deploy + outputs extraction).
2. Runs Up Front BGG Benchmark Tests 1 to 60 against live Cloud Production API (/api/ask).
3. Compares Cloud Production vs Local Baseline (upfront_bgg_eval_checkpoint.json).
4. Prints comprehensive comparison scorecard (Hit Rate, Recall, Latency, Concordance).
"""

import json
import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Directories
RAG_DOLL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLOUD_ROOT = os.path.abspath(os.path.join(RAG_DOLL_ROOT, "..", "gaiia-rag-doll-cloud"))

if RAG_DOLL_ROOT not in sys.path:
    sys.path.insert(0, RAG_DOLL_ROOT)

from scripts.run_upfront_bgg_eval import run_evaluation_batch, DEFAULT_CLOUD_API_URL


def run_cdk_deployment():
    print("\n" + "=" * 75)
    print("STEP 1: DEPLOYING CLOUD INFRASTRUCTURE VIA CDK (deploy.sh)")
    print("=" * 75)
    deploy_script = os.path.join(CLOUD_ROOT, "deploy.sh")
    if not os.path.exists(deploy_script):
        raise FileNotFoundError(f"deploy.sh not found at: {deploy_script}")

    # Determine bash executable: Git Bash or system bash
    git_bash = r"C:\Program Files\Git\bin\bash.exe"
    bash_cmd = git_bash if os.path.exists(git_bash) else "bash"

    print(f"🚀 Running: {bash_cmd} deploy.sh in {CLOUD_ROOT}")
    start = time.time()
    res = subprocess.run([bash_cmd, "deploy.sh"], cwd=CLOUD_ROOT, shell=False)
    elapsed = time.time() - start

    if res.returncode != 0:
        print(f"\n❌ Deployment failed with exit code {res.returncode}")
        sys.exit(res.returncode)

    print(f"\n✅ CDK Deployment completed successfully in {elapsed:.1f}s.")


def run_cloud_eval(start_idx: int = 1, end_idx: int = 60):
    print("\n" + "=" * 75)
    print(f"STEP 2: RUNNING BENCHMARK EVALUATION (TESTS {start_idx} TO {end_idx}) AGAINST CLOUD")
    print(f"Endpoint: {DEFAULT_CLOUD_API_URL}")
    print("=" * 75)

    checkpoint_file = os.path.join(RAG_DOLL_ROOT, "data", "eval", "upfront_cloud_eval_1_60.json")

    # If checkpoint exists from prior run, back it up or overwrite fresh
    if os.path.exists(checkpoint_file):
        backup_file = checkpoint_file.replace(".json", f"_prev_{int(time.time())}.json")
        try:
            os.rename(checkpoint_file, backup_file)
            print(f"📦 Archived previous checkpoint to: {os.path.basename(backup_file)}")
        except Exception as e:
            print(f"Notice: Could not archive previous checkpoint: {e}")

    run_evaluation_batch(
        checkpoint_path=checkpoint_file,
        start_index=start_idx,
        end_index=end_idx,
        target="cloud",
        api_url=DEFAULT_CLOUD_API_URL,
        tenant_slug="internal-test",
        title_slug="up-front",
        skip_judge=True,
    )

    print(f"\n✅ Completed evaluation batch for tests {start_idx} to {end_idx}.")
    return checkpoint_file


def compare_local_vs_cloud(cloud_path: str):
    print("\n" + "=" * 75)
    print("STEP 3: COMPARATIVE EVALUATION: LOCAL BASELINE vs CLOUD PRODUCTION")
    print("=" * 75)

    local_path = os.path.join(RAG_DOLL_ROOT, "data", "eval", "upfront_bgg_eval_checkpoint.json")
    benchmark_path = os.path.join(RAG_DOLL_ROOT, "data", "eval", "upfront_bgg_eval_benchmark.json")

    if not os.path.exists(local_path):
        print(f"❌ Local checkpoint not found at: {local_path}")
        return
    if not os.path.exists(cloud_path):
        print(f"❌ Cloud checkpoint not found at: {cloud_path}")
        return

    with open(local_path, "r", encoding="utf-8") as f:
        local_data = json.load(f)
    with open(cloud_path, "r", encoding="utf-8") as f:
        cloud_data = json.load(f)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    local_results = {r["id"]: r for r in local_data.get("results", [])}
    cloud_results = {r["id"]: r for r in cloud_data.get("results", [])}

    benchmark_1_60 = benchmark[:60]
    eval_ids = [b["id"] for b in benchmark_1_60]

    common_ids = [cid for cid in eval_ids if cid in local_results and cid in cloud_results]
    with_rules = [cid for cid in common_ids if benchmark_1_60[eval_ids.index(cid)].get("expected_rule_citations")]

    # Local metrics
    loc_hits = [cid for cid in with_rules if local_results[cid].get("rule_hit")]
    loc_recalls = [local_results[cid].get("rule_recall", 0) for cid in with_rules]
    loc_latencies = [local_results[cid].get("latency_seconds", 0) for cid in common_ids]

    loc_hit_rate = (len(loc_hits) / len(with_rules)) * 100 if with_rules else 0
    loc_avg_recall = sum(loc_recalls) / len(loc_recalls) if loc_recalls else 0
    loc_avg_latency = sum(loc_latencies) / len(loc_latencies) if loc_latencies else 0

    # Cloud metrics
    cld_hits = [cid for cid in with_rules if cloud_results[cid].get("rule_hit")]
    cld_recalls = [cloud_results[cid].get("rule_recall", 0) for cid in with_rules]
    cld_latencies = [cloud_results[cid].get("latency_seconds", 0) for cid in common_ids]

    cld_hit_rate = (len(cld_hits) / len(with_rules)) * 100 if with_rules else 0
    cld_avg_recall = sum(cld_recalls) / len(cld_recalls) if cld_recalls else 0
    cld_avg_latency = sum(cld_latencies) / len(cld_latencies) if cld_latencies else 0

    speedup = (loc_avg_latency / cld_avg_latency) if cld_avg_latency > 0 else 0

    print("\n" + "=" * 75)
    print("⚖️ FINAL SCORECARD: LOCAL RAG-DOLL vs CLOUD PRODUCTION (TESTS 1–60)")
    print("=" * 75)
    print(f"{'Metric':<32} | {'Local Baseline':<20} | {'Cloud Production':<20}")
    print("-" * 75)
    print(f"{'Total Evaluated':<32} | {len(common_ids):<20} | {len(common_ids):<20}")
    print(f"{'Rule Citation Hit Rate':<32} | {len(loc_hits)}/{len(with_rules)} ({loc_hit_rate:.1f}%)" + " " * 8 + f" | {len(cld_hits)}/{len(with_rules)} ({cld_hit_rate:.1f}%)")
    print(f"{'Average Rule Recall':<32} | {loc_avg_recall:.2f}" + " " * 16 + f" | {cld_avg_recall:.2f}")
    print(f"{'Average Latency per Query':<32} | {loc_avg_latency:.2f}s" + " " * 14 + f" | {cld_avg_latency:.2f}s")
    print(f"{'Throughput / Speedup':<32} | 1.0x (Baseline)" + " " * 5 + f" | {speedup:.1f}x Faster")
    print("=" * 75)

    both_hit = [cid for cid in with_rules if cid in loc_hits and cid in cld_hits]
    cld_only = [cid for cid in with_rules if cid not in loc_hits and cid in cld_hits]
    loc_only = [cid for cid in with_rules if cid in loc_hits and cid not in cld_hits]
    both_miss = [cid for cid in with_rules if cid not in loc_hits and cid not in cld_hits]

    print("\n🔍 CONCORDANCE:")
    print(f"  • Both Hit:                {len(both_hit)} tests ({len(both_hit)/len(with_rules)*100:.1f}%)")
    print(f"  • Cloud Only (Local Miss): {len(cld_only)} tests")
    print(f"  • Local Only (Cloud Miss): {len(loc_only)} tests")
    print(f"  • Both Missed:             {len(both_miss)} tests")

    if loc_only:
        print("\n⚠️ REMAINING CLOUD MISSES (Local Hit, Cloud Missed):")
        for cid in loc_only:
            b_item = benchmark_1_60[eval_ids.index(cid)]
            exp = ", ".join(b_item.get("expected_rule_citations", []))
            c_r = cloud_results.get(cid, {})
            c_retrieved = ", ".join(c_r.get("retrieved_rules", []))
            c_verdict = c_r.get("verdict", "")
            print(f"  - [{cid}] Expected: {exp:<8} | Retrieved: [{c_retrieved}] | Verdict: {c_verdict}")
            print(f"    Title: {b_item.get('title', '')}")

    if both_miss:
        print("\n⚠️ BOTH MISSED (Local Miss & Cloud Miss):")
        for cid in both_miss:
            b_item = benchmark_1_60[eval_ids.index(cid)]
            exp = ", ".join(b_item.get("expected_rule_citations", []))
            c_r = cloud_results.get(cid, {})
            c_retrieved = ", ".join(c_r.get("retrieved_rules", []))
            print(f"  - [{cid}] Expected: {exp:<8} | Retrieved: [{c_retrieved}]")
            print(f"    Title: {b_item.get('title', '')}")


def main():
    compare_only = "--compare-only" in sys.argv
    skip_deploy = "--skip-deploy" in sys.argv or compare_only

    checkpoint_file = os.path.join(RAG_DOLL_ROOT, "data", "eval", "upfront_cloud_eval_1_60.json")

    if compare_only:
        print("📊 Running in compare-only mode on latest checkpoint.")
        compare_local_vs_cloud(checkpoint_file)
        return

    if not skip_deploy:
        run_cdk_deployment()
    else:
        print("⏩ Skipping CDK deployment as --skip-deploy was provided.")

    cloud_checkpoint = run_cloud_eval(start_idx=1, end_idx=60)
    compare_local_vs_cloud(cloud_checkpoint)


if __name__ == "__main__":
    main()
