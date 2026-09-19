import os
import sys
import json
import time
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CHECKPOINT_FILE = "data/eval/asl_bgg_eval_checkpoint.json"

def auto_chain():
    print("=" * 70)
    print("🔄 ASL BGG EVAL AUTO-CHAIN CONTROLLER (Items 21–60)")
    print("=" * 70)
    print("Monitoring completion of Items 1–20 in checkpoint...")
    
    last_count = 0
    while True:
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    chk = json.load(f)
                results = chk.get("results", [])
                curr_count = len(results)
                if curr_count != last_count:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint progress: {curr_count} / 20 items completed.", flush=True)
                    last_count = curr_count
                
                if curr_count >= 20:
                    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Items 1–20 complete ({curr_count} items in checkpoint).", flush=True)
                    print("🚀 Launching evaluation for Items 21 to 60...", flush=True)
                    break
            except Exception as e:
                print(f"Warning reading checkpoint: {e}", flush=True)
        time.sleep(15)

    # Launch run_asl_bgg_eval.py --start 21 --end 60
    cmd = [
        sys.executable,
        "scripts/run_asl_bgg_eval.py",
        "--start", "21",
        "--end", "60"
    ]
    print(f"Running command: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Batch 21–60 execution finished with exit code {proc.returncode}.", flush=True)

if __name__ == "__main__":
    auto_chain()
