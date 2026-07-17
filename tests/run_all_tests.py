import os
import sys
import subprocess
import time
import json

# Ensure log directory exists
os.makedirs("data/logs", exist_ok=True)
MASTER_LOG = "data/logs/master_test_run.log"
REPORT_FILE = "data/logs/master_report.md"

TEST_SUITES = [
    {
        "name": "Pytest Unit & Local RAG Tests",
        "type": "unit",
        "command": [sys.executable, "-m", "pytest", "tests/"],
        "log_file": "data/logs/pytest_unit_tests.log"
    },
    {
        "name": "Rules Lawyer Evaluation Suite",
        "type": "llm",
        "command": [sys.executable, "tests/test_rules_lawyer.py"],
        "log_file": "data/logs/rules_lawyer_eval.log"
    },
    {
        "name": "RAG Massive Accuracy Suite",
        "type": "llm",
        "command": [sys.executable, "tests/run_massive_tests.py"],
        "log_file": "data/logs/massive_tests.log"
    },
    {
        "name": "Advanced Policy Comparison UAT",
        "type": "llm",
        "command": [sys.executable, "tests/test_policy_rag.py"],
        "log_file": "data/logs/policy_rag_tests.log"
    },
    {
        "name": "ASL Game Smoke Test",
        "type": "smoke",
        "command": [sys.executable, "tests/test_asl_sample.py"],
        "log_file": "data/logs/smoke_asl.log"
    },
    {
        "name": "Star Fleet Battles Smoke Test",
        "type": "smoke",
        "command": [sys.executable, "tests/test_sfb_sample.py"],
        "log_file": "data/logs/smoke_sfb.log"
    },
    {
        "name": "Warhammer 40K Smoke Test",
        "type": "smoke",
        "command": [sys.executable, "tests/test_40k_sample.py"],
        "log_file": "data/logs/smoke_40k.log"
    },
    {
        "name": "Home Insurance Smoke Test",
        "type": "smoke",
        "command": [sys.executable, "tests/test_insurance.py"],
        "log_file": "data/logs/smoke_insurance.log"
    },
    {
        "name": "Renegade Legion Smoke Test",
        "type": "smoke",
        "command": [sys.executable, "tests/test_renegade_legion.py"],
        "log_file": "data/logs/smoke_renegade_legion.log"
    },
    {
        "name": "Prime Radiant Level 1 — RL Retrieval Validation",
        "type": "llm",
        "command": [sys.executable, "tests/test_rl_level1_retrieval.py"],
        "log_file": "data/logs/rl_level1_retrieval.log"
    },
    {
        "name": "Glossary Extraction Test",
        "type": "smoke",
        "command": [sys.executable, "tests/test_glossary.py"],
        "log_file": "data/logs/smoke_glossary.log"
    },
    {
        "name": "Regex Acronyms Extractor Test",
        "type": "smoke",
        "command": [sys.executable, "tests/test_regex_acronyms.py"],
        "log_file": "data/logs/smoke_regex_acronyms.log"
    }
]

def log_to_master(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def run_suite(suite):
    name = suite["name"]
    cmd = suite["command"]
    log_path = suite["log_file"]
    
    log_to_master(f"STARTING SUITE: {name}")
    log_to_master(f"Command: {' '.join(cmd)}")
    log_to_master(f"Logging to: {log_path}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    start_time = time.time()
    
    # We will run the subprocess and capture both stdout and stderr in a file
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            
            # Read stdout line by line as it executes and write it to log file
            # also print major updates periodically
            last_print_time = time.time()
            line_count = 0
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    log_file.write(line)
                    log_file.flush()
                    line_count += 1
                    
                    # Print progress every 15 seconds to master output
                    current_time = time.time()
                    if current_time - last_print_time > 15:
                        duration = current_time - start_time
                        log_to_master(f"  --> Progress {name}: {duration:.0f}s elapsed, {line_count} lines of log written.")
                        last_print_time = current_time
            
            process.wait()
            return_code = process.returncode
            
    except Exception as e:
        log_to_master(f"  ERROR executing process: {e}")
        return_code = -99
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\nExecution error: {e}\n")
            
    duration = time.time() - start_time
    status = "PASSED" if return_code == 0 else "FAILED"
    
    log_to_master(f"COMPLETED SUITE: {name} | Status: {status} | Return Code: {return_code} | Duration: {duration:.1f}s")
    log_to_master("-" * 60)
    
    return {
        "name": name,
        "type": suite["type"],
        "status": status,
        "return_code": return_code,
        "duration": duration,
        "log_file": log_path
    }

def generate_report(results, total_duration):
    log_to_master("Generating final markdown report...")
    
    # Check if we have specific JSON summary reports to extract additional info
    rules_lawyer_sum = "data/logs/rules_lawyer_test_summary.json"
    massive_sum = "data/logs/massive_test_summary.json"
    policy_sum = "data/logs/policy_test_summary.json"
    
    additional_info = ""
    
    if os.path.exists(rules_lawyer_sum):
        try:
            with open(rules_lawyer_sum, "r", encoding="utf-8") as f:
                d = json.load(f)
                scores = d.get("scores", {})
                additional_info += f"### Rules Lawyer Evaluation Summary\n"
                additional_info += f"- **Total Cases**: {scores.get('total')}\n"
                additional_info += f"- **Perfect Pass Rate**: {scores.get('perfect') / scores.get('total') * 100:.1f}%\n"
                additional_info += f"- **Faithfulness Pass Rate**: {scores.get('faithful') / scores.get('total') * 100:.1f}%\n"
                additional_info += f"- **Relevancy Pass Rate**: {scores.get('relevant') / scores.get('total') * 100:.1f}%\n"
                additional_info += f"- **Citation Pass Rate**: {scores.get('cited') / scores.get('total') * 100:.1f}%\n\n"
        except Exception as e:
            pass

    if os.path.exists(massive_sum):
        try:
            with open(massive_sum, "r", encoding="utf-8") as f:
                d = json.load(f)
                additional_info += f"### RAG Massive Accuracy Summary\n"
                additional_info += f"- **Total Cases**: {d.get('overall_total')}\n"
                additional_info += f"- **Perfect Pass Rate**: {d.get('overall_pass_rate')}%\n"
                additional_info += f"- **Faithfulness Pass Rate**: {d.get('overall_faithful') / d.get('overall_total') * 100:.1f}%\n"
                additional_info += f"- **Relevancy Pass Rate**: {d.get('overall_relevant') / d.get('overall_total') * 100:.1f}%\n\n"
        except Exception as e:
            pass

    if os.path.exists(policy_sum):
        try:
            with open(policy_sum, "r", encoding="utf-8") as f:
                d = json.load(f)
                additional_info += f"### Advanced Policy Comparison Summary\n"
                additional_info += f"- **Total Cases**: {d.get('total_cases')}\n"
                additional_info += f"- **Perfect Pass Rate**: {d.get('perfect_pass_rate')}%\n"
                additional_info += f"- **Faithfulness Pass Rate**: {d.get('total_faithful') / d.get('total_cases') * 100:.1f}%\n"
                additional_info += f"- **Relevancy Pass Rate**: {d.get('total_relevant') / d.get('total_cases') * 100:.1f}%\n\n"
        except Exception as e:
            pass

    report = f"""# Master Test Execution Report

**Date/Time**: {time.strftime("%Y-%m-%d %H:%M:%S")}  
**Total Run Duration**: {total_duration / 60:.2f} minutes

## Test Execution Summary

| Test Suite | Type | Status | Return Code | Duration | Log File |
| :--- | :---: | :---: | :---: | :---: | :--- |
"""
    for res in results:
        status_emoji = "✅ PASSED" if res["status"] == "PASSED" else "❌ FAILED"
        report += f"| {res['name']} | {res['type'].upper()} | {status_emoji} | {res['return_code']} | {res['duration']:.1f}s | [{os.path.basename(res['log_file'])}](file:///{os.path.abspath(res['log_file']).replace(os.sep, '/')}) |\n"

    report += "\n" + additional_info
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    
    log_to_master(f"Master report written to: {REPORT_FILE}")

def main():
    if os.path.exists(MASTER_LOG):
        os.remove(MASTER_LOG)
        
    # Clean up old cached results to run from scratch
    for cached_file in [
        "data/logs/rules_lawyer_test_results.json",
        "data/logs/massive_test_results.json",
        "data/logs/policy_test_results.json"
    ]:
        if os.path.exists(cached_file):
            try:
                os.remove(cached_file)
                print(f"Cleared cache: {cached_file}")
            except Exception as e:
                print(f"Error clearing cache {cached_file}: {e}")

    log_to_master("=" * 60)
    log_to_master("GAIIA MASTER TEST RUNNER STARTED")
    log_to_master("=" * 60)
    
    start_time = time.time()
    results = []
    
    for suite in TEST_SUITES:
        res = run_suite(suite)
        results.append(res)
        
    total_duration = time.time() - start_time
    
    log_to_master("=" * 60)
    log_to_master(f"GAIIA MASTER TEST RUNNER COMPLETED | Total Duration: {total_duration / 60:.2f} minutes")
    log_to_master("=" * 60)
    
    generate_report(results, total_duration)

if __name__ == "__main__":
    main()
