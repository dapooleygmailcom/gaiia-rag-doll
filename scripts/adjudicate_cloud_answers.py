"""
adjudicate_cloud_answers.py

Runs the authoritative Local LLM Referee (llama3.1:8b via Ollama REST API)
using the exact prompt template from engine/evaluation/adjudicator.py
on the candidate answers returned by Cloud Bedrock Nova Pro for the 6 remaining citation misses.
Judges whether the answers are substantively and mechanically accurate despite missing or differing BGG citations.
"""

import os
import sys
import json
import time
import re
import urllib.request
import urllib.error

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_FILE = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_bgg_eval_benchmark.json")
RETEST_FILE = os.path.join(PROJECT_ROOT, "data", "eval", "retest_failures_results.json")
FULL_100_FILE = os.path.join(PROJECT_ROOT, "data", "eval", "upfront_cloud_eval_1_100.json")

JUDGE_PROMPT_TEMPLATE = """You are an authoritative rules referee and evaluation adjudicator for wargaming rules reference systems.
A candidate AI answered a user's rules question. In automated testing, it had a missing or partial rule citation match.
Your task is to judge whether the Candidate Answer is SUBSTANTIVELY, MECHANICALLY, and FACTUALLY ACCURATE and COMPLETE when compared to the authoritative Ground Truth ruling.

USER QUESTION:
{query}

GROUND TRUTH EXPERT RULING:
{ground_truth}

CANDIDATE AI ANSWER:
{candidate_answer}

JUDGING CRITERIA:
1. Ignore conversational filler, tone, and formatting in either answer.
2. Determine if the Candidate reaches the same core mechanical outcome as the Ground Truth (e.g., allowed vs forbidden, modifier calculation, phase timing, procedure validity).
3. If the Candidate provides MORE GRANULAR, PRECISE governing sub-rules (e.g., citing specific sub-clauses instead of broad parent sections) or omits tangential/irrelevant footnotes from forum banter, classify as "AGREE" (substantive_accuracy: true).
4. If the Ground Truth states that no specific rule prohibits an action (or references a rule purely by analogy) and the Candidate correctly concludes the action is permitted under governing movement/combat rules, classify as "AGREE" (substantive_accuracy: true).
5. If the Candidate misses an essential required modifier, contradicts the core ruling outcome, or gives false mechanics, classify as "DISAGREE" or "PARTIAL" (substantive_accuracy: false).

Respond with ONLY a JSON object:
{{
  "ruling": "AGREE" | "DISAGREE" | "PARTIAL",
  "substantive_accuracy": true | false,
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "rationale": "<1-2 concise sentences explaining why the rulings align or disagree>"
}}
"""

def adjudicate_via_ollama(query: str, ground_truth: str, candidate_answer: str, model: str = "llama3.1:8b") -> dict:
    if not candidate_answer or not ground_truth:
        return {
            "adjudicated": False,
            "ruling": "SKIPPED",
            "substantive_accuracy": False,
            "confidence": "LOW",
            "rationale": "Missing answer or ground truth.",
            "latency_seconds": 0.0
        }

    prompt = (
        JUDGE_PROMPT_TEMPLATE
        .replace("{query}", query.strip())
        .replace("{ground_truth}", ground_truth.strip())
        .replace("{candidate_answer}", candidate_answer.strip())
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9
        }
    }

    t0 = time.time()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            latency = round(time.time() - t0, 2)
            data = json.loads(body)
            raw = data.get("response", "").strip()

            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                ruling = str(parsed.get("ruling", "DISAGREE")).upper()
                substantive = bool(parsed.get("substantive_accuracy", ruling == "AGREE"))
                confidence = str(parsed.get("confidence", "MEDIUM")).upper()
                rationale = str(parsed.get("rationale", "")).strip()
            else:
                substantive = "agree" in raw.lower() and "disagree" not in raw.lower()
                ruling = "AGREE" if substantive else "DISAGREE"
                confidence = "LOW"
                rationale = raw[:200]

            return {
                "adjudicated": True,
                "judge_model": model,
                "ruling": ruling,
                "substantive_accuracy": substantive,
                "confidence": confidence,
                "rationale": rationale,
                "latency_seconds": latency
            }
    except Exception as e:
        return {
            "adjudicated": False,
            "ruling": "ERROR",
            "substantive_accuracy": False,
            "confidence": "LOW",
            "rationale": str(e),
            "latency_seconds": round(time.time() - t0, 2)
        }


def main():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        bench_map = {b["id"]: b for b in json.load(f)}

    with open(FULL_100_FILE, "r", encoding="utf-8") as f:
        full_100 = {r["id"]: r for r in json.load(f).get("results", [])}

    with open(RETEST_FILE, "r", encoding="utf-8") as f:
        retest_results = json.load(f)

    # The 6 citation misses
    misses = [r for r in retest_results if not r.get("is_hit")]
    print(f"🎯 Evaluating {len(misses)} citation misses with Local LLM Referee (llama3.1:8b via Ollama)...\n")

    results = []
    for m in misses:
        cid = m["id"]
        idx = m["overall_idx"]
        b = bench_map[cid]
        query = b["query"]
        gt = b.get("ground_truth_answer", "")
        c_item = full_100.get(cid, {})
        candidate_answer = c_item.get("generated_answer", "")
        if not candidate_answer:
            candidate_answer = m.get("answer_preview", "")

        print("=" * 80)
        print(f"[#{idx}] {cid}: {m['title']}")
        print(f"Expected BGG Citation: {m['expected']}")
        print(f"Candidate Answer Preview: {candidate_answer[:160]}...")
        print(f"Ground Truth Preview:     {gt[:160]}...")
        print("-" * 80)

        adj = adjudicate_via_ollama(
            query=query,
            ground_truth=gt,
            candidate_answer=candidate_answer,
            model="llama3.1:8b"
        )

        ruling = adj.get("ruling", "UNKNOWN")
        substantive = adj.get("substantive_accuracy", False)
        rationale = adj.get("rationale", "")
        conf = adj.get("confidence", "LOW")

        print(f"⚖️  Referee Verdict: {'✅ AGREE (Substantively Accurate)' if substantive else '❌ DISAGREE'}")
        print(f"   Confidence: {conf} | Latency: {adj.get('latency_seconds')}s")
        print(f"   Rationale : {rationale}\n")

        results.append({
            "overall_idx": idx,
            "id": cid,
            "title": m["title"],
            "expected_rules": m["expected"],
            "citation_hit": False,
            "referee_ruling": ruling,
            "substantive_accuracy": substantive,
            "confidence": conf,
            "rationale": rationale,
            "candidate_answer": candidate_answer,
            "ground_truth": gt
        })

    print("=" * 80)
    print("📊 LOCAL LLM REFEREE ADJUDICATION SUMMARY (ON CITATION MISSES)")
    print("=" * 80)
    print(f"{'#':<4} | {'ID':<16} | {'Expected':<12} | {'Citation':<10} | {'Referee Verdict':<24} | {'Accurate?'}")
    print("-" * 80)
    agree_count = 0
    for r in results:
        sub = r["substantive_accuracy"]
        if sub:
            agree_count += 1
        verdict_str = f"{r['referee_ruling']} ({r['confidence']})"
        acc_str = "YES ✅" if sub else "NO ❌"
        print(f"{r['overall_idx']:<4} | {r['id']:<16} | {str(r['expected_rules']):<12} | {'MISS':<10} | {verdict_str:<24} | {acc_str}")
    print("=" * 80)
    print(f"🎯 Outcome: {agree_count} of {len(results)} citation misses were ruled SUBSTANTIVELY & MECHANICALLY ACCURATE by Local Referee!")
    print("=" * 80)

    out_file = os.path.join(PROJECT_ROOT, "data", "eval", "cloud_referee_adjudication_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
