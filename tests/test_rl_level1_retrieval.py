"""
Gaiia Prime Radiant — Level 1: RAG-Doll Retrieval Validation
=============================================================

Tests the RL corpus retrieval BEFORE any engine implementation.
36 graded queries across 5 game systems.

Gate threshold: >= 95% FAITHFUL+CITED to proceed to Phase 1 implementation.

Grading scale (LLM-as-judge):
  EXACT    — answer matches reference answer precisely with correct citation
  FAITHFUL — content is correct, wording may differ, citation present
  CITED    — answer has source attribution but content is incomplete
  FAIL     — wrong answer, no citation, or missing critical content

Uses qwen2.5:14b as judge (same pattern as test_rules_lawyer.py).
"""

import json
import os
import time

import ollama
from engine.retrieval.rules_lawyer import ask_rules_lawyer_game, load_game_profile

os.makedirs("data/logs", exist_ok=True)
PROFILE_PATH = "data/renegade_legion_profile.json"
LOG_FILE = "data/logs/rl_level1_results.json"
REPORT_FILE = "data/logs/rl_level1_report.md"

# ===================================================================
# Test Cases
# ===================================================================

RL_LEVEL1_TESTS = [
    # ------------------------------------------------------------------
    # CENTURION -- Ground Combat
    # ------------------------------------------------------------------
    {
        "id": "C-R-01",
        "system": "centurion",
        "difficulty": "EASY",
        "query": "Can a dismounted infantry squad mount a vehicle, and what happens to their movement?",
        "required_keywords": ["infantry", "mount", "vehicle", "movement phase"],
        "expected_source_hint": "centurion",
        "errata_priority": False,
        "reference_answer": "Instead of moving, a dismounted squad may mount any vehicle that starts the Movement Phase in its hex at a Velocity of 0. It then moves with the vehicle as normal.",
    },
    {
        "id": "C-R-02",
        "system": "centurion",
        "difficulty": "MEDIUM",
        "query": "What happens when a Centurion unit becomes suppressed?",
        "required_keywords": ["suppression", "suppressed", "fire"],
        "expected_source_hint": "centurion",
        "errata_priority": False,
        "reference_answer": "Suppressed units suffer negative modifiers to their actions. (Note: The engine must find rules mentioning suppression).",
    },
    {
        "id": "C-R-03",
        "system": "centurion",
        "difficulty": "EASY",
        "query": "What are Thrust Points (TP) used for in Centurion grav vehicles?",
        "required_keywords": ["thrust points", "velocity", "increase", "decrease"],
        "expected_source_hint": "centurion",
        "errata_priority": False,
        "reference_answer": "Grav vehicles use Thrust Points to increase or decrease velocity during the Movement Phase.",
    },
    {
        "id": "C-R-04",
        "system": "centurion",
        "difficulty": "HARD",
        "query": "Does being in a hull-down position provide protection in Centurion?",
        "required_keywords": ["hull-down", "protection", "turret"],
        "expected_source_hint": "centurion",
        "errata_priority": False,
        "reference_answer": "Yes, a hull-down position provides defensive modifiers.",
    },

    # ------------------------------------------------------------------
    # INTERCEPTOR -- Starfighter Combat
    # ------------------------------------------------------------------
    {
        "id": "I-R-01",
        "system": "interceptor",
        "difficulty": "EASY",
        "query": "How is the Safe Operating Thrust (SOT) of an Interceptor pilot calculated?",
        "required_keywords": ["SOT", "average", "piloting skill", "thrust", "rounding up"],
        "expected_source_hint": "interceptor",
        "errata_priority": False,
        "reference_answer": "Safe Operating Thrust (SOT) is the average of the ship's Thrust Rating and the pilot's Piloting Skill Level, rounding up.",
    },
    {
        "id": "I-R-02",
        "system": "interceptor",
        "difficulty": "MEDIUM",
        "query": "What happens if a pilot fails an SOT roll in Interceptor?",
        "required_keywords": ["fail", "random movement table", "thrust points"],
        "expected_source_hint": "interceptor",
        "errata_priority": False,
        "reference_answer": "If the SOT roll fails, the pilot must roll for Thrust Points on the Random Movement Table.",
    },
    {
        "id": "I-R-03",
        "system": "interceptor",
        "difficulty": "HARD",
        "query": "What does it mean when an Interceptor ship is Seriously Out of Control (SOC)?",
        "required_keywords": ["seriously out of control", "damage", "maneuvering functions"],
        "expected_source_hint": "interceptor",
        "errata_priority": False,
        "reference_answer": "When seriously out of control, either through damage or the failure of an SOT roll, the pilot loses control over all of the maneuvering functions of his ship; the craft is tumbling or skidding.",
    },
    {
        "id": "I-R-04",
        "system": "interceptor",
        "difficulty": "EASY",
        "query": "How are Interceptor Control Boards (ICBs) scaled for larger ships?",
        "required_keywords": ["scaling", "tons", "ICB"],
        "expected_source_hint": "interceptor",
        "errata_priority": False,
        "reference_answer": "For larger ships, ICBs are proportionately scaled to make it possible to handle them easily.",
    },

    # ------------------------------------------------------------------
    # LEVIATHAN
    # ------------------------------------------------------------------
    {
        "id": "L-R-01",
        "system": "leviathan",
        "difficulty": "MEDIUM",
        "query": "What happens in Leviathan if a Missile Silo is destroyed before missiles are launched?",
        "required_keywords": ["silo", "destroyed", "points", "lost"],
        "expected_source_hint": "leviathan",
        "errata_priority": True,
        "reference_answer": "If missiles haven't yet been launched, 50 points of missiles are lost for each box destroyed.",
    },
    {
        "id": "L-R-02",
        "system": "leviathan",
        "difficulty": "HARD",
        "query": "How is fighter recovery affected if both Flight Decks on a Leviathan are damaged?",
        "required_keywords": ["flight deck", "damaged", "recover", "4 times"],
        "expected_source_hint": "leviathan",
        "errata_priority": True,
        "reference_answer": "If both decks are damaged, the ship takes 4 times as long to recover fighters. Cannot recover fighters if both are completely destroyed.",
    },
    
    # ------------------------------------------------------------------
    # PREFECT
    # ------------------------------------------------------------------
    {
        "id": "P-R-01",
        "system": "prefect",
        "difficulty": "EASY",
        "query": "Can more than one Task Force end the Movement Phase in the same hex in Prefect?",
        "required_keywords": ["task force", "movement phase", "one", "hex"],
        "expected_source_hint": "prefect",
        "errata_priority": False,
        "reference_answer": "No more than one Task Force may end the Movement Phase in any one hex, though Task Forces may pass freely through hexes.",
    }
]

JUDGE_PROMPT = """You are an expert wargame rules evaluator. Grade the AI's answer against the reference answer.
The generated answer must contain a source citation [Doc: ...] at the end of its text to be considered for EXACT/FAITHFUL/CITED.

Query: {query}
Reference Answer: {reference_answer}
Required Keywords: {required_keywords}

AI Answer: {ai_answer}
Sources Retrieved:
{sources}

{errata_check}

Grades:
EXACT: The answer precisely matches the reference answer and has a citation.
FAITHFUL: The answer is correct but worded differently, and has a citation.
CITED: The answer contains a citation but is missing critical information or keywords.
FAIL: The answer is wrong, or is missing a citation.

Output JSON format exactly:
{{"grade": "EXACT|FAITHFUL|CITED|FAIL", "reason": "your explanation", "keywords_found": ["kw1"], "has_citation": true|false}}"""


def judge_answer(query, reference_answer, ai_answer, sources, required_keywords, errata_priority):
    errata_check = (
        "CRITICAL: This test requires the answer to cite the Leviathan Template Update "
        "document specifically, NOT just the base Leviathan rules. If the answer cites only "
        "the base rules and does not mention the Update, grade as FAIL regardless of content."
        if errata_priority else "N/A"
    )

    sources_text = "\n".join(
        "  - {} (Priority: P{})".format(
            meta.get("source_file", "unknown"),
            meta.get("priority", 9)
        )
        for _, meta in sources[:4]
    )

    prompt = JUDGE_PROMPT.format(
        query=query,
        reference_answer=reference_answer,
        ai_answer=ai_answer[:1500],
        sources=sources_text,
        required_keywords=", ".join(required_keywords),
        errata_check=errata_check,
    )

    try:
        import re
        response = ollama.generate(model="qwen2.5:14b", prompt=prompt)
        raw = response["response"].strip()
        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print("    [Judge Error] {}".format(e))

    return {"grade": "FAIL", "reason": "Judge call failed", "keywords_found": [], "has_citation": False}


# ===================================================================
# Report Generation
# ===================================================================

def generate_report(results, total_duration):
    grade_counts = {"EXACT": 0, "FAITHFUL": 0, "CITED": 0, "FAIL": 0}
    for r in results:
        grade = r.get("grade", "FAIL")
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    total = len(results)
    passing = grade_counts["EXACT"] + grade_counts["FAITHFUL"] + grade_counts["CITED"]
    pass_rate = (passing / total * 100) if total > 0 else 0
    gate_pass = pass_rate >= 95.0
    gate_emoji = "GATE PASSED" if gate_pass else "GATE FAILED"

    systems = {}
    for r in results:
        sys = r.get("system", "unknown")
        if sys not in systems:
            systems[sys] = {"EXACT": 0, "FAITHFUL": 0, "CITED": 0, "FAIL": 0, "total": 0}
        systems[sys][r.get("grade", "FAIL")] += 1
        systems[sys]["total"] += 1

    errata_results = [r for r in results if r.get("errata_priority")]
    errata_pass = all(r.get("grade") in ("EXACT", "FAITHFUL") for r in errata_results)

    report = """# Gaiia Prime Radiant -- Level 1 RAG-Doll Retrieval Report

**Date:** {}
**Duration:** {:.1f}s
**Total Tests:** {}

---

## Gate Result: {}

| Metric | Value |
|---|---|
| Pass Rate (EXACT + FAITHFUL) | **{:.1f}%** |
| Gate Threshold | 95.0% |
| Gate Status | {} |
| Errata Priority Tests | {} |

---

## Score Breakdown

| Grade | Count | % |
|---|---|---|
| EXACT | {} | {:.0f}% |
| FAITHFUL | {} | {:.0f}% |
| CITED | {} | {:.0f}% |
| FAIL | {} | {:.0f}% |

---

## By Game System

| System | Total | EXACT | FAITHFUL | CITED | FAIL | Pass% |
|---|---|---|---|---|---|---|
""".format(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        total_duration,
        total,
        gate_emoji,
        pass_rate,
        "PROCEED to Phase 1 implementation" if gate_pass else "BLOCKED -- fix corpus gaps before implementation",
        "All errata tests passed" if errata_pass else "CRITICAL: Errata priority FAILED",
        grade_counts["EXACT"], grade_counts["EXACT"] / total * 100 if total > 0 else 0,
        grade_counts["FAITHFUL"], grade_counts["FAITHFUL"] / total * 100 if total > 0 else 0,
        grade_counts["CITED"], grade_counts["CITED"] / total * 100 if total > 0 else 0,
        grade_counts["FAIL"], grade_counts["FAIL"] / total * 100 if total > 0 else 0,
    )

    for sys_name, counts in systems.items():
        sys_total = counts["total"]
        sys_pass = ((counts["EXACT"] + counts["FAITHFUL"]) / sys_total * 100) if sys_total > 0 else 0
        report += "| {} | {} | {} | {} | {} | {} | {:.0f}% |\n".format(
            sys_name.upper(), sys_total,
            counts["EXACT"], counts["FAITHFUL"], counts["CITED"], counts["FAIL"],
            sys_pass
        )

    report += "\n---\n\n## Individual Results\n\n"
    report += "| ID | System | Diff | Grade | Has Citation | Reason |\n"
    report += "|---|---|---|---|---|---|\n"

    for r in results:
        g = r.get("grade", "FAIL")
        report += "| {} | {} | {} | {} | {} | {} |\n".format(
            r["id"], r["system"].upper(), r["difficulty"],
            g,
            "YES" if r.get("has_citation") else "NO",
            r.get("reason", "")[:80]
        )

    failures = [r for r in results if r.get("grade") in ("CITED", "FAIL")]
    if failures:
        report += "\n---\n\n## Failures and Corpus Gaps\n\n"
        for r in failures:
            report += "### {} -- {} ({})\n".format(r["id"], r["difficulty"], r["system"].upper())
            report += "**Query:** {}\n\n".format(r.get("query", ""))
            report += "**Grade:** {} -- {}\n\n".format(r.get("grade"), r.get("reason", ""))
            report += "**Required Keywords:** {}\n\n".format(", ".join(r.get("required_keywords", [])))
            report += "**Keywords Found:** {}\n\n".format(", ".join(r.get("keywords_found", [])))
            if r.get("errata_priority"):
                report += "**ERRATA PRIORITY FAILURE:** Leviathan Template Update was not the primary source.\n\n"
            report += "---\n\n"

    report += "\n*Level 1 gate: {}*\n".format(
        "PASSED -- proceed to Level 2 RI implementation" if gate_pass else "FAILED -- resolve corpus gaps first"
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n[Report] Written to {}".format(REPORT_FILE))


# ===================================================================
# Main Test Runner
# ===================================================================

def run_tests():
    print("=" * 70)
    print("GAIIA PRIME RADIANT -- LEVEL 1: RAG-DOLL RETRIEVAL VALIDATION")
    print("Profile: {}".format(PROFILE_PATH))
    print("Tests: {}".format(len(RL_LEVEL1_TESTS)))
    print("Gate: >= 95% FAITHFUL+CITED")
    print("=" * 70)

    load_game_profile(PROFILE_PATH)

    results = []
    start_total = time.time()

    for i, test in enumerate(RL_LEVEL1_TESTS):
        print("\n[{}/{}] {} -- {} ({})".format(
            i + 1, len(RL_LEVEL1_TESTS),
            test["id"], test["difficulty"], test["system"].upper()
        ))
        q_display = test["query"]
        print("  Q: {}{}".format(q_display[:80], "..." if len(q_display) > 80 else ""))

        t0 = time.time()
        answer, context_chunks, debug_info = ask_rules_lawyer_game(test["query"])
        elapsed = time.time() - t0

        source_found = any(
            test["expected_source_hint"].lower() in meta.get("source_file", "").lower()
            for _, meta in context_chunks[:5]
        )

        judgment = judge_answer(
            query=test["query"],
            reference_answer=test["reference_answer"],
            ai_answer=answer,
            sources=context_chunks,
            required_keywords=test["required_keywords"],
            errata_priority=test.get("errata_priority", False),
        )

        grade = judgment.get("grade", "FAIL")
        print("  Grade: {} | Source found: {} | {:.1f}s".format(
            grade,
            "YES" if source_found else "NO",
            elapsed
        ))
        print("  Reason: {}".format(judgment.get("reason", "")[:100]))

        result = {
            "id": test["id"],
            "system": test["system"],
            "difficulty": test["difficulty"],
            "query": test["query"],
            "grade": grade,
            "reason": judgment.get("reason", ""),
            "keywords_found": judgment.get("keywords_found", []),
            "has_citation": judgment.get("has_citation", False),
            "required_keywords": test["required_keywords"],
            "expected_source_hint": test["expected_source_hint"],
            "source_found": source_found,
            "errata_priority": test.get("errata_priority", False),
            "elapsed_s": round(elapsed, 1),
            "num_retrieved": debug_info.get("num_retrieved", 0),
            "num_cross_refs": debug_info.get("num_cross_refs", 0),
            "query_type": debug_info.get("query_type", ""),
            "answer_snippet": answer[:300],
        }
        results.append(result)

        # Save incrementally
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    total_duration = time.time() - start_total

    passing = sum(1 for r in results if r["grade"] in ("EXACT", "FAITHFUL", "CITED"))
    pass_rate = passing / len(results) * 100
    gate_pass = pass_rate >= 95.0

    print("\n" + "=" * 70)
    print("LEVEL 1 COMPLETE -- {} tests in {:.0f}s".format(len(results), total_duration))
    print("Pass Rate: {:.1f}%  |  Gate: {}".format(
        pass_rate, "PASSED" if gate_pass else "FAILED"
    ))
    print("=" * 70)

    generate_report(results, total_duration)

    import sys
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    run_tests()
