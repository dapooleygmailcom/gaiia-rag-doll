"""
Secondary Rules Adjudicator — Gaiia RAG Doll.
Judges semantic and substantive correctness of full retrieval/citation misses
and partial citation gaps against ground truth community rulings, sandboxing original metrics.
"""
import re
import json
import time
import ollama

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


def adjudicate_answer(query: str, ground_truth: str, candidate_answer: str, judge_model: str = "llama3.1:8b") -> dict:
    """
    Adjudicate a candidate answer against ground truth.
    Returns structured adjudication payload.
    """
    if not candidate_answer or not ground_truth:
        return {
            "adjudicated": False,
            "judge_model": judge_model,
            "ruling": "SKIPPED",
            "substantive_accuracy": False,
            "confidence": "LOW",
            "rationale": "Missing candidate answer or ground truth answer.",
            "latency_seconds": 0.0
        }

    prompt = (
        JUDGE_PROMPT_TEMPLATE
        .replace("{query}", query.strip())
        .replace("{ground_truth}", ground_truth.strip())
        .replace("{candidate_answer}", candidate_answer.strip())
    )

    t0 = time.time()
    try:
        response = ollama.generate(
            model=judge_model,
            prompt=prompt,
            options={"temperature": 0.0, "top_p": 0.9}
        )
        latency = round(time.time() - t0, 2)
        raw = response.get("response", "").strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            ruling = str(data.get("ruling", "DISAGREE")).upper()
            substantive_accuracy = bool(data.get("substantive_accuracy", ruling == "AGREE"))
            confidence = str(data.get("confidence", "MEDIUM")).upper()
            rationale = str(data.get("rationale", "")).strip()
        else:
            ruling = "AGREE" if "agree" in raw.lower() and "disagree" not in raw.lower() else "DISAGREE"
            substantive_accuracy = ruling == "AGREE"
            confidence = "LOW"
            rationale = raw[:200]

        return {
            "adjudicated": True,
            "judge_model": judge_model,
            "ruling": ruling,
            "substantive_accuracy": substantive_accuracy,
            "confidence": confidence,
            "rationale": rationale,
            "latency_seconds": latency
        }
    except Exception as e:
        return {
            "adjudicated": False,
            "judge_model": judge_model,
            "ruling": "ERROR",
            "substantive_accuracy": False,
            "confidence": "NONE",
            "rationale": f"Adjudication error: {e}",
            "latency_seconds": round(time.time() - t0, 2)
        }
