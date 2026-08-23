"""
Rules Lawyer Test Suite.

Tests the Rules Lawyer agent across 8 categories:
  1. Direct rule lookup
  2. Errata supersession
  3. Cross-reference resolution
  4. Scenario-specific queries
  5. Concept questions
  6. Situation-based rulings
  7. Variant awareness
  8. Multi-hop queries

Uses gemma2:9b as impartial judge (same pattern as policy_agent tests).
"""

import json
import time
import os
import ollama
from engine.retrieval.rules_lawyer import ask_rules_lawyer_game, load_game_profile

os.makedirs("data/logs", exist_ok=True)
LOG_FILE = "data/logs/rules_lawyer_test_results.json"

# ═══════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════

RULES_TEST_CASES = [
    # ─── Category 1: Direct Rule Lookup ───
    {
        "id": 1,
        "category": "direct_rule",
        "query": "What does rule 5.41 say about terrain placement?",
        "expected_keywords": ["terrain", "played", "accepted", "movement"],
        "expected_rule": "5.41"
    },
    {
        "id": 2,
        "category": "direct_rule",
        "query": "What is rule 6.1 about fire attacks?",
        "expected_keywords": ["fire card", "firepower", "minimum"],
        "expected_rule": "6.1"
    },
    {
        "id": 3,
        "category": "direct_rule",
        "query": "Explain rule 10.45 about Hero cards.",
        "expected_keywords": ["hero", "firepower", "double"],
        "expected_rule": "10.45"
    },

    # ─── Category 2: Errata Supersession ───
    {
        "id": 4,
        "category": "errata_supersession",
        "query": "Can a player with multi-card discard capability discard one terrain card on a group, have it rejected, and still discard another terrain card on the same group in the same turn?",
        "expected_keywords": ["no"],
        "expected_rule": "7.32"
    },
    {
        "id": 5,
        "category": "errata_supersession",
        "query": "Can you stop Fire Combat Resolution partway through a target group?",
        "expected_keywords": ["no"],
        "expected_rule": "6.5"
    },
    {
        "id": 6,
        "category": "errata_supersession",
        "query": "Does the attacker have the option of foregoing the resolution of an ordnance hit after seeing the Final Strength Number?",
        "expected_keywords": ["no", "cancelled", "to hit"],
        "expected_rule": "6.5"
    },

    # ─── Category 3: Cross-Reference Resolution ───
    {
        "id": 7,
        "category": "cross_reference",
        "query": "When can a group rearrange their order? What rules govern this?",
        "expected_keywords": ["3.3", "4.25", "11.12", "18.2", "rearrange"],
        "expected_rule": "11.11"
    },
    {
        "id": 8,
        "category": "cross_reference",
        "query": "How does lateral distance affect relative range? Include the related blocking rules.",
        "expected_keywords": ["5.61", "decreased", "one", "adjacent"],
        "expected_rule": "5.61"
    },

    # ─── Category 4: Scenario-Specific ───
    {
        "id": 9,
        "category": "scenario",
        "query": "What are the special rules for Scenario A: Meeting of Patrols?",
        "expected_keywords": ["pillbox", "minefield", "cower"],
        "expected_rule": None
    },
    {
        "id": 10,
        "category": "scenario",
        "query": "What cards are treated as Cower cards in Scenario C for the attacker?",
        "expected_keywords": ["minefield", "sniper", "attacker", "cower"],
        "expected_rule": None
    },
    {
        "id": 11,
        "category": "scenario",
        "query": "What are the victory conditions for Scenario B: City Fight?",
        "expected_keywords": ["victory points", "buildings", "time limit"],
        "expected_rule": None
    },

    # ─── Category 5: Concept Questions ───
    {
        "id": 12,
        "category": "concept",
        "query": "How does the Japanese discard phase work compared to the American discard phase?",
        "expected_keywords": ["japanese", "american", "discard"],
        "expected_rule": None
    },
    {
        "id": 13,
        "category": "concept",
        "query": "How does relative range work in Up Front?",
        "expected_keywords": ["range", "movement", "0", "5"],
        "expected_rule": "5.5"
    },
    {
        "id": 14,
        "category": "concept",
        "query": "What is a Personality card and what information does it contain?",
        "expected_keywords": ["man", "firepower", "morale", "panic"],
        "expected_rule": None
    },

    # ─── Category 6: Situation-Based Rulings ───
    {
        "id": 15,
        "category": "situation",
        "query": "Can you hero an unpinned man when the group is not firing, just to get it out of your hand?",
        "expected_keywords": ["no"],
        "expected_rule": "10.45"
    },
    {
        "id": 16,
        "category": "situation",
        "query": "If a group has pinned men, can they reject terrain?",
        "expected_keywords": ["yes"],
        "expected_rule": "7.32"
    },
    {
        "id": 17,
        "category": "situation",
        "query": "Can a PC whose primary weapon is a crew-served weapon act as a crewman for another crew-served weapon?",
        "expected_keywords": ["no", "exception", "unarmed"],
        "expected_rule": "11.11"
    },

    # ─── Category 7: Variant Awareness ───
    {
        "id": 18,
        "category": "variant",
        "query": "Are there any variant or house rules for Up Front?",
        "expected_keywords": ["variant", "house", "experimental"],
        "expected_rule": None
    },

    # ─── Category 8: Multi-Hop ───
    {
        "id": 19,
        "category": "multi_hop",
        "query": "In Scenario C, the attacker has Sniper cards treated as Cower cards. What does the base Sniper rule say, and how does this scenario modify it?",
        "expected_keywords": ["sniper", "cower", "scenario c", "attacker"],
        "expected_rule": "14.2"
    },
    {
        "id": 20,
        "category": "multi_hop",
        "query": "What is the full rule for crewed weapons including how Heroes interact with crewed LMGs?",
        "expected_keywords": ["crewed", "hero", "bracketed", "doubled", "17.4"],
        "expected_rule": "10.45"
    },
    {
        "id": 21,
        "category": "multi_hop",
        "query": "Quote: 17.1 LATERAL FLANKING FIRE: Flanking Fire can originate at any Relative Range from a group whose last placed \"Flank\" Movement card was played sideways and remains in play at the end of the turn in which it was played. Does the flanking unit suffer from moving fire penalty? Does flanking stop once the flanking unit enters some terrain?",
        "expected_keywords": ["moving fire", "halved", "doubled", "17.4", "17.41", "terrain"],
        "expected_rule": "17.41"
    },
]


# ═══════════════════════════════════════════════════════════════════
# Judge
# ═══════════════════════════════════════════════════════════════════

def judge_response(question, answer, context, expected_keywords, expected_rule):
    """Use gemma2:9b to judge faithfulness and relevancy."""
    keyword_section = ""
    if expected_keywords:
        keyword_section = f"\nExpected keywords/concepts: {', '.join(expected_keywords)}"
    
    rule_section = ""
    if expected_rule:
        rule_section = f"\nExpected rule citation: {expected_rule}"
    
    prompt = f"""You are an impartial judge evaluating an AI assistant's response in a wargame rules RAG system.
Evaluate the response based on three criteria:
1. Faithfulness: Is the answer strictly derived from the provided context? (Does it hallucinate?)
2. Relevancy: Does the answer directly address the user's rules question?
3. Citation: Does the answer cite specific rule numbers and source documents?

Context: {context[:3000]}
Question: {question}
Answer: {answer[:2000]}{keyword_section}{rule_section}

Output exactly ONE line in the following format:
Faithfulness: [Yes/No], Relevancy: [Yes/No], Citation: [Yes/No]"""

    try:
        response = ollama.generate(model="gemma2:9b", prompt=prompt)
        verdict = response['response'].strip()
        return verdict
    except Exception as e:
        print(f"Error calling judge: {e}")
        return "Faithfulness: Error, Relevancy: Error, Citation: Error"


def parse_verdict(verdict_text):
    """Parse the judge's verdict into individual scores."""
    v = verdict_text.lower()
    
    faithfulness = "Yes" if "faithfulness: yes" in v else "No"
    relevancy = "Yes" if "relevancy: yes" in v else "No"
    citation = "Yes" if "citation: yes" in v else "No"
    
    return faithfulness, relevancy, citation


# ═══════════════════════════════════════════════════════════════════
# Keyword Check
# ═══════════════════════════════════════════════════════════════════

def check_keywords(answer, expected_keywords):
    """Check if expected keywords appear in the answer."""
    answer_lower = answer.lower()
    found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    return found, missing


# ═══════════════════════════════════════════════════════════════════
# Main Test Runner
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("RULES LAWYER TEST SUITE (AGNOSTIC UP FRONT)")
    print(f"Total test cases: {len(RULES_TEST_CASES)}")
    print("=" * 60)
    
    # Load the agnostic Up Front profile!
    load_game_profile("data/up_front_profile.json")
    
    results = []
    scores = {
        "total": len(RULES_TEST_CASES),
        "faithful": 0,
        "relevant": 0,
        "cited": 0,
        "perfect": 0,
        "keyword_hits": 0,
        "keyword_total": 0
    }
    
    # Category tracking
    category_scores = {}
    completed_ids = set()
    
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Resuming from {len(results)} completed tests...")
            for res in results:
                completed_ids.add(res["id"])
                
                is_faithful = (res["faithfulness"] == "Yes")
                is_relevant = (res["relevancy"] == "Yes")
                is_cited = (res["citation"] == "Yes")
                is_perfect = is_faithful and is_relevant and is_cited
                
                if is_faithful: scores["faithful"] += 1
                if is_relevant: scores["relevant"] += 1
                if is_cited: scores["cited"] += 1
                if is_perfect: scores["perfect"] += 1
                
                scores["keyword_hits"] += len(res.get("keywords_found", []))
                scores["keyword_total"] += len(res.get("keywords_found", [])) + len(res.get("keywords_missing", []))
                
                cat = res["category"]
                if cat not in category_scores:
                    category_scores[cat] = {"total": 0, "perfect": 0}
                category_scores[cat]["total"] += 1
                if is_perfect:
                    category_scores[cat]["perfect"] += 1
        except Exception as e:
            print(f"Could not load previous results: {e}")
            results = []
            completed_ids = set()
            
    start_time = time.time()
    
    for idx, tc in enumerate(RULES_TEST_CASES, 1):
        tc_id = tc["id"]
        if tc_id in completed_ids:
            continue
            
        category = tc["category"]
        query = tc["query"]
        expected_kw = tc.get("expected_keywords", [])
        expected_rule = tc.get("expected_rule")
        
        print(f"\n{'-'*60}")
        print(f"[{idx}/{len(RULES_TEST_CASES)}] ID: {tc_id} | Category: {category}")
        print(f"Query: {query}")
        
        case_start = time.time()
        
        # Run the Rules Lawyer
        try:
            answer, context, debug = ask_rules_lawyer_game(query)
        except Exception as e:
            print(f"  ERROR: {e}")
            answer, context, debug = f"ERROR: {e}", "", {}
        
        case_duration = time.time() - case_start
        
        # Judge the response
        verdict = judge_response(query, answer, context, expected_kw, expected_rule)
        faithful, relevant, cited = parse_verdict(verdict)
        
        # Keyword check
        kw_found, kw_missing = check_keywords(answer, expected_kw)
        scores["keyword_hits"] += len(kw_found)
        scores["keyword_total"] += len(expected_kw)
        
        # Score
        is_faithful = (faithful == "Yes")
        is_relevant = (relevant == "Yes")
        is_cited = (cited == "Yes")
        is_perfect = is_faithful and is_relevant and is_cited
        
        if is_faithful:
            scores["faithful"] += 1
        if is_relevant:
            scores["relevant"] += 1
        if is_cited:
            scores["cited"] += 1
        if is_perfect:
            scores["perfect"] += 1
        
        # Category tracking
        if category not in category_scores:
            category_scores[category] = {"total": 0, "perfect": 0}
        category_scores[category]["total"] += 1
        if is_perfect:
            category_scores[category]["perfect"] += 1
        
        print(f"  Time: {case_duration:.1f}s")
        print(f"  Verdict: {verdict}")
        print(f"  Keywords found: {kw_found} | Missing: {kw_missing}")
        
        results.append({
            "id": tc_id,
            "category": category,
            "query": query,
            "answer": answer,
            "context": context[:2000],
            "debug": debug,
            "verdict": verdict,
            "faithfulness": faithful,
            "relevancy": relevant,
            "citation": cited,
            "keywords_found": kw_found,
            "keywords_missing": kw_missing,
            "duration_seconds": round(case_duration, 2)
        })
        
        # Save intermediate
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    
    total_duration = time.time() - start_time
    
    # ─── Final Report ───
    print(f"\n{'='*60}")
    print("RULES LAWYER TEST RESULTS")
    print(f"Total Time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"{'='*60}")
    
    total = scores["total"]
    print(f"\n### Overall Scores\n")
    print(f"| Metric | Passes | Total | Rate |")
    print(f"|--------|--------|-------|------|")
    print(f"| Faithfulness | {scores['faithful']} | {total} | {scores['faithful']/total*100:.1f}% |")
    print(f"| Relevancy | {scores['relevant']} | {total} | {scores['relevant']/total*100:.1f}% |")
    print(f"| Citation | {scores['cited']} | {total} | {scores['cited']/total*100:.1f}% |")
    print(f"| **Perfect** | **{scores['perfect']}** | **{total}** | **{scores['perfect']/total*100:.1f}%** |")
    
    if scores["keyword_total"] > 0:
        kw_rate = scores["keyword_hits"] / scores["keyword_total"] * 100
        print(f"| Keyword Recall | {scores['keyword_hits']} | {scores['keyword_total']} | {kw_rate:.1f}% |")
    
    print(f"\n### By Category\n")
    print(f"| Category | Perfect | Total | Rate |")
    print(f"|----------|---------|-------|------|")
    for cat, cat_scores in sorted(category_scores.items()):
        rate = cat_scores["perfect"] / cat_scores["total"] * 100
        print(f"| {cat} | {cat_scores['perfect']} | {cat_scores['total']} | {rate:.1f}% |")
    
    # Save summary
    summary = {
        "total_duration_minutes": round(total_duration / 60, 2),
        "scores": scores,
        "category_scores": category_scores
    }
    with open("data/logs/rules_lawyer_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nLogs saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
