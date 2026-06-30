"""
ASL Rules Lawyer Query Tests.

20 test cases covering 8 categories for the Advanced Squad Leader rules lawyer.
Mirrors the structure of test_rules_lawyer.py for Up Front.

DO NOT RUN until:
  1. auto_discover.py has been built and run on data/asl/
  2. ingest_rules.py has been made generic and run on data/asl/
  3. The rules_lawyer.py agent has been updated to be game-agnostic

The test structure here defines the expected benchmark so that we can
compare the ASL agent's performance to the Up Front baseline.

Categories:
  1. direct_rule         — Specific chapter-decimal rule lookup (e.g. A7.212)
  2. errata_supersession — Latest errata overrides core rulebook
  3. cross_reference     — EXC/parenthetical rule chains
  4. scenario            — Numeric scenario specific rules
  5. concept             — Game mechanics concept questions
  6. situation           — Ruling on an in-game situation
  7. multi_hop           — Combining a base rule with its scenario modifier
  8. qa_source           — Testing QA doc vs. official source hierarchy
"""

import json
import time
import os


# ═══════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════

ASL_TEST_CASES = [

    # ─── Category 1: Direct Rule Lookup ───
    {
        "id": 1,
        "category": "direct_rule",
        "query": "What does rule A7.212 say about entry of forces into the mapboard?",
        "expected_keywords": ["enter", "mapboard", "turn", "movement"],
        "expected_rule": "A7.212",
        "notes": "Core infantry movement entry rule. Should be in core_rules with priority 1."
    },
    {
        "id": 2,
        "category": "direct_rule",
        "query": "What is the basic infantry movement rule A4.1?",
        "expected_keywords": ["attacker", "move", "infantry", "fire", "broken"],
        "expected_rule": "A4.1",
        "notes": "Core MPh movement rule. Should cite the ASL 2nd edition core rules."
    },
    {
        "id": 3,
        "category": "direct_rule",
        "query": "What does rule D5.6 say about PRC survival checks?",
        "expected_keywords": ["prc", "survival", "eliminated", "vehicle"],
        "expected_rule": "D5.6",
        "notes": "Vehicle chapter rule. Tests detection of chapter-D rules."
    },

    # ─── Category 2: Errata Supersession ───
    {
        "id": 4,
        "category": "errata_supersession",
        "query": "What is the current definition of 'Armed' according to the latest errata?",
        "expected_keywords": ["armed", "gun", "functioning", "sw"],
        "expected_rule": None,
        "notes": (
            "The Dec 2025 errata adds 'not possessing a functioning Gun/SW' to the Armed definition. "
            "The agent must return the errata version, not the original definition."
        )
    },
    {
        "id": 5,
        "category": "errata_supersession",
        "query": "What is the correct definition of 'Squad Equivalent' after the latest errata?",
        "expected_keywords": ["squad", "equivalent", "non-inherent", "crew"],
        "expected_rule": None,
        "notes": (
            "Dec 2025 errata changes 'crews' to 'non-Inherent-crews'. "
            "Agent must prefer the errata version over the core rulebook."
        )
    },
    {
        "id": 6,
        "category": "errata_supersession",
        "query": "What does the latest Field Phone definition say, after corrections?",
        "expected_keywords": ["field", "phone", "ocg6"],
        "expected_rule": None,
        "notes": (
            "Dec 2025 errata changes 'O6' to 'OCG6' in the Field Phone definition. "
            "Agent must detect and apply the correction."
        )
    },

    # ─── Category 3: Cross-Reference Resolution ───
    {
        "id": 7,
        "category": "cross_reference",
        "query": "When a unit cannot enter a designated hex due to enemy occupation, what rules govern alternate entry?",
        "expected_keywords": ["A4.14", "entry", "blocked", "alternate"],
        "expected_rule": "A7.212",
        "notes": (
            "A7.212 contains an EXC referencing A4.14. "
            "Agent should chase the reference and include A4.14 content."
        )
    },
    {
        "id": 8,
        "category": "cross_reference",
        "query": "What is a Minimum Move and when can it be claimed? Include the terrain cost exception.",
        "expected_keywords": ["minimum", "move", "mf", "na", "not allowed"],
        "expected_rule": "A4.134",
        "notes": (
            "Minimum Move rule references NA terrain entry costs. "
            "Agent must pull both the rule and the NA definition."
        )
    },

    # ─── Category 4: Scenario-Specific ───
    {
        "id": 9,
        "category": "scenario",
        "query": "What are the victory conditions and special rules for Scenario A: The Guards Counterattack?",
        "expected_keywords": ["stalingrad", "german", "russian", "victory", "buildings"],
        "expected_rule": None,
        "notes": (
            "ScenarioA.pdf contains 'THE GUARDS COUNTERATTACK'. "
            "Agent should retrieve the scenario file and summarize conditions."
        )
    },
    {
        "id": 10,
        "category": "scenario",
        "query": "What are the forces and setup for Scenario B?",
        "expected_keywords": ["german", "russian", "setup", "hex"],
        "expected_rule": None,
        "notes": "ScenarioB.pdf — tests numeric/named scenario retrieval."
    },
    {
        "id": 11,
        "category": "scenario",
        "query": "What special balance adjustments apply to scenario A according to the Nov 2025 balance document?",
        "expected_keywords": ["balance", "adjustment"],
        "expected_rule": None,
        "notes": (
            "ASL_Scenario_Balance_Nov_2025.pdf — tests that the balance document "
            "is retrieved alongside the base scenario rules."
        )
    },

    # ─── Category 5: Concept Questions ───
    {
        "id": 12,
        "category": "concept",
        "query": "How does Bypass movement work in ASL and when is it blocked?",
        "expected_keywords": ["bypass", "hexside", "building", "woods", "blocked"],
        "expected_rule": "A4.31",
        "notes": "Core movement concept. Tests the agent can explain a multi-paragraph mechanic."
    },
    {
        "id": 13,
        "category": "concept",
        "query": "What is a Fire Group (FG) in ASL and how is it formed?",
        "expected_keywords": ["fire group", "fg", "firepower", "combine"],
        "expected_rule": None,
        "notes": "Fundamental concept question. Agent must synthesize across the core rules."
    },
    {
        "id": 14,
        "category": "concept",
        "query": "What does CX mean in ASL and how does it affect movement?",
        "expected_keywords": ["cx", "encumbered", "movement", "mf", "hmg"],
        "expected_rule": None,
        "notes": "CX (carrying extra) is illustrated in the core rules movement chapter."
    },

    # ─── Category 6: Situation-Based Rulings ───
    {
        "id": 15,
        "category": "situation",
        "query": "A Russian squad with an HMG is CX and lacks 4 MF to enter a hex. Can they still enter it?",
        "expected_keywords": ["minimum", "move", "yes", "mf", "cx"],
        "expected_rule": "A4.134",
        "notes": (
            "This is the exact example from the core rules minimum move section. "
            "Agent must recognize the CX squad scenario and apply the Minimum Move rule."
        )
    },
    {
        "id": 16,
        "category": "situation",
        "query": "A vehicle is hit and the Final DR equals the Kill Number. What happens to the vehicle and its PRC?",
        "expected_keywords": ["immobilized", "hd", "unaffected", "prc"],
        "expected_rule": "A7.308",
        "notes": "Vehicle elimination table rule. Tests correct application of a conditional outcome."
    },
    {
        "id": 17,
        "category": "situation",
        "query": "Two wadi overlays on different boards are adjacent to each other. Are their common hexsides treated as wadi hexsides?",
        "expected_keywords": ["yes", "wadi", "adjacent", "hexside"],
        "expected_rule": "A2.76",
        "notes": "Tests retrieval of an edge-case terrain rule about overlay adjacency."
    },

    # ─── Category 7: Multi-Hop ───
    {
        "id": 18,
        "category": "multi_hop",
        "query": (
            "In Scenario A, the German forces set up first. "
            "What does the base infantry setup rule say, and how does this scenario modify the standard procedure?"
        ),
        "expected_keywords": ["german", "sets up first", "scenario", "setup"],
        "expected_rule": None,
        "notes": (
            "Multi-hop: requires combining the base setup rules with the scenario-specific "
            "GERMAN Sets Up First instruction from ScenarioA.pdf."
        )
    },
    {
        "id": 19,
        "category": "multi_hop",
        "query": (
            "A unit performs a Bypass move past a building that touches a wall/hedge depiction. "
            "Is Bypass blocked, and what rules govern this interaction?"
        ),
        "expected_keywords": ["bypass", "blocked", "wall", "hedge", "building", "hexside"],
        "expected_rule": "A4.31",
        "notes": (
            "A4.31 explains that walls/hedges are extensions of hexsides. "
            "Tests multi-hop: Bypass rule + terrain interaction exception."
        )
    },

    # ─── Category 8: QA Source Hierarchy ───
    {
        "id": 20,
        "category": "qa_source",
        "query": "According to the Q&A document, what is the official source for each clarification, and which items are unofficial?",
        "expected_keywords": ["official", "unofficial", "source", "square brackets"],
        "expected_rule": None,
        "notes": (
            "The QA document header explicitly states that items from unofficial sources "
            "are NOT official. Agent must acknowledge this hierarchy. "
            "Tests the agent's ability to reason about the authority stack itself."
        )
    },
]


# ═══════════════════════════════════════════════════════════════════
# NOTE: DO NOT RUN — Framework Only
# ═══════════════════════════════════════════════════════════════════

def validate_test_structure():
    """
    Validates the test case structure without running the agent.
    Call this to confirm all test cases are well-formed.
    """
    required_keys = {"id", "category", "query", "expected_keywords", "notes"}
    valid_categories = {
        "direct_rule", "errata_supersession", "cross_reference",
        "scenario", "concept", "situation", "multi_hop", "qa_source"
    }

    ids = set()
    errors = []

    for tc in ASL_TEST_CASES:
        tc_id = tc.get("id", "?")

        # Check required keys
        missing = required_keys - set(tc.keys())
        if missing:
            errors.append(f"Test {tc_id}: missing keys {missing}")

        # Check unique IDs
        if tc_id in ids:
            errors.append(f"Duplicate ID: {tc_id}")
        ids.add(tc_id)

        # Check valid category
        cat = tc.get("category")
        if cat not in valid_categories:
            errors.append(f"Test {tc_id}: unknown category '{cat}'")

        # Check keywords is a list
        kws = tc.get("expected_keywords", [])
        if not isinstance(kws, list):
            errors.append(f"Test {tc_id}: expected_keywords must be a list")

        # Check query is non-empty
        if not tc.get("query", "").strip():
            errors.append(f"Test {tc_id}: empty query")

    return errors


def print_test_summary():
    """Print a readable summary of all test cases."""
    from collections import Counter
    cats = Counter(tc["category"] for tc in ASL_TEST_CASES)

    print("=" * 60)
    print("ASL RULES LAWYER QUERY TESTS — SUMMARY")
    print(f"Total tests: {len(ASL_TEST_CASES)}")
    print("=" * 60)
    print("\nBy Category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat:25s}: {count}")

    errors = validate_test_structure()
    print(f"\nValidation: {'PASS' if not errors else 'FAIL'}")
    for e in errors:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    print_test_summary()
    # Save as JSON for reference
    os.makedirs("data/logs", exist_ok=True)
    with open("data/logs/asl_test_cases.json", "w", encoding="utf-8") as f:
        json.dump(ASL_TEST_CASES, f, indent=2)
    print(f"\nTest cases saved to data/logs/asl_test_cases.json")
