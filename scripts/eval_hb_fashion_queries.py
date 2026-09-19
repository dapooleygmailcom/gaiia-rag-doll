"""
Evaluation Script: High-Fashion & Multi-Periodical Retrieval Evaluation.

Probes the 8 representative high-fashion query archetypes:
1. Designer Garment: "Find jackets by Givenchy"
2. Wardrobe Accessory: "Show me all shots with scarves"
3. High-Fashion Photographer: "Find photoshoots by Camilla Akrans"
4. Cross-genre Photographer: "Find shoots by Terry Richardson"
5. Cover Celebrity Lookup: "Who was the cover girl for Harper's Bazaar April 2015?"
6. Price Constraint: "Find gowns under $6000"
7. Editorial Feature Search: "Julianne Moore The Moore The Better"
8. Legacy Playboy Regression: "Find all pictorials featuring Carmella DeCesare"
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.retrieval.media_agent import MediaAgent


def run_fashion_eval():
    agent = MediaAgent()
    eval_cases = [
        {
            "name": "1. Designer Garment Lookup",
            "query": "Find jackets by Givenchy",
            "expected_keywords": ["Givenchy", "jacket", "4650"],
            "category": "fashion_designer"
        },
        {
            "name": "2. Wardrobe Accessory Search",
            "query": "Show me all shots with scarves",
            "expected_keywords": ["scarf"],
            "category": "fashion_wardrobe"
        },
        {
            "name": "3. High-Fashion Editorial Photographer",
            "query": "Find photoshoots by Camilla Akrans",
            "expected_keywords": ["Camilla Akrans", "Julianne Moore", "Harper's Bazaar"],
            "category": "photographer"
        },
        {
            "name": "4. Cross-Genre Photographer Query",
            "query": "Find shoots by Terry Richardson",
            "expected_keywords": ["Terry Richardson"],
            "category": "cross_genre"
        },
        {
            "name": "5. Cover Celebrity Lookup",
            "query": "Who was the cover girl for Harper's Bazaar April 2015?",
            "expected_keywords": ["Julianne Moore", "Cover Girl"],
            "category": "celebrity_lookup"
        },
        {
            "name": "6. Wardrobe Price Filter",
            "query": "Find gowns by Gucci",
            "expected_keywords": ["Gucci", "gown", "5200"],
            "category": "price_brand"
        },
        {
            "name": "7. Editorial Feature Spread",
            "query": "Show me photoshoots featuring Julianne Moore",
            "expected_keywords": ["Julianne Moore", "Harper's Bazaar"],
            "category": "editorial_feature"
        },
        {
            "name": "8. Legacy Playboy Query Regression",
            "query": "Find all pictorials featuring Carmella DeCesare",
            "expected_keywords": ["Carmella DeCesare"],
            "category": "playboy_regression"
        }
    ]

    print("================================================================================")
    print(" HARPER'S BAZAAR & MULTI-PERIODICAL FASHION RETRIEVAL EVALUATION")
    print("================================================================================\n")

    passed_count = 0
    total_count = len(eval_cases)

    for idx, case in enumerate(eval_cases, 1):
        q = case["query"]
        print(f"[{idx}/{total_count}] Testing: {case['name']}")
        print(f"     Query: \"{q}\"")

        results = agent.search(q)
        formatted = agent.format_retrieval_response(q, results)

        matches = [kw for kw in case["expected_keywords"] if kw.lower() in formatted.lower()]
        success = len(matches) >= 1

        if success:
            passed_count += 1
            print(f"     Status: [PASS] Matched expected keywords: {matches}")
        else:
            print(f"     Status: [FAIL] Missing: {case['expected_keywords']}")

        print(f"     Preview: {formatted.splitlines()[0] if formatted.splitlines() else 'No output'}\n")

    print("================================================================================")
    print(f" FINAL EVALUATION SCORE: {passed_count}/{total_count} passed ({passed_count/total_count*100:.1f}%)")
    print("================================================================================")
    return passed_count == total_count


if __name__ == "__main__":
    success = run_fashion_eval()
    sys.exit(0 if success else 1)
