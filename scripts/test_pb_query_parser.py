import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)

from engine.retrieval.media_agent import parse_query_intent, extract_rule_based_filters

test_queries = [
    "Find all pictorials featuring Carmella DeCesare",
    "Show me blonde models reclining on a beach from 2006",
    "Who was the cover girl for the September 1990 Book of Lingerie?",
    "Find photoshoots by Arny Freytag in the 1980s",
    "Models with 34D measurements and brunette hair in lingerie",
    "What college girls appeared in the 2009 Big 10 issue?",
    "Find German special edition pictorials from 2016 or 2017",
    "Where was the Sara Stokes photoshoot located in Hot Housewives 2008?",
    "List all models in the Table of Contents of Wet and Wild 1996"
]

print("=== MEDIA AGENT QUERY PARSER PROBING ===\n")
for q in test_queries:
    parsed = parse_query_intent(q)
    print(f"Query: \"{q}\"")
    print(f"  Filters: {parsed.get('filters')}")
    print(f"  Semantic Text: {parsed.get('semantic_search_text')}")
    print()
