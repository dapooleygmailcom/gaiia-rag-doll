import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)

from engine.retrieval.media_agent import MediaAgent

agent = MediaAgent()

queries = [
    "how many special issues is Alley Baggett present?",
    "Find all appearances and pictorials of Alley Baggett",
    "Show me pictorials featuring Alley Baggett"
]

print("=== EVALUATION TEST CASE: ALLEY BAGGETT ===")
for q in queries:
    print(f"\n--- QUERY: {q} ---")
    res = agent.search(q, top_k=5)
    print(res)
