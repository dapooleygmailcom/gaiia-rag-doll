import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)

from engine.retrieval.media_agent import MediaAgent

agent = MediaAgent()

test_queries = [
    "Find all pictorials featuring Laurie Carr",
    "Show me pictorials featuring Sara Stokes",
    "Find appearances of Lonny Chin",
    "Who was the cover girl of Vixens in 2006?",
    "Find appearances of Tiffany Richardson",
    "Find photoshoots by Mizuno"
]

print("=== TESTING MEDIA AGENT DUAL RETRIEVAL ON GENERIFIED STAGE 1 INDEX ===\n")
for q in test_queries:
    res = agent.search(q, top_k=3)
    print(f"Query: \"{q}\"")
    print(res)
    print("=" * 60 + "\n")
