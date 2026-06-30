"""
Smoke test for Warhammer 40K semantic ingestion and multi-edition conflict resolution.
"""

from engine.retrieval.rules_lawyer import ask_rules_lawyer_game

QUERIES = [
    {
        "desc": "CROSS-EDITION LOGIC (Testing Edition fallback & Follow-up question)",
        "query": "How far can a Space Marine move in a turn?"
    },
    {
        "desc": "STAT BLOCK (Testing tabular parsing of Datasheets)",
        "query": "What is the Toughness and Wounds of a Space Marine Intercessor?"
    },
    {
        "desc": "RULE LOOKUP (Testing semantic header chunking)",
        "query": "Can a unit shoot in the same turn it Fell Back?"
    }
]

def run_tests():
    print("=" * 60)
    print("WARHAMMER 40K SMOKE TEST")
    print("=" * 60)
    
    for q in QUERIES:
        print(f"\n\n============================================================")
        print(f"{q['desc']}")
        print(f"QUERY: {q['query']}")
        print(f"============================================================\n")
        
        answer, context_chunks, debug_info = ask_rules_lawyer_game(
            q['query'], profile_path="data/warhammer_40k_profile.json"
        )
        
        print("ANSWER:\n")
        safe_answer = answer.encode("ascii", errors="replace").decode("ascii")
        print(safe_answer)
        
        print("\n[DEBUG]")
        print(f"Type: {debug_info.get('query_type')}")
        print(f"Retrieved: {debug_info.get('num_retrieved')}")
        print(f"Cross-Refs: {debug_info.get('num_cross_refs')}")
        
        print("\n[TOP SOURCES]")
        for doc, meta in context_chunks[:3]:
            source = meta.get("source_file", "unknown")
            edition = meta.get("edition", "N/A")
            print(f" - {source} (Edition: {edition})")
        print("\n" + "-"*60)

if __name__ == "__main__":
    run_tests()
