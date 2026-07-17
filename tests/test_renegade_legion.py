"""
Smoke test for Renegade Legion wargame rules.
"""

from engine.retrieval.rules_lawyer import ask_rules_lawyer_game

QUERIES = [
    {
        "desc": "CONCEPT TEST",
        "query": "What is the Terran Overlord Government (TOG)?"
    },
    {
        "desc": "GAMEPLAY MECHANIC TEST",
        "query": "How does movement or turning work for grav vehicles?"
    }
]

def run_tests():
    print("=" * 60)
    print("RENEGADE LEGION SMOKE TEST")
    print("=" * 60)
    
    for q in QUERIES:
        print(f"\n\n============================================================")
        print(f"{q['desc']}")
        print(f"QUERY: {q['query']}")
        print(f"============================================================\n")
        
        answer, context_chunks, debug_info = ask_rules_lawyer_game(
            q['query'], profile_path="data/renegade_legion_profile.json"
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
