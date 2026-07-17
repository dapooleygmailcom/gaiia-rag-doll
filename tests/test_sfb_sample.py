from engine.retrieval import rules_lawyer

def run_tests():
    rules_lawyer.load_game_profile('data/star_fleet_battles_profile.json')

    tests = [
        ("EASY (Direct Rule)", "What is the rule (D2.3) for Emergency Deceleration?"),
        ("MEDIUM (Concept)", "How do Tractor Beams work in combat?"),
        ("HARD (Situation)", "A Battlecruiser has allocated 10 points of power to its shields. Does it receive a defense bonus from Electronic Warfare (EW) if it uses erratic maneuvering?"),
    ]

    for difficulty, query in tests:
        print(f"\n{'='*60}")
        print(f"{difficulty}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        ans, ctx, dbg = rules_lawyer.ask_rules_lawyer_game(query)
        
        # Safely print the answer
        safe_ans = ans.encode('ascii', errors='replace').decode('ascii')
        print("\nANSWER:")
        print(safe_ans)
        
        print("\n[DEBUG]")
        print(f"Type: {dbg.get('query_type')}")
        print(f"Retrieved: {dbg.get('num_retrieved')}")
        print(f"Cross-Refs: {dbg.get('num_cross_refs')}")
        
        # Print the top 2 sources used
        print("\n[TOP SOURCES]")
        for doc, meta in ctx[:2]:
            src = meta.get("source_file", "unknown")
            p = meta.get("priority", 9)
            rn = meta.get("rule_number", "")
            print(f" - {src} (P{p}) Rule: {rn}")
        
        print("\n" + "-"*60)

if __name__ == "__main__":
    run_tests()
