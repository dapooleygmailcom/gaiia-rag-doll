import engine.retrieval.rules_lawyer

rules_lawyer.load_game_profile('data/asl_profile.json')

tests = [
    ("EASY (Direct Rule)", "What is the basic infantry movement rule A4.1?"),
    ("MEDIUM (Cross Reference)", "What is a Minimum Move and when can it be claimed? Include the terrain cost exception."),
    ("HARD (Errata Supersession)", "What is the current definition of 'Armed' according to the latest errata?"),
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
