import engine.retrieval.rules_lawyer
rules_lawyer.load_game_profile('data/asl_profile.json')
ans, ctx, dbg = rules_lawyer.ask_rules_lawyer_game('Does a vehicle have to pay the stop MP penalty if it bogs down in a woods hex?')
print('\n==================')
print('ANSWER:')
print(ans)
print('==================')
print(f'Retrieved: {dbg.get("num_retrieved")}, Xrefs: {dbg.get("num_cross_refs")}')
