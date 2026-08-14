import sys
sys.path.append('.')
from engine.retrieval.rules_lawyer import ask_rules_lawyer_game

query = "In the planetary game of Prefecr, is it ever positively stated that Task Forces cannot contain units from different formations? I.e. is it possible for the RL player to have a task force that includes units from both 3099th legion and 199th legion?"
answer, context, debug = ask_rules_lawyer_game(query, profile_path="data/renegade_legion_profile.json")

print("\n--- ANSWER ---")
print(answer)
