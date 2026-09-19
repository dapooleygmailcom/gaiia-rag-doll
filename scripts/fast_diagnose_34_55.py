import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chromadb
import ollama

client = chromadb.PersistentClient("data/chroma")
col = client.get_collection("asl-rules-semantic")

with open("data/asl_rule_index.json", "r", encoding="utf-8") as f:
    rule_index = json.load(f)

from engine.retrieval.rules_lawyer import (
    load_game_profile,
    extract_keywords
)

load_game_profile("data/asl_profile.json")

def test_retrieval(query_id, query_text, expected_rules):
    print("=" * 80)
    print(f"QUERY {query_id}: {query_text}")
    print(f"EXPECTED RULES: {expected_rules}")
    print("=" * 80)

    # 1. Semantic query
    emb = ollama.embeddings(model="nomic-embed-text", prompt=query_text)["embedding"]
    results = col.query(query_embeddings=[emb], n_results=18, include=["documents", "metadatas", "distances"])
    
    print("\n--- Direct Semantic Search Top 10 in Chroma ---")
    for i in range(min(10, len(results["ids"][0]))):
        rn = results["metadatas"][0][i].get("rule_number", "")
        doc = results["documents"][0][i]
        dist = results["distances"][0][i]
        print(f"  [{i+1}] Rule: {rn} (dist={dist:.3f}) | {doc[:140].strip()}...")

    # 2. Check if expected rules are in rule_index and why exact lookup didn't trigger
    print("\n--- Exact Lookup & Keyword Relevance ---")
    keywords = extract_keywords(query_text)
    print(f"Keywords: {keywords}")
    
    for exp_r in expected_rules:
        in_index = exp_r in rule_index
        print(f"  Expected '{exp_r}' in rule_index: {in_index}")
        if in_index:
            for entry in rule_index[exp_r]:
                cid = entry["chunk_id"]
                chunk_res = col.get(ids=[cid])
                if chunk_res["documents"]:
                    cdoc = chunk_res["documents"][0]
                    kw_matches = [kw for kw in keywords if kw in cdoc.lower()]
                    print(f"    Chunk ID: {cid} | Keyword matches ({len(kw_matches)}): {kw_matches}")
                    print(f"    Snippet: {cdoc[:180].strip()}...")

if __name__ == "__main__":
    test_retrieval("34", "Advance phase and foxholes: Can a squad advance into an adjacent hex and into a foxhole in the same move?", ["A4.7", "B27.13"])
    test_retrieval("55", "Close Combat Sequence and Infiltration: So my opponent and I just completed a Close Combat. No ambush was obtained. I had 2:1 odds and rolled a 5, eliminating him, he rolled Snakes. He has the option now of withdrawing from the hex and no elimination occurs to either side, or he can stay and both participants are eliminated, correct?", ["A11.12", "A11.22"])
