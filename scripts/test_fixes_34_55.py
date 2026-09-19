import sys
import os
import json
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chromadb
import ollama

from engine.retrieval.rules_lawyer import (
    load_game_profile,
    _get_active_collection,
    _load_active_rule_index,
    _load_active_section_tree,
    _load_active_cooccurrence_graph,
    extract_keywords,
    build_game_prompt
)

load_game_profile("data/asl_profile.json")

collection = _get_active_collection()
rule_index = _load_active_rule_index()
section_tree = _load_active_section_tree()

print("Testing enhanced section expansion...")

def test_enhanced_expansion_34():
    query = "Advance phase and foxholes: Can a squad advance into an adjacent hex and into a foxhole in the same move?"
    # Find candidate rules from semantic search
    emb = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    res = collection.query(query_embeddings=[emb], n_results=10, include=["documents", "metadatas"])
    
    metas = res["metadatas"][0]
    candidate_rules = [m.get("rule_number") for m in metas if m.get("rule_number")]
    print("Initial candidate rules:", candidate_rules)

    # Expand parents, siblings, AND children
    expanded_chunk_ids = set()
    expanded_rules = set()

    for r in candidate_rules:
        # Check rule prefix matching in rule_index (e.g. B27.1 -> B27.11, B27.12, B27.13)
        parts = r.split('.')
        base = parts[0]
        # Chapter / Section prefix expansion
        prefix_matches = [k for k in rule_index.keys() if k.startswith(r) or k.startswith(f"{base}.1") or k.startswith(f"{base}.7")]
        for pm in prefix_matches[:8]:
            expanded_rules.add(pm)
            for entry in rule_index[pm][:2]:
                expanded_chunk_ids.add(entry["chunk_id"])

        # Section tree node expansion
        sec_node = section_tree.sections.get(r) or section_tree.sections.get(f"{r}.0")
        if sec_node:
            for cid in sec_node.chunk_ids[:4]:
                expanded_chunk_ids.add(cid)
            for cr in sec_node.child_rules:
                expanded_rules.add(cr)
                if cr in rule_index:
                    for entry in rule_index[cr][:2]:
                        expanded_chunk_ids.add(entry["chunk_id"])

    print(f"Total expanded rules ({len(expanded_rules)}):", sorted(list(expanded_rules))[:15])
    print(f"Total expanded chunk IDs ({len(expanded_chunk_ids)}):", list(expanded_chunk_ids)[:10])
    
    # Verify if chunk_825 (B27.13) is in expanded chunk IDs
    b27_13_present = any("chunk_825" in cid for cid in expanded_chunk_ids)
    print("Is B27.13 (chunk_825) present in expanded chunks?", b27_13_present)

if __name__ == "__main__":
    test_enhanced_expansion_34()
