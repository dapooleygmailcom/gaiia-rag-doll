"""
Ingestion Artifacts Builder — Gaiia RAG Doll.

Parses document text for a game profile, constructs the hierarchical SectionTree
and document-derived CooccurrenceGraph, and persists all ingestion artifacts.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.ingestion.ingest_rules import (
    load_profile,
    get_compiled_patterns,
    get_text_for_game_file,
    route_chunk_generic,
    build_rule_index,
    build_section_tree,
    build_ingestion_cooccurrence_graph
)


def build_artifacts_for_profile(profile_path: str):
    print("=" * 70)
    print(f"BUILDING INGESTION ARTIFACTS FOR: {profile_path}")
    print("=" * 70)

    profile = load_profile(profile_path)
    patterns = get_compiled_patterns(profile)
    game_id = profile.get("game_id", "generic")
    game_name = profile.get("game_name", "Unknown Game")
    rule_index_file = profile.get("rule_index_file", f"data/{game_id}_rule_index.json")

    cooc_path = profile.get("cooccurrence_graph_file") or rule_index_file.replace("_rule_index.json", "_cooccurrence_graph.json")
    sec_tree_path = profile.get("section_tree_file") or rule_index_file.replace("_rule_index.json", "_section_tree.json")

    all_chunks = {}
    docs_sorted = sorted(
        profile["documents"].items(),
        key=lambda x: x[1].get("priority", 9)
    )

    for fname, doc_info in docs_sorted:
        if doc_info.get("max_pages") == 0:
            continue
        doc_type = doc_info.get("doc_type", "unknown")
        priority = doc_info.get("priority", 9)

        text = get_text_for_game_file(fname, profile)
        if not text or len(text.strip()) < 50:
            continue

        chunks = route_chunk_generic(text, fname, doc_info, profile, patterns)
        print(f"  Parsed {fname} -> {len(chunks)} chunks")

        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_id = f"{doc_type}_{fname.replace('.pdf', '').replace(' ', '_').lower()}_chunk_{chunk_idx}"
            all_chunks[chunk_id] = chunk

    print(f"\nTotal Chunks Analyzed: {len(all_chunks)}")

    # 1. Build Rule Index
    rule_index = build_rule_index(all_chunks)

    # 2. Build Section Tree
    section_tree = build_section_tree(all_chunks, game_id=game_id)

    # 3. Build Co-occurrence Knowledge Graph
    glossary = profile.get("glossary", {})
    cooc_graph = build_ingestion_cooccurrence_graph(
        all_chunks=all_chunks,
        rule_index=rule_index,
        glossary=glossary,
        section_tree=section_tree,
        game_id=game_id
    )

    # Embed SectionTree in rule index
    rule_index["__section_tree__"] = section_tree.model_dump()

    # Save all artifacts
    with open(rule_index_file, "w", encoding="utf-8") as f:
        json.dump(rule_index, f, indent=2)

    cooc_graph.save_json(cooc_path)
    section_tree.save_json(sec_tree_path)

    total_edges = sum(len(edges) for edges in cooc_graph.adjacency.values())
    print("\n" + "=" * 70)
    print("ARTIFACTS GENERATION COMPLETE")
    print(f"  • Unique Rules in Index: {len(rule_index) - 1}")
    print(f"  • Hierarchical Sections: {len(section_tree.sections)}")
    print(f"  • Co-occurrence Graph Nodes: {len(cooc_graph.adjacency)}")
    print(f"  • Co-occurrence Graph Edges: {total_edges}")
    print(f"  • Rule Index File: {rule_index_file}")
    print(f"  • Co-occurrence Graph File: {cooc_path}")
    print(f"  • Section Tree File: {sec_tree_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ingestion Artifacts for a Game Profile")
    parser.add_argument("profile_path", nargs="?", default="data/up_front_profile.json", help="Path to game profile JSON")
    args = parser.parse_args()

    build_artifacts_for_profile(args.profile_path)
