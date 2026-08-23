"""
Unit & Integration Tests for HyDE, SectionTree, and Ingestion Cooccurrence Graph.
"""

import os
import json
import pytest
from engine.models.cooccurrence_graph import CooccurrenceGraph, SectionTree, SectionNode
from engine.retrieval.hyde_generator import HydeGenerator
from engine.ingestion.ingest_rules import build_section_tree, build_ingestion_cooccurrence_graph, _extract_hierarchy_from_rule


def test_extract_hierarchy_from_rule():
    """Verify rule hierarchy extraction across decimal formats."""
    # Numeric decimal (Up Front)
    root, parent, level = _extract_hierarchy_from_rule("20.73")
    assert root == "20.0"
    assert parent == "20.7"
    assert level == 3

    root, parent, level = _extract_hierarchy_from_rule("20.7")
    assert root == "20.0"
    assert parent == "20.0"
    assert level == 2

    root, parent, level = _extract_hierarchy_from_rule("20.0")
    assert root == "20.0"
    assert parent == ""
    assert level == 1

    # Chapter decimal (ASL)
    root, parent, level = _extract_hierarchy_from_rule("A7.212")
    assert root == "A7.0"
    assert parent == "A7.2"
    assert level == 3


def test_cooccurrence_graph_operations(tmp_path):
    """Verify graph edge creation, thresholding, and JSON serialization."""
    graph = CooccurrenceGraph(game_id="test_game")
    graph.add_edge("15.2", "4.1", weight=0.88, relation_type="glossary_pmi", shared_terms=["squad", "cards"])
    graph.add_edge("15.2", "4.5", weight=0.75, relation_type="glossary_pmi", shared_terms=["draw"])
    graph.add_edge("15.2", "99.9", weight=0.30, relation_type="weak_association")

    neighbors = graph.get_neighbors("15.2", min_weight=0.50, limit=5)
    assert len(neighbors) == 2
    assert neighbors[0][0] == "4.1"
    assert neighbors[0][1] == 0.88
    assert neighbors[1][0] == "4.5"

    # Test file roundtrip
    out_file = os.path.join(tmp_path, "test_graph.json")
    graph.save_json(out_file)
    loaded = CooccurrenceGraph.load_json(out_file)
    assert loaded.game_id == "test_game"
    assert "15.2" in loaded.adjacency
    assert len(loaded.adjacency["15.2"]) == 3


def test_section_tree_operations(tmp_path):
    """Verify SectionTree hierarchy mapping, sibling lookups, and JSON serialization."""
    tree = SectionTree(game_id="test_game")
    tree.add_section(section_id="20.0", title="INFILTRATION", level=1)
    tree.add_section(section_id="20.7", title="Close Combat", parent_id="20.0", level=2)

    tree.register_rule("20.73", parent_section_id="20.7", chunk_id="chunk_20_73")
    tree.register_rule("20.74", parent_section_id="20.7", chunk_id="chunk_20_74")
    tree.register_rule("20.9", parent_section_id="20.0", chunk_id="chunk_20_9")

    parent = tree.get_parent_section("20.73")
    assert parent is not None
    assert parent.section_id == "20.7"

    siblings = tree.get_sibling_rules("20.73")
    assert "20.74" in siblings

    out_file = os.path.join(tmp_path, "test_tree.json")
    tree.save_json(out_file)
    loaded = SectionTree.load_json(out_file)
    assert "20.0" in loaded.sections
    assert loaded.rule_to_section_map.get("20.73") == "20.7"


def test_build_ingestion_cooccurrence_graph():
    """Verify generic ingestion graph builder on synthetic chunk corpus."""
    sample_chunks = {
        "chunk_1": {
            "rule_number": "17.7",
            "cross_refs": ["5.4", "7.3"],
            "text": "17.7 LATERAL GROUP TRANSFER: A squad may move into an adjacent column. See [5.4] and [7.3].",
            "metadata": {"root_section": "17.0", "parent_id": "17.0"}
        },
        "chunk_2": {
            "rule_number": "15.2",
            "cross_refs": [],
            "text": "15.2 SQUAD LEADER CASUALTY: When a Squad Leader is killed, the player card hand size decreases.",
            "metadata": {"root_section": "15.0", "parent_id": "15.0"}
        },
        "chunk_3": {
            "rule_number": "4.1",
            "cross_refs": [],
            "text": "4.1 SQUAD HAND SIZE: A player's card hand size depends on the alive Squad Leader status.",
            "metadata": {"root_section": "4.0", "parent_id": "4.0"}
        }
    }

    sample_index = {"17.7": [{"chunk_id": "chunk_1"}], "15.2": [{"chunk_id": "chunk_2"}], "4.1": [{"chunk_id": "chunk_3"}]}
    glossary = {"SL": "Squad Leader", "FP": "Firepower"}

    tree = build_section_tree(sample_chunks, game_id="test_game")
    graph = build_ingestion_cooccurrence_graph(sample_chunks, sample_index, glossary=glossary, section_tree=tree, game_id="test_game")

    # 17.7 should have cross-ref edge to 5.4 with weight 1.0
    neighbors_17 = graph.get_neighbors("17.7", min_weight=0.5)
    assert any(n[0] == "5.4" and n[1] == 1.0 for n in neighbors_17)

    # 15.2 and 4.1 share "squad leader", "hand size", "card" -> should have glossary PMI edge
    neighbors_15 = graph.get_neighbors("15.2", min_weight=0.4)
    assert any(n[0] == "4.1" for n in neighbors_15)


def test_hyde_generator_fallback():
    """Verify HydeGenerator fallback on empty/error input."""
    gen = HydeGenerator()
    fallback = gen.generate_pseudo_clause("", model="nonexistent_model")
    assert fallback == ""

    test_q = "Can a pinned squad fire?"
    # If model is unavailable or throws, should return distilled query cleanly
    result = gen.generate_pseudo_clause(test_q, model="nonexistent_model_xyz")
    assert result == test_q
