"""
Co-Occurrence Knowledge Graph & Section Hierarchy Models — Gaiia RAG Doll.

Provides data structures for representing:
1. Document-derived rule co-occurrence graphs (cross-refs, glossary PMI, structural adjacency).
2. Hierarchical section trees (parent-child containers, sibling clauses).
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field


class CooccurrenceEdge(BaseModel):
    source_rule: str
    target_rule: str
    weight: float = Field(ge=0.0, le=1.0)
    relation_type: str = Field(description="cross_reference | glossary_pmi | structural_sibling | parent_child")
    shared_terms: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class CooccurrenceGraph(BaseModel):
    """
    Represents an offline or ingestion-derived knowledge graph of rule relationships.
    """
    game_id: str = "generic"
    version: str = "1.0"
    adjacency: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_edge(self, source: str, target: str, weight: float, relation_type: str, shared_terms: Optional[List[str]] = None, description: Optional[str] = None):
        """Add or update an edge between source and target rule IDs."""
        source = str(source).strip()
        target = str(target).strip()
        if not source or not target or source == target:
            return

        if source not in self.adjacency:
            self.adjacency[source] = []

        # Check if edge already exists
        for edge in self.adjacency[source]:
            if edge["target"] == target:
                # Update weight to max of existing and new
                if weight > edge.get("weight", 0.0):
                    edge["weight"] = round(weight, 3)
                    edge["relation_type"] = relation_type
                    if shared_terms:
                        edge["shared_terms"] = list(set(edge.get("shared_terms", []) + shared_terms))
                return

        self.adjacency[source].append({
            "target": target,
            "weight": round(weight, 3),
            "relation_type": relation_type,
            "shared_terms": shared_terms or [],
            "description": description or ""
        })

    def add_bidirectional_edge(self, node_a: str, node_b: str, weight: float, relation_type: str, shared_terms: Optional[List[str]] = None):
        """Add symmetric edges between node_a and node_b."""
        self.add_edge(node_a, node_b, weight, relation_type, shared_terms)
        self.add_edge(node_b, node_a, weight, relation_type, shared_terms)

    def get_neighbors(self, rule_id: str, min_weight: float = 0.50, limit: int = 5) -> List[Tuple[str, float, str]]:
        """
        Get highest-weighted connected rule IDs for a given rule.
        Returns List of (target_rule, weight, relation_type).
        """
        rule_id = str(rule_id).strip()
        edges = self.adjacency.get(rule_id, [])
        filtered = [
            (e["target"], float(e.get("weight", 0.0)), e.get("relation_type", "unknown"))
            for e in edges
            if float(e.get("weight", 0.0)) >= min_weight
        ]
        # Sort descending by weight
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:limit]

    def save_json(self, filepath: str):
        """Save graph to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "CooccurrenceGraph":
        """Load graph from JSON file."""
        if not os.path.exists(filepath):
            return cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


class SectionNode(BaseModel):
    """
    Represents a major section or parent rule in the document hierarchy.
    """
    section_id: str
    title: str = ""
    parent_id: Optional[str] = None
    level: int = 1  # 1 = Chapter, 2 = Section, 3 = Leaf Rule
    child_rules: List[str] = Field(default_factory=list)
    child_sections: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    doc_type: str = "core_rules"
    priority: int = 1


class SectionTree(BaseModel):
    """
    Represents the complete hierarchical structure of a document collection.
    """
    game_id: str = "generic"
    sections: Dict[str, SectionNode] = Field(default_factory=dict)
    rule_to_section_map: Dict[str, str] = Field(default_factory=dict)

    def add_section(self, section_id: str, title: str, parent_id: Optional[str] = None, level: int = 1, doc_type: str = "core_rules", priority: int = 1) -> SectionNode:
        sec_id = str(section_id).strip()
        if sec_id not in self.sections:
            self.sections[sec_id] = SectionNode(
                section_id=sec_id,
                title=title,
                parent_id=parent_id,
                level=level,
                doc_type=doc_type,
                priority=priority
            )
        else:
            if title and not self.sections[sec_id].title:
                self.sections[sec_id].title = title
            if parent_id and not self.sections[sec_id].parent_id:
                self.sections[sec_id].parent_id = parent_id

        if parent_id and parent_id in self.sections:
            if sec_id not in self.sections[parent_id].child_sections:
                self.sections[parent_id].child_sections.append(sec_id)

        return self.sections[sec_id]

    def register_rule(self, rule_id: str, parent_section_id: str, chunk_id: Optional[str] = None):
        """Map a leaf rule to its parent section container."""
        r_id = str(rule_id).strip()
        sec_id = str(parent_section_id).strip()
        self.rule_to_section_map[r_id] = sec_id

        if sec_id in self.sections:
            sec = self.sections[sec_id]
            if r_id not in sec.child_rules:
                sec.child_rules.append(r_id)
            if chunk_id and chunk_id not in sec.chunk_ids:
                sec.chunk_ids.append(chunk_id)

    def get_parent_section(self, rule_or_section_id: str) -> Optional[SectionNode]:
        """Find the immediate parent SectionNode for a rule or sub-section."""
        query_id = str(rule_or_section_id).strip()
        if query_id in self.rule_to_section_map:
            sec_id = self.rule_to_section_map[query_id]
            return self.sections.get(sec_id)
        if query_id in self.sections:
            p_id = self.sections[query_id].parent_id
            if p_id:
                return self.sections.get(p_id)
        return None

    def get_sibling_rules(self, rule_id: str) -> List[str]:
        """Get all sibling rule IDs in the same parent section."""
        parent_sec = self.get_parent_section(rule_id)
        if parent_sec:
            return [r for r in parent_sec.child_rules if r != rule_id]
        return []

    def get_symmetric_sibling_rules(self, rule_id: str, window: int = 2) -> List[str]:
        """
        Get numerically symmetric adjacent sibling rule IDs within +/- window distance.
        E.g. for '17.7' in ['17.1', '17.2', ..., '17.7', '17.8', '17.9'],
        window=2 returns ['17.5', '17.6', '17.8', '17.9'].
        """
        parent_sec = self.get_parent_section(rule_id)
        if not parent_sec or not parent_sec.child_rules:
            return []

        sorted_children = sorted(list(set(parent_sec.child_rules)))
        if rule_id in sorted_children:
            idx = sorted_children.index(rule_id)
            start = max(0, idx - window)
            end = min(len(sorted_children), idx + window + 1)
            return [r for r in sorted_children[start:end] if r != rule_id]

        return [r for r in sorted_children[:window * 2] if r != rule_id]

    def save_json(self, filepath: str):
        """Save section tree to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "SectionTree":
        """Load section tree from JSON file."""
        if not os.path.exists(filepath):
            return cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
