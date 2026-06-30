"""
ASL Ingestion Unit Tests.

Tests that verify correct parsing, chunking, and indexing of 
Advanced Squad Leader documents using the ASL rule schema.

These tests work on REAL ASL PDF text samples without requiring
a running Ollama or ChromaDB instance. They validate:
  1. ASL rule number detection (Chapter-Decimal format: A7.212)
  2. ASL cross-reference extraction (parenthetical and EXC formats)
  3. ASL document classification (core rules, errata, scenarios, QA)
  4. Chunk structure correctness for ASL content
  5. Rule index correctness for ASL multi-source conflicts
  6. Isolation from Up Front patterns (regression safety)

Run with:
    venv\\Scripts\\python.exe -m pytest test_ingest_asl.py -v
"""

import re
import os
import sys
import json
import pytest
import fitz  # PyMuPDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.ingestion.ingest_rules import build_rule_index, _build_chunk

# ═══════════════════════════════════════════════════════════════════
# ASL Rule Schema — mirrors what auto_discover.py will produce
# ═══════════════════════════════════════════════════════════════════

ASL_RULE_PATTERN = re.compile(
    r'(?:^|\s)([A-Z]\d{1,2}\.\d{1,4})\b',
    re.MULTILINE
)

ASL_CROSS_REF_PATTERN = re.compile(
    r'\(([A-Z]\d{1,2}\.\d{1,4})\)|'              # (A20.54)
    r'\[EXC:\s*([A-Z]\d{1,2}\.\d{1,4})[^\]]*\]|' # [EXC: A4.1 ...trailing text...]
    r'(?:see|See|SEE)\s+([A-Z]\d{1,2}\.\d{1,4})', # see A7.2
    re.MULTILINE
)

# ASL numeric-only sections (used in non-lettered chapters like the intro)
ASL_NUMERIC_RULE_PATTERN = re.compile(
    r'(?:^|\n)\s*(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\b',
    re.MULTILINE
)

# ASL document corpus with authority stack
ASL_DOCUMENT_CLASSIFICATION = {
    "pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf": {
        "doc_type": "core_rules",
        "priority": 1,
        "description": "ASL 2nd Edition Core Rulebook — primary authority"
    },
    "ASL_Rulebook_Version_Tracker_v1.5.pdf": {
        "doc_type": "version_tracker",
        "priority": 2,
        "description": "Official change log tracking differences between editions"
    },
    "ASLRB_Errata_Dec_2025.pdf": {
        "doc_type": "errata",
        "priority": 3,
        "description": "Official errata for the ASL Rulebook (Dec 2025)"
    },
    "ASL_HASL_Errata_Mar_2025.pdf": {
        "doc_type": "errata",
        "priority": 3,
        "description": "Historical ASL errata (Mar 2025)"
    },
    "ASL_Scenario_Errata_Nov_2025.pdf": {
        "doc_type": "scenario_errata",
        "priority": 4,
        "description": "Scenario-specific errata (Nov 2025)"
    },
    "ASL_Scenario_Balance_Nov_2025.pdf": {
        "doc_type": "scenario_balance",
        "priority": 5,
        "description": "Scenario balance adjustments (Nov 2025)"
    },
    "SR ASL_QA v22.pdf": {
        "doc_type": "qa",
        "priority": 6,
        "description": "Q&A / clarifications document v22 (2005) — unofficial unless in official source"
    },
    "pdfcoffee.com_asl-core-module-scenarios-1-136-pdf-free.pdf": {
        "doc_type": "scenarios",
        "priority": 7,
        "description": "Core module scenarios 1-136"
    },
}

ASL_DATA_DIR = "data/asl"


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def extract_asl_rule_numbers(text):
    """Extract ASL-style rule numbers from text."""
    return list(set(ASL_RULE_PATTERN.findall(text)))


def extract_asl_cross_refs(text):
    """Extract ASL-style cross-references from text."""
    refs = set()
    for match in ASL_CROSS_REF_PATTERN.finditer(text):
        for group in match.groups():
            if group:
                refs.add(group)
    return list(refs)


def extract_asl_numeric_rules(text):
    """Extract numeric-style rules (used in early ASL chapters)."""
    return list(set(ASL_NUMERIC_RULE_PATTERN.findall(text)))


def get_asl_page_text(filename, page_idx):
    """Extract text from a specific ASL PDF page. Skips test if file missing."""
    path = os.path.join(ASL_DATA_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"ASL file not found: {filename}")
    doc = fitz.open(path)
    if page_idx >= len(doc):
        pytest.skip(f"Page {page_idx} does not exist in {filename}")
    text = doc[page_idx].get_text()
    doc.close()
    return text


def get_asl_sample_text(filename, max_pages=5):
    """Extract a sample of text from an ASL PDF (first N pages)."""
    path = os.path.join(ASL_DATA_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"ASL file not found: {filename}")
    doc = fitz.open(path)
    pages = []
    for i in range(min(max_pages, len(doc))):
        text = doc[i].get_text().strip()
        if text:
            pages.append(f"--- PAGE {i+1} ---\n{text}")
    doc.close()
    return "\n\n".join(pages)


# ═══════════════════════════════════════════════════════════════════
# 1. ASL Rule Number Detection
# ═══════════════════════════════════════════════════════════════════

class TestASLRuleNumberDetection:
    """Tests for detecting ASL chapter-decimal rule numbers."""

    RULE_SAMPLES = [
        ("A7.212 ENTRY: All forces arriving on a certain turn must enter the mapboard.", ["A7.212"]),
        ("D5.6 survival check must be made by all PRC.", ["D5.6"]),
        ("When B3.1 applies, infantry movement is halved.", ["B3.1"]),
        ("See also A4.14 and A4.15 for adjacent hex rules.", ["A4.14", "A4.15"]),
        ("7.308 vehicles/horses present in the same Location.", []),  # numeric-only, not ASL chapter
    ]

    @pytest.mark.parametrize("text, expected_rules", RULE_SAMPLES)
    def test_detects_asl_rule_numbers(self, text, expected_rules):
        result = extract_asl_rule_numbers(text)
        for rule in expected_rules:
            assert rule in result, f"Expected '{rule}' in {result}"

    def test_does_not_match_year(self):
        text = "Published in 1985. First edition came in 1983."
        result = extract_asl_rule_numbers(text)
        assert len(result) == 0

    def test_does_not_match_upfront_rules(self):
        """ASL pattern must NOT match Up Front decimal rules like 5.41."""
        text = "5.41 TERRAIN PLACEMENT. See also 17.4 for crew weapons."
        result = extract_asl_rule_numbers(text)
        assert "5.41" not in result
        assert "17.4" not in result

    def test_real_asl_core_rules_page_contains_rules(self):
        """Integration check: real ASL PDF pages contain detectable rules."""
        text = get_asl_page_text(
            "pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf", 20
        )
        result = extract_asl_rule_numbers(text) + extract_asl_numeric_rules(text)
        assert len(result) > 0, "Real ASL core rules pages should contain rule numbers"


# ═══════════════════════════════════════════════════════════════════
# 2. ASL Cross-Reference Detection
# ═══════════════════════════════════════════════════════════════════

class TestASLCrossReferenceDetection:
    """Tests for ASL cross-reference patterns: parenthetical, EXC, see."""

    CROSS_REF_SAMPLES = [
        ("not possessing a functioning Gun/SW (A20.54).", ["A20.54"]),
        ("vehicle is immobilized [EXC: A4.1 if HD]", ["A4.1"]),
        ("entry may be delayed (see A4.14)", ["A4.14"]),
        ("Armed (A20.54) and FBE (B3.1) both apply.", ["A20.54", "B3.1"]),
    ]

    @pytest.mark.parametrize("text, expected_refs", CROSS_REF_SAMPLES)
    def test_extracts_asl_cross_refs(self, text, expected_refs):
        result = extract_asl_cross_refs(text)
        for ref in expected_refs:
            assert ref in result, f"Expected '{ref}' in {result}"

    def test_real_errata_page_has_rule_references(self):
        """Integration: real errata should have inline rule refs."""
        text = get_asl_page_text("ASLRB_Errata_Dec_2025.pdf", 0)
        rule_refs = extract_asl_cross_refs(text)
        # Errata pages extensively reference chapter-decimal rules
        assert len(rule_refs) > 0 or len(extract_asl_rule_numbers(text)) > 0


# ═══════════════════════════════════════════════════════════════════
# 3. ASL Document Classification
# ═══════════════════════════════════════════════════════════════════

class TestASLDocumentClassification:
    """Tests for the ASL authority stack document classification."""

    def test_all_known_files_classified(self):
        known_files = [
            "pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf",
            "ASLRB_Errata_Dec_2025.pdf",
            "SR ASL_QA v22.pdf",
        ]
        for fname in known_files:
            assert fname in ASL_DOCUMENT_CLASSIFICATION, \
                f"File '{fname}' should have a classification entry"

    def test_core_rules_highest_priority(self):
        core = ASL_DOCUMENT_CLASSIFICATION[
            "pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf"
        ]
        errata = ASL_DOCUMENT_CLASSIFICATION["ASLRB_Errata_Dec_2025.pdf"]
        qa = ASL_DOCUMENT_CLASSIFICATION["SR ASL_QA v22.pdf"]
        # Core rules should have a lower priority number than errata and QA
        assert core["priority"] < errata["priority"]
        assert core["priority"] < qa["priority"]

    def test_errata_higher_priority_than_qa(self):
        errata = ASL_DOCUMENT_CLASSIFICATION["ASLRB_Errata_Dec_2025.pdf"]
        qa = ASL_DOCUMENT_CLASSIFICATION["SR ASL_QA v22.pdf"]
        assert errata["priority"] < qa["priority"]

    def test_scenario_errata_higher_priority_than_scenarios(self):
        scenario_errata = ASL_DOCUMENT_CLASSIFICATION["ASL_Scenario_Errata_Nov_2025.pdf"]
        scenarios = ASL_DOCUMENT_CLASSIFICATION[
            "pdfcoffee.com_asl-core-module-scenarios-1-136-pdf-free.pdf"
        ]
        assert scenario_errata["priority"] < scenarios["priority"]

    def test_qa_doc_is_marked_unofficial(self):
        qa = ASL_DOCUMENT_CLASSIFICATION["SR ASL_QA v22.pdf"]
        # Should have a description mentioning unofficial status
        assert "unofficial" in qa["description"].lower()


# ═══════════════════════════════════════════════════════════════════
# 4. ASL Chunk Structure
# ═══════════════════════════════════════════════════════════════════

class TestASLChunkStructure:
    """Tests that chunks built from ASL text have correct structure."""

    ASL_RULE_TEXT = (
        "A7.212 ENTRY: All forces scheduled to arrive on a certain turn must "
        "enter the mapboard on that turn—although if capable of movement in the "
        "APh, entry may be delayed until then. If entry was via a certain hex "
        "but that hex is unenterable (see A4.14), or is blocked by rubble (D5.6), "
        "then alternate entry is allowed."
    )

    def test_asl_chunk_has_required_keys(self):
        chunk = _build_chunk(
            [self.ASL_RULE_TEXT], "pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf",
            "core_rules", 1, "A. Infantry", "A7.2", "A7.212", 22
        )
        assert "text" in chunk
        assert "metadata" in chunk
        assert "rule_number" in chunk
        assert "cross_refs" in chunk

    def test_asl_chunk_metadata_chroma_compatible(self):
        """All metadata values must be ChromaDB-compatible types."""
        chunk = _build_chunk(
            [self.ASL_RULE_TEXT], "pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf",
            "core_rules", 1, "A. Infantry", "A7.2", "A7.212", 22
        )
        for key, val in chunk["metadata"].items():
            assert isinstance(val, (str, int, float, bool)), \
                f"Metadata key '{key}' has non-ChromaDB type {type(val)}: {val!r}"

    def test_asl_chunk_text_enriched_with_header(self):
        chunk = _build_chunk(
            [self.ASL_RULE_TEXT], "pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf",
            "core_rules", 1, "A. Infantry", "A7.2", "A7.212", 22
        )
        assert "[Doc: core_rules]" in chunk["text"]


# ═══════════════════════════════════════════════════════════════════
# 5. ASL Rule Index — Multi-Source Temporal Conflict
# ═══════════════════════════════════════════════════════════════════

class TestASLRuleIndex:
    """Tests that rule index correctly handles ASL multi-source conflicts."""

    def _make_asl_chunks(self):
        return {
            "asl_core_chunk_1": {
                "rule_number": "A7.212",
                "all_rule_numbers": ["A7.212", "A4.14"],
                "metadata": {
                    "doc_type": "core_rules",
                    "priority": 1,
                    "source_file": "pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf",
                    "content_type": "rule"
                }
            },
            "asl_errata_chunk_1": {
                "rule_number": "A7.212",
                "all_rule_numbers": ["A7.212"],
                "metadata": {
                    "doc_type": "errata",
                    "priority": 3,
                    "source_file": "ASLRB_Errata_Dec_2025.pdf",
                    "content_type": "qa"
                }
            },
            "asl_qa_chunk_1": {
                "rule_number": "A4.14",
                "all_rule_numbers": ["A4.14", "A7.212"],
                "metadata": {
                    "doc_type": "qa",
                    "priority": 6,
                    "source_file": "SR ASL_QA v22.pdf",
                    "content_type": "qa"
                }
            }
        }

    def test_index_contains_asl_rules(self):
        index = build_rule_index(self._make_asl_chunks())
        assert "A7.212" in index
        assert "A4.14" in index

    def test_core_rules_appear_before_errata(self):
        index = build_rule_index(self._make_asl_chunks())
        entries = index["A7.212"]
        priorities = [e["priority"] for e in entries]
        assert priorities[0] < priorities[-1], "Core rules (P1) must appear before errata (P3)"

    def test_qa_appears_last(self):
        index = build_rule_index(self._make_asl_chunks())
        # A4.14 is primary in QA chunk and secondary in core chunk
        entries = index.get("A4.14", [])
        # Primary entry from core (if cross-referenced) or QA's own chunk
        doc_types = [e["doc_type"] for e in entries]
        if "core_rules" in doc_types and "qa" in doc_types:
            core_pos = next(i for i, e in enumerate(entries) if e["doc_type"] == "core_rules")
            qa_pos = next(i for i, e in enumerate(entries) if e["doc_type"] == "qa")
            assert core_pos < qa_pos, "Core rules must appear before QA in the index"

    def test_secondary_references_flagged(self):
        index = build_rule_index(self._make_asl_chunks())
        # A4.14 is cross-referenced (secondary) in core chunk, primary in QA chunk
        entries = index.get("A4.14", [])
        secondary = [e for e in entries if e.get("secondary")]
        assert len(secondary) > 0, "Secondary references should be flagged"


# ═══════════════════════════════════════════════════════════════════
# 6. Regression: ASL Changes Don't Break Up Front
# ═══════════════════════════════════════════════════════════════════

class TestASLDoesNotBreakUpFront:
    """
    Regression tests ensuring ASL ingestion additions
    do not affect the Up Front pipeline.
    """

    UPFRONT_RULE_TEXT = "5.41 TERRAIN PLACEMENT: A card may be played on a group. [6.1]"
    ASL_RULE_TEXT = "A7.212 ENTRY: Forces must enter the mapboard. (A4.14)"

    def test_upfront_pattern_still_matches_uf_rules(self):
        from engine.ingestion.ingest_rules import RULE_NUMBER_PATTERN
        matches = RULE_NUMBER_PATTERN.findall(self.UPFRONT_RULE_TEXT)
        assert "5.41" in matches

    def test_upfront_pattern_not_confused_by_asl_text(self):
        from engine.ingestion.ingest_rules import RULE_NUMBER_PATTERN
        matches = RULE_NUMBER_PATTERN.findall(self.ASL_RULE_TEXT)
        # Should not detect ASL rules like "A7.212" as Up Front rules
        assert "A7.212" not in matches

    def test_build_rule_index_handles_mixed_game_chunks(self):
        """Rule index correctly separates by chunk_id, not by rule format."""
        mixed_chunks = {
            "uf_chunk_1": {
                "rule_number": "5.41",
                "all_rule_numbers": ["5.41", "6.1"],
                "metadata": {
                    "doc_type": "core_rules", "priority": 2,
                    "source_file": "Up_Front.pdf", "content_type": "rule"
                }
            },
            "asl_chunk_1": {
                "rule_number": "A7.212",
                "all_rule_numbers": ["A7.212"],
                "metadata": {
                    "doc_type": "core_rules", "priority": 1,
                    "source_file": "pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf",
                    "content_type": "rule"
                }
            }
        }
        index = build_rule_index(mixed_chunks)
        assert "5.41" in index
        assert "A7.212" in index
        # They must not cross-contaminate
        uf_sources = [e["source_file"] for e in index["5.41"]]
        assert all("Up_Front" in s for s in uf_sources)

    def test_upfront_chunk_builder_still_works(self):
        from engine.ingestion.ingest_rules import _build_chunk
        chunk = _build_chunk(
            ["5.41 TERRAIN: Cards may be played on a group. [6.1]"],
            "Up_Front.pdf", "core_rules", 2,
            "5. Movement", "5.41", "5.41", 12
        )
        assert chunk["metadata"]["source_file"] == "Up_Front.pdf"
        assert chunk["rule_number"] == "5.41"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
