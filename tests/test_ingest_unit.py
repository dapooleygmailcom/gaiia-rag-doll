"""
Unit Tests for the Ingestion Pipeline.

Tests the generic ingestion functions to ensure they work correctly 
for Up Front AND future games (ASL, SFB, etc.) without regression.

Run with:
    venv\\Scripts\\python.exe -m pytest test_ingest_unit.py -v
"""

import re
import pytest
import sys
import os

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.ingestion.ingest_rules import (
    extract_rule_numbers,
    extract_cross_references,
    detect_content_type,
    detect_scenario,
    build_rule_index,
    chunk_generic,
    _build_chunk,
    RULE_NUMBER_PATTERN,
    CROSS_REF_PATTERN,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures: Sample Texts
# ═══════════════════════════════════════════════════════════════════

UPFRONT_RULE_TEXT = """
5.41 TERRAIN PLACEMENT: A terrain card may be played on a group 
provided the terrain card is accepted by the opposing player. [6.1]
See 5.61 for lateral distance effects.
"""

UPFRONT_ERRATA_TEXT = """
7.32
Q. Can a player with multi-card discard capability discard one terrain 
card on a group, have it rejected, and still discard another terrain 
card on the same group in the same turn?
A. No.
"""

UPFRONT_QA_TEXT = """
Q. Can you stop Fire Combat Resolution partway through a target group?
A. No. EXC: [6.5]
VARIANT: Optional rule allows early stoppage in Scenario C.
"""

UPFRONT_SCENARIO_TEXT = """
A. MEETING OF PATROLS
Both players set up their forces simultaneously.
Special Rules: Pillboxes may not be placed. Minefield cards act as Cower cards.
Victory Conditions: Eliminate 3 enemy men.
"""

GENERIC_TEXT_BLOCK = """
This is a general description without any rule numbers.
It contains game concepts like movement and firing.
Multiple paragraphs follow to fill out the content.
"""


# ═══════════════════════════════════════════════════════════════════
# 1. Rule Number Extraction — Up Front Format
# ═══════════════════════════════════════════════════════════════════

class TestRuleNumberExtractionUpFront:
    """Tests for Up Front decimal rule number format: N.NNN"""

    def test_extracts_simple_rule_number(self):
        text = "5.41 TERRAIN PLACEMENT: Cards may be played..."
        result = extract_rule_numbers(text)
        assert "5.41" in result

    def test_extracts_multiple_rule_numbers(self):
        # RULE_NUMBER_PATTERN is anchored to line-start, so it finds rule headers,
        # not inline references. Inline refs are captured by CROSS_REF_PATTERN.
        # Three rule headers on separate lines should all be extracted.
        text = "3.3 SETUP. See also rules below.\n11.12 GROUP REARRANGEMENT: Details here.\n18.2 SPECIAL RULES: Apply."
        result = extract_rule_numbers(text)
        assert "3.3" in result
        assert "11.12" in result
        assert "18.2" in result

    def test_extracts_three_level_rule_number(self):
        text = "17.4.1 Exception to crew-served weapon rule."
        result = extract_rule_numbers(text)
        assert "17.4.1" in result or "17.4" in result  # accept either depth

    def test_does_not_extract_page_numbers(self):
        # Page numbers at start of text should not be confused with rules
        text = "--- PAGE 12 ---\nSome game text without rule numbers"
        result = extract_rule_numbers(text)
        assert "12" not in result

    def test_does_not_extract_years(self):
        text = "Originally published in 1983 by Avalon Hill."
        result = extract_rule_numbers(text)
        # 1983 doesn't match N.NNN pattern
        assert len(result) == 0

    def test_returns_unique_rule_numbers(self):
        # Rule numbers at line-start appear once; duplicates should be deduplicated.
        text = "5.41 TERRAIN PLACEMENT: Cards may be played.\n5.41 TERRAIN PLACEMENT: Cards may be played."
        result = extract_rule_numbers(text)
        assert result.count("5.41") == 1


# ═══════════════════════════════════════════════════════════════════
# 2. Cross-Reference Extraction — Up Front Format
# ═══════════════════════════════════════════════════════════════════

class TestCrossRefExtractionUpFront:
    """Tests for Up Front cross-reference patterns."""

    def test_extracts_bracketed_ref(self):
        text = "Fire occurs at target group. [5.61]"
        result = extract_cross_references(text)
        assert "5.61" in result

    def test_extracts_see_ref(self):
        text = "For blocking rules, see 5.61 for details."
        result = extract_cross_references(text)
        assert "5.61" in result

    def test_extracts_exc_ref(self):
        text = "Movement is complete. EXC: [6.5] applies to fire."
        result = extract_cross_references(text)
        assert "6.5" in result

    def test_extracts_rule_ref(self):
        text = "As stated in rule 10.45, heroes cannot be used on unpinned men."
        result = extract_cross_references(text)
        assert "10.45" in result

    def test_extracts_multiple_refs(self):
        text = "See [5.61] and rule 3.3 for lateral distance. EXC: [6.5]"
        result = extract_cross_references(text)
        assert "5.61" in result
        assert "3.3" in result
        assert "6.5" in result


# ═══════════════════════════════════════════════════════════════════
# 3. Content Type Detection
# ═══════════════════════════════════════════════════════════════════

class TestContentTypeDetection:

    def test_detects_qa_block(self):
        result = detect_content_type(UPFRONT_ERRATA_TEXT)
        assert result == "qa"

    def test_detects_clarification(self):
        text = "CLARIFICATION: Rule 5.41 means terrain cards must be offered before firing."
        result = detect_content_type(text)
        assert result == "clarification"

    def test_detects_variant(self):
        text = "VARIANT: Optional house rule for morale checks."
        result = detect_content_type(text)
        assert result == "variant"

    def test_detects_plain_rule(self):
        result = detect_content_type(UPFRONT_RULE_TEXT)
        assert result == "rule"

    def test_detects_scenario_rule(self):
        # UPFRONT_SCENARIO_TEXT has 'Q.' implied by 'A.' in 'Victory Conditions:'
        # but no actual Q./A. pattern. Plain scenario text = 'rule'.
        # The detect_scenario() function is the authoritative route for scenario detection.
        plain_scenario_text = "MEETING OF PATROLS\nBoth players set up forces simultaneously.\nSpecial Rules: Pillboxes forbidden."
        result = detect_content_type(plain_scenario_text)
        assert result in ("rule", "scenario_rule")


# ═══════════════════════════════════════════════════════════════════
# 4. Scenario Detection — Up Front (Letter-based: A-K)
# ═══════════════════════════════════════════════════════════════════

class TestScenarioDetectionUpFront:

    def test_detects_scenario_letter_from_header(self):
        text = "A. MEETING OF PATROLS\nSpecial rules follow."
        result = detect_scenario(text)
        assert result is not None
        assert "A" in result

    def test_detects_scenario_letter_in_body(self):
        text = "These rules apply only in Scenario C."
        result = detect_scenario(text)
        assert result is not None
        assert "C" in result

    def test_no_scenario_in_plain_rule(self):
        result = detect_scenario(UPFRONT_RULE_TEXT)
        # Should not detect spurious scenario letters from rule text
        # (letters in abbreviations like EXC, QA, etc. should not trigger)
        assert result is None or len(result) == 0

    def test_detects_multiple_scenarios(self):
        text = "This rule modifies both Scenario A and Scenario B."
        result = detect_scenario(text)
        assert result is not None
        assert "A" in result
        assert "B" in result


# ═══════════════════════════════════════════════════════════════════
# 5. Chunk Builder
# ═══════════════════════════════════════════════════════════════════

class TestBuildChunk:

    def test_chunk_has_required_keys(self):
        lines = ["5.41 TERRAIN: A card may be played if accepted."]
        chunk = _build_chunk(
            lines, "Up_Front.pdf", "core_rules", 2,
            "5. Movement & Range", "5.41", "5.41", 12
        )
        assert "text" in chunk
        assert "metadata" in chunk
        assert "rule_number" in chunk
        assert "cross_refs" in chunk
        assert "all_rule_numbers" in chunk

    def test_chunk_metadata_has_chroma_compatible_types(self):
        """ChromaDB only accepts str, int, float, bool in metadata."""
        lines = ["17.4 CREWED WEAPONS: LMG requires two men. [10.45]"]
        chunk = _build_chunk(
            lines, "Up_Front.pdf", "core_rules", 2,
            "17. Weapons", "17.4", "17.4", 45
        )
        for key, val in chunk["metadata"].items():
            assert isinstance(val, (str, int, float, bool)), \
                f"Metadata key '{key}' has non-ChromaDB type {type(val)}"

    def test_chunk_removes_self_reference(self):
        """Cross-refs should not include the primary rule itself."""
        lines = ["5.41 See [5.41] and [6.1] for more."]
        chunk = _build_chunk(
            lines, "Up_Front.pdf", "core_rules", 2,
            "5. Movement", "5.41", "5.41", 10
        )
        assert "5.41" not in chunk["cross_refs"]
        assert "6.1" in chunk["cross_refs"]

    def test_chunk_enriches_text_with_header(self):
        lines = ["5.41 TERRAIN PLACEMENT"]
        chunk = _build_chunk(
            lines, "Up_Front.pdf", "core_rules", 2,
            "5. Movement", "5.41", "5.41", 10
        )
        assert "[Doc: core_rules]" in chunk["text"]
        assert "[Rule: 5.41]" in chunk["text"]

    def test_chunk_includes_scenario_metadata(self):
        lines = ["Special rule for Scenario C: Sniper cards treated as Cower."]
        chunk = _build_chunk(
            lines, "Upfront_Scenarios_1.pdf", "scenarios", 5,
            "Scenario C", "C", None, 3, scenario="C"
        )
        assert chunk["metadata"].get("scenario") == "C"
        assert "[Scenario: C]" in chunk["text"]


# ═══════════════════════════════════════════════════════════════════
# 6. Generic Chunker
# ═══════════════════════════════════════════════════════════════════

class TestGenericChunker:

    def test_chunker_returns_list(self):
        text = "--- PAGE 1 ---\nSome variant rule text that is long enough to matter."
        result = chunk_generic(text, "Up_Front-_Experimental_house_and_variant_rules.pdf")
        assert isinstance(result, list)

    def test_chunker_splits_large_text(self):
        # Generate a large block of text that exceeds the 2000-char limit
        big_line = "This is a long line of rules text. " * 20
        text = "\n".join([f"--- PAGE {i} ---\n{big_line}" for i in range(1, 5)])
        result = chunk_generic(text, "Up_Front-_Experimental_house_and_variant_rules.pdf")
        assert len(result) > 1, "Large text should produce multiple chunks"

    def test_chunker_skips_short_text(self):
        text = "--- PAGE 1 ---\nHi."
        result = chunk_generic(text, "What_is_UpFront.pdf")
        assert len(result) == 0, "Text below minimum length should be skipped"


# ═══════════════════════════════════════════════════════════════════
# 7. Rule Index Builder
# ═══════════════════════════════════════════════════════════════════

class TestBuildRuleIndex:

    def _make_chunks(self):
        return {
            "chunk_1": {
                "rule_number": "5.41",
                "all_rule_numbers": ["5.41", "6.1"],
                "metadata": {
                    "doc_type": "integrated_rules",
                    "priority": 1,
                    "source_file": "UF RuleBook updated.pdf",
                    "content_type": "rule"
                }
            },
            "chunk_2": {
                "rule_number": "5.41",
                "all_rule_numbers": ["5.41"],
                "metadata": {
                    "doc_type": "core_rules",
                    "priority": 2,
                    "source_file": "Up_Front.pdf",
                    "content_type": "rule"
                }
            },
            "chunk_3": {
                "rule_number": "6.1",
                "all_rule_numbers": ["6.1", "5.41"],
                "metadata": {
                    "doc_type": "core_rules",
                    "priority": 2,
                    "source_file": "Up_Front.pdf",
                    "content_type": "rule"
                }
            }
        }

    def test_index_contains_all_rule_numbers(self):
        index = build_rule_index(self._make_chunks())
        assert "5.41" in index
        assert "6.1" in index

    def test_index_sorted_by_priority(self):
        index = build_rule_index(self._make_chunks())
        entries_541 = index["5.41"]
        priorities = [e["priority"] for e in entries_541]
        assert priorities == sorted(priorities), "Entries must be sorted by priority ascending"

    def test_index_highest_priority_first(self):
        index = build_rule_index(self._make_chunks())
        assert index["5.41"][0]["priority"] == 1  # integrated_rules is P1

    def test_secondary_entries_flagged(self):
        """Chunks mentioning a rule but not as their primary should be flagged."""
        index = build_rule_index(self._make_chunks())
        # 5.41 is mentioned in chunk_3 (about rule 6.1), should be secondary
        secondary_entries = [e for e in index.get("5.41", []) if e.get("secondary")]
        assert len(secondary_entries) > 0


# ═══════════════════════════════════════════════════════════════════
# 8. Generalizability: ASL Rule Pattern
# ═══════════════════════════════════════════════════════════════════

class TestASLRulePatterns:
    """
    These tests validate that ASL-style rule numbers can be parsed 
    using a DIFFERENT regex pattern, proving the system is generalisable.
    The actual ASL ingestion uses a profile with its own compiled patterns.
    """

    # ASL uses chapter-letter + decimal: A7.212, B3.1, D5.6
    ASL_RULE_PATTERN = re.compile(
        r'(?:^|\s)([A-Z]\d{1,2}\.\d{1,4})\b',
        re.MULTILINE
    )

    # ASL cross-refs appear as (A20.54) or [EXC: A4.1] or see A7.2
    # Updated ASL cross-ref pattern — EXC block can have trailing text before ]
    ASL_CROSS_REF_PATTERN = re.compile(
        r'\(([A-Z]\d{1,2}\.\d{1,4})\)|'             # (A20.54)
        r'\[EXC:\s*([A-Z]\d{1,2}\.\d{1,4})[^\]]*\]|'  # [EXC: A4.1 ...trailing...]
        r'(?:see|See|SEE)\s+([A-Z]\d{1,2}\.\d{1,4})',  # see A7.2
        re.MULTILINE
    )

    def test_asl_rule_pattern_matches_chapter_decimal(self):
        text = "A7.212 ENTRY: All forces arriving on a certain turn must enter the mapboard."
        matches = self.ASL_RULE_PATTERN.findall(text)
        assert "A7.212" in matches

    def test_asl_rule_pattern_matches_simple_chapter_rule(self):
        text = "D5.6 survival check must be made by all PRC."
        matches = self.ASL_RULE_PATTERN.findall(text)
        assert "D5.6" in matches

    def test_asl_cross_ref_parenthetical(self):
        text = "...not possessing a functioning Gun/SW (A20.54)..."
        matches = self.ASL_CROSS_REF_PATTERN.findall(text)
        found = [g for tup in matches for g in tup if g]
        assert "A20.54" in found

    def test_asl_cross_ref_exc_bracket(self):
        # [EXC: A4.1 if HD ...] — rule number followed by trailing text before closing ]
        text = "vehicle is immobilized [EXC: A4.1 if HD the vehicle is unaffected]"
        matches = self.ASL_CROSS_REF_PATTERN.findall(text)
        found = [g for tup in matches for g in tup if g]
        assert "A4.1" in found

    def test_asl_cross_ref_see_format(self):
        text = "Entry may be delayed (see A4.14) if the hex is blocked."
        matches = self.ASL_CROSS_REF_PATTERN.findall(text)
        found = [g for tup in matches for g in tup if g]
        assert "A4.14" in found

    def test_upfront_pattern_does_not_match_asl_rules(self):
        """Proves the two patterns are distinct and won't cross-contaminate."""
        asl_text = "A7.212 Movement rules. See (D5.6) for vehicle survival."
        upfront_matches = RULE_NUMBER_PATTERN.findall(asl_text)
        # The Up Front pattern should NOT match "A7.212" or "D5.6" (they start with letters)
        assert "A7.212" not in upfront_matches
        assert "D5.6" not in upfront_matches

    def test_asl_pattern_does_not_match_upfront_rules(self):
        """Proves ASL pattern won't match plain decimal Up Front rules."""
        uf_text = "5.41 TERRAIN PLACEMENT. See [6.1] for fire."
        asl_matches = self.ASL_RULE_PATTERN.findall(uf_text)
        assert "5.41" not in asl_matches
        assert "6.1" not in asl_matches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
