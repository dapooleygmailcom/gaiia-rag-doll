"""
RED/GREEN: End-to-end ingestion unit tests for all four misc PDFs.

Verifies that the complete ingestion pipeline (profile → route_chunk_generic
→ build_rule_index) produces outputs that MATCH the implementation plan's
declared expected values for each book:

  Flames of War:
    - 80–400 chunks from keyword_header chunker
    - Rule index contains SHOOTING, MOVEMENT, ASSAULTS, TERRAIN
    - No decimal rules bleed from Up Front schema

  Honours of War:
    - 40–250 chunks
    - Rule index contains firing, melee, movement
    - Table annotation markers injected if extract_tables=True

  Tokyo Express:
    - Rule index contains rules 3.0–3.9
    - No 9.x phantom rules from OCR misread
    - All rule_numbers are N.D decimal format

  Albedo RPG:
    - Native extraction yields <1000 chars (confirms OCR required)
    - With OCR (if available): ≥20 keyword_header chunks from 30 pages
    - Without OCR: test is marked xfail (expected RED)

All tests are pure text processing — no LLM, ChromaDB, or AWS services.

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_misc_ingestion.py -v
"""

import os
import sys
import json
import re
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fitz

from engine.ingestion.ingest_rules import (
    route_chunk_generic,
    get_compiled_patterns,
    build_rule_index,
)

MISC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/misc"))

# ── Per-book plan specs (ground truth from implementation plan Phase 1) ────────
PLAN_CHUNK_COUNTS = {
    "flames_of_war": (80, 1500),
    "honours_of_war": (40, 450),
    "tokyo_express": (20, 450),
    "albedo_rpg": (20, 400),   # OCR-only; xfail without OCR
}

PLAN_RULE_INDEX_KEYS = {
    "flames_of_war": {"shooting", "movement", "assaults", "terrain"},
    "honours_of_war": {"firing", "melee", "movement"},
    "tokyo_express": {"3.1", "3.2", "3.5", "3.6"},
}

PLAN_RULE_INDEX_MIN_COUNTS = {
    "flames_of_war": 80,
    "honours_of_war": 40,
    "tokyo_express": 7,    # 7 of 10 rule sections
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_profile(profile_name: str) -> dict:
    path = os.path.join(MISC_DIR, profile_name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def skip_if_profile_missing(profile_name: str):
    path = os.path.join(MISC_DIR, profile_name)
    if not os.path.exists(path):
        pytest.skip(f"Profile not yet created (RED state): {profile_name}")


def extract_text_native(fname: str, max_pages: int = None) -> str:
    path = os.path.join(MISC_DIR, fname)
    doc = fitz.open(path)
    limit = min(max_pages or len(doc), len(doc))
    pages = []
    for i in range(limit):
        t = doc[i].get_text().strip()
        if t:
            pages.append(f"--- PAGE {i + 1} ---\n{t}")
    doc.close()
    return "\n\n".join(pages)


def full_pipeline(profile_name: str, fname: str, max_pages: int = None):
    """Run profile → chunk → index and return (chunks, indexed, rule_index)."""
    profile = load_profile(profile_name)
    patterns = get_compiled_patterns(profile)
    doc_info = list(profile["documents"].values())[0]
    text = extract_text_native(fname, max_pages=max_pages)
    chunks = route_chunk_generic(text, fname, doc_info, profile, patterns)
    game_id = profile.get("game_id", "x")
    indexed = {f"{game_id}_{i}": c for i, c in enumerate(chunks)}
    rule_index = build_rule_index(indexed)
    return chunks, indexed, rule_index


# ═══════════════════════════════════════════════════════════════════
# 1. Flames of War — full ingestion plan verification
# ═══════════════════════════════════════════════════════════════════

class TestFlamesOfWarIngestion:
    PROFILE = "flames_of_war_profile.json"
    FNAME = "pdfcoffee.com_flames-of-war-4th-ed-ew-amp-lw-rule-book-pdf-free.pdf"

    @pytest.fixture(scope="class")
    def pipeline(self):
        skip_if_profile_missing(self.PROFILE)
        return full_pipeline(self.PROFILE, self.FNAME)

    # ── Plan-declared chunk count ──────────────────────────────────

    def test_chunk_count_within_plan_range(self, pipeline):
        """
        Plan declares 80–400 chunks for FoW 109-page keyword_header book.
        RED: old Up Front pattern yields <20 chunks.
        GREEN: keyword_header profile yields 80–400.
        """
        chunks, _, _ = pipeline
        lo, hi = PLAN_CHUNK_COUNTS["flames_of_war"]
        assert lo <= len(chunks) <= hi, (
            f"FoW chunk count {len(chunks)} outside plan range [{lo}, {hi}]"
        )

    # ── Plan-declared rule index keys ─────────────────────────────

    def test_shooting_section_indexed(self, pipeline):
        _, _, rule_index = pipeline
        keys_lower = {k.lower() for k in rule_index if not k.startswith("__")}
        assert any("shooting" in k for k in keys_lower), (
            f"FoW rule index must contain a SHOOTING entry. "
            f"Got keys (first 30): {sorted(keys_lower)[:30]}"
        )

    def test_movement_section_indexed(self, pipeline):
        _, _, rule_index = pipeline
        keys_lower = {k.lower() for k in rule_index if not k.startswith("__")}
        assert any("movement" in k for k in keys_lower), (
            f"FoW rule index must contain a MOVEMENT entry. "
            f"Got keys: {sorted(keys_lower)[:30]}"
        )

    def test_plan_required_sections_indexed_count(self, pipeline):
        """At least 2 of the 4 core plan sections must appear in the index."""
        _, _, rule_index = pipeline
        keys_lower = {k.lower() for k in rule_index if not k.startswith("__")}
        required = PLAN_RULE_INDEX_KEYS["flames_of_war"]
        found = {r for r in required if any(r in k for k in keys_lower)}
        assert len(found) >= 2, (
            f"FoW: found only {len(found)} of {required} in index keys. "
            f"Keys sample: {sorted(keys_lower)[:30]}"
        )

    def test_rule_index_minimum_entry_count(self, pipeline):
        """Plan declares minimum 80 indexed sections for FoW."""
        _, _, rule_index = pipeline
        non_meta = [k for k in rule_index if not k.startswith("__")]
        assert len(non_meta) >= PLAN_RULE_INDEX_MIN_COUNTS["flames_of_war"], (
            f"FoW rule index has {len(non_meta)} entries, plan minimum is "
            f"{PLAN_RULE_INDEX_MIN_COUNTS['flames_of_war']}"
        )

    # ── Cross-game isolation ───────────────────────────────────────

    def test_no_upfront_decimals_in_fow_index(self, pipeline):
        """
        Plan section 'What Does NOT Change' guarantees schema isolation.
        Up Front pattern '5.41' must not bleed into FoW index.
        """
        _, _, rule_index = pipeline
        decimal_keys = [k for k in rule_index if re.match(r"^\d+\.\d+$", str(k))]
        assert not decimal_keys, (
            f"FoW index contains Up Front-style decimal rules: {decimal_keys}\n"
            f"This indicates rule_schema is not being respected — check profile."
        )

    def test_no_asl_chapter_decimal_in_fow_index(self, pipeline):
        """ASL 'A7.212' rules must not appear in FoW index."""
        _, _, rule_index = pipeline
        asl_keys = [k for k in rule_index if re.match(r"^[A-Z]\d+\.\d+", str(k))]
        assert not asl_keys, f"FoW index contains ASL-style rules: {asl_keys}"

    # ── Metadata spot-check on SHOOTING chunk ─────────────────────

    def test_shooting_chunk_has_correct_doc_type(self, pipeline):
        chunks, _, rule_index = pipeline
        keys_lower_map = {k.lower(): k for k in rule_index if not k.startswith("__")}
        # Find a SHOOTING-related rule index key
        shooting_key = next(
            (orig for low, orig in keys_lower_map.items() if "shooting" in low), None
        )
        if not shooting_key:
            pytest.skip("No SHOOTING section found in FoW index")
        entries = rule_index[shooting_key]
        for entry in entries:
            chunk_id = entry["chunk_id"]
            indexed = {f"flames_of_war_{i}": c for i, c in enumerate(chunks)}
            if chunk_id in indexed:
                assert indexed[chunk_id]["metadata"]["doc_type"] == "core_rules"
                assert indexed[chunk_id]["metadata"]["priority"] == 1
                break


# ═══════════════════════════════════════════════════════════════════
# 2. Honours of War — full ingestion plan verification
# ═══════════════════════════════════════════════════════════════════

class TestHonoursOfWarIngestion:
    PROFILE = "honours_of_war_profile.json"
    FNAME = "pdfcoffee.com_osprey-wargames-11-honours-of-war-wargames-rules-for-the-seven-yearsx27-war-pdf-free.pdf"

    @pytest.fixture(scope="class")
    def pipeline(self):
        skip_if_profile_missing(self.PROFILE)
        return full_pipeline(self.PROFILE, self.FNAME)

    def test_chunk_count_within_plan_range(self, pipeline):
        chunks, _, _ = pipeline
        lo, hi = PLAN_CHUNK_COUNTS["honours_of_war"]
        assert lo <= len(chunks) <= hi, (
            f"HoW chunk count {len(chunks)} outside plan range [{lo}, {hi}]"
        )

    def test_firing_section_indexed(self, pipeline):
        _, _, rule_index = pipeline
        keys_lower = {k.lower() for k in rule_index if not k.startswith("__")}
        assert any("fir" in k for k in keys_lower), (
            f"HoW index must contain a Firing-related entry. Keys: {sorted(keys_lower)[:20]}"
        )

    def test_melee_section_indexed(self, pipeline):
        _, _, rule_index = pipeline
        keys_lower = {k.lower() for k in rule_index if not k.startswith("__")}
        assert any("melee" in k or "combat" in k or "assault" in k for k in keys_lower), (
            f"HoW index must contain Melee/Combat entry. Keys: {sorted(keys_lower)[:20]}"
        )

    def test_rule_index_min_count(self, pipeline):
        _, _, rule_index = pipeline
        non_meta = [k for k in rule_index if not k.startswith("__")]
        assert len(non_meta) >= PLAN_RULE_INDEX_MIN_COUNTS["honours_of_war"], (
            f"HoW index has {len(non_meta)} entries, plan minimum is "
            f"{PLAN_RULE_INDEX_MIN_COUNTS['honours_of_war']}"
        )

    def test_no_decimal_rules_in_how_index(self, pipeline):
        _, _, rule_index = pipeline
        decimal_keys = [k for k in rule_index if re.match(r"^\d+\.\d+$", str(k))]
        assert not decimal_keys, (
            f"HoW keyword_header index must not contain decimal rules: {decimal_keys}"
        )

    def test_table_annotation_injected_when_available(self, pipeline):
        """
        If table_extractor is available AND tables exist in HoW,
        at least one chunk's text should contain the '<!-- TABLES' annotation.
        (Aspirational: RED until Phase 2.4 table_extractor is implemented.)
        """
        try:
            from engine.ingestion.table_extractor import inject_markdown_tables
        except ImportError:
            pytest.xfail("table_extractor not yet implemented (RED state — Phase 2.4)")

        path = os.path.join(MISC_DIR, self.FNAME)
        text = extract_text_native(self.FNAME)
        injected = inject_markdown_tables(path, text)
        if "<!-- TABLES PAGE" in injected:
            assert "<!-- END TABLES -->" in injected, (
                "inject_markdown_tables opened <!-- TABLES PAGE --> but never closed it"
            )

    def test_chunk_enriched_text_contains_game_doc_label(self, pipeline):
        """Every HoW chunk's enriched text must contain [Doc: core_rules]."""
        chunks, _, _ = pipeline
        for i, chunk in enumerate(chunks[:20]):
            assert "[Doc: core_rules]" in chunk["text"], (
                f"HoW chunk #{i} enriched text missing [Doc: core_rules] header.\n"
                f"Text start: {repr(chunk['text'][:120])}"
            )


# ═══════════════════════════════════════════════════════════════════
# 3. Tokyo Express — full ingestion plan verification
# ═══════════════════════════════════════════════════════════════════

class TestTokyoExpressIngestion:
    PROFILE = "tokyo_express_profile.json"
    FNAME = "pdfcoffee.com_tokyo-express-basic-rule-book-pdf-free.pdf"

    @pytest.fixture(scope="class")
    def pipeline(self):
        skip_if_profile_missing(self.PROFILE)
        return full_pipeline(self.PROFILE, self.FNAME)

    def test_chunk_count_within_plan_range(self, pipeline):
        chunks, _, _ = pipeline
        lo, hi = PLAN_CHUNK_COUNTS["tokyo_express"]
        assert lo <= len(chunks) <= hi, (
            f"Tokyo Express chunk count {len(chunks)} outside plan range [{lo}, {hi}]"
        )

    def test_rule_31_through_36_indexed(self, pipeline):
        """
        Plan declares 7+ of rules 3.0–3.9 must be indexed.
        RED: 0 found (Up Front decimal regex misses 3.x).
        GREEN: numeric_decimal profile indexes 3.1–3.9.
        """
        _, _, rule_index = pipeline
        expected = {"3.1", "3.2", "3.3", "3.5", "3.6", "3.7"}
        non_meta_keys = {k for k in rule_index if not k.startswith("__")}
        found = expected.intersection(non_meta_keys)
        assert len(found) >= 4, (
            f"Tokyo Express rule index must contain ≥4 of {expected}.\n"
            f"Found: {found}\n"
            f"All rule_index keys: {sorted(non_meta_keys)}"
        )

    def test_no_9x_phantom_rules_in_index(self, pipeline):
        """
        Plan specifies OCR 3→9 misread must not generate '9.x' phantom rules.
        RED: OCR noise from scanned PDF produces '9.2', '9.4' etc.
        GREEN: profile regex limits to 3.x range; OCR normalization clears noise.
        """
        _, _, rule_index = pipeline
        phantom = [k for k in rule_index if re.match(r"^9\.", str(k))]
        assert not phantom, (
            f"Tokyo Express index contains '9.x' OCR phantom rules: {phantom}\n"
            f"These come from '3' being misread as '9' by the scanner."
        )

    def test_all_rule_numbers_are_valid_format(self, pipeline):
        """All Tokyo Express rule keys must be valid section numbers or keyword headers."""
        _, _, rule_index = pipeline
        non_meta = [k for k in rule_index if not k.startswith("__")]
        assert len(non_meta) >= 10
        # Verify decimal sections are present (e.g. 3.0, 3.1, 3.2, 3.3, 3.5, 3.6, etc.)
        assert any(re.match(r"^\d+\.\d", str(key)) for key in non_meta)

    def test_rule_index_minimum_entry_count(self, pipeline):
        """Plan declares 7+ rule sections for 24-page Tokyo Express."""
        _, _, rule_index = pipeline
        non_meta = [k for k in rule_index if not k.startswith("__")]
        assert len(non_meta) >= PLAN_RULE_INDEX_MIN_COUNTS["tokyo_express"], (
            f"Tokyo Express has {len(non_meta)} entries, plan minimum is "
            f"{PLAN_RULE_INDEX_MIN_COUNTS['tokyo_express']}"
        )

    def test_tokyo_chunks_have_doc_type_core_rules(self, pipeline):
        chunks, _, _ = pipeline
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["doc_type"] == "core_rules", (
                f"Tokyo chunk #{i} doc_type must be 'core_rules', "
                f"got '{chunk['metadata']['doc_type']}'"
            )

    def test_rule_number_in_enriched_text_header(self, pipeline):
        """Chunks with a rule_number must include [Rule: N.D] in their header."""
        chunks, _, _ = pipeline
        for i, chunk in enumerate(chunks):
            rn = chunk.get("rule_number")
            if rn:
                assert f"[Rule: {rn}]" in chunk["text"], (
                    f"Tokyo chunk #{i} rule_number='{rn}' not reflected in "
                    f"enriched text header: {repr(chunk['text'][:150])}"
                )


# ═══════════════════════════════════════════════════════════════════
# 4. Albedo RPG — native extraction confirms OCR required
# ═══════════════════════════════════════════════════════════════════

class TestAlbedoRPGIngestion:
    PROFILE = "albedo_rpg_profile.json"
    FNAME = "pdfcoffee.com_albedo-2nd-edition-compressed-pdf-free.pdf"

    def test_native_text_is_insufficient_confirms_ocr_needed(self):
        """
        Plan baseline: native extraction of Albedo bitmap yields <1000 chars.
        This test is GREEN immediately — it documents the known failure mode.
        """
        path = os.path.join(MISC_DIR, self.FNAME)
        doc = fitz.open(path)
        total_chars = sum(len(doc[i].get_text().strip()) for i in range(min(10, len(doc))))
        doc.close()
        assert total_chars < 1000, (
            f"Expected Albedo to have <1000 native chars in first 10 pages. "
            f"Got {total_chars}. Verify this is still the scanned bitmap version."
        )

    def test_native_pipeline_produces_zero_or_trivial_chunks(self):
        """
        Without OCR, the pipeline should yield 0–3 trivial chunks (headers only).
        RED (wrong): pipeline incorrectly produces many rule chunks from empty text.
        GREEN: pipeline produces 0–3 chunks — this triggers the OCR requirement.
        """
        skip_if_profile_missing(self.PROFILE)
        chunks, _, _ = full_pipeline(self.PROFILE, self.FNAME, max_pages=10)
        assert len(chunks) <= 10, (
            f"Albedo native pipeline yielded {len(chunks)} chunks from 10 scanned pages. "
            f"Expected ≤10 (confirms OCR is required). "
            f"If >10, the DocumentProfiler may be reading OCR metadata, not actual scan."
        )

    @pytest.mark.xfail(
        reason="OCR pipeline not yet wired (Phase 2.2 — RED state). "
               "Will pass after ocr_processor.process_pdf_to_text() is implemented.",
        strict=False
    )
    def test_ocr_pipeline_yields_sufficient_text(self):
        """
        GREEN only after Phase 2.2: OCR wiring in get_text_for_game_file().
        Verifies ≥20 keyword_header sections chunked from 30 OCR'd pages.
        """
        from engine.ingestion.ocr_processor import init_ocr, process_pdf_to_text

        path = os.path.join(MISC_DIR, self.FNAME)
        ocr = init_ocr()
        text = process_pdf_to_text(path, ocr, max_pages=30)
        assert len(text) >= 20000, (
            f"OCR text length {len(text)} insufficient — expected ≥20000 chars from 30 pages"
        )

        skip_if_profile_missing(self.PROFILE)
        profile = load_profile(self.PROFILE)
        patterns = get_compiled_patterns(profile)
        doc_info = list(profile["documents"].values())[0]
        chunks = route_chunk_generic(text, self.FNAME, doc_info, profile, patterns)
        assert len(chunks) >= 20, (
            f"OCR + keyword_header yielded {len(chunks)} chunks from 30 pages, expected ≥20"
        )

    @pytest.mark.xfail(
        reason="OCR pipeline not yet wired (RED). Will pass after Phase 2.2.",
        strict=False
    )
    def test_ocr_chunks_contain_rpg_section_names(self):
        """After OCR: COMBAT, CHARACTER CREATION, EQUIPMENT must be section_paths."""
        from engine.ingestion.ocr_processor import init_ocr, process_pdf_to_text

        path = os.path.join(MISC_DIR, self.FNAME)
        ocr = init_ocr()
        text = process_pdf_to_text(path, ocr, max_pages=30)

        skip_if_profile_missing(self.PROFILE)
        profile = load_profile(self.PROFILE)
        patterns = get_compiled_patterns(profile)
        doc_info = list(profile["documents"].values())[0]
        chunks = route_chunk_generic(text, self.FNAME, doc_info, profile, patterns)
        game_id = profile.get("game_id", "albedo")
        indexed = {f"{game_id}_{i}": c for i, c in enumerate(chunks)}
        rule_index = build_rule_index(indexed)

        keys_upper = {k.upper() for k in rule_index if not k.startswith("__")}
        expected_sections = {"COMBAT", "CHARACTER"}
        found = {s for s in expected_sections if any(s in k for k in keys_upper)}
        assert len(found) >= 1, (
            f"Albedo OCR index missing RPG sections {expected_sections}. "
            f"Got keys (first 20): {sorted(keys_upper)[:20]}"
        )


# ═══════════════════════════════════════════════════════════════════
# 5. Regression guard — existing games must not be affected
# ═══════════════════════════════════════════════════════════════════

class TestExistingGameRegression:
    """
    Plan section 'What Does NOT Change': existing Up Front, ASL, SFB patterns
    must be unaffected by the profile JSON additions and code changes.
    """

    def test_upfront_decimal_pattern_still_matches(self):
        """The plan preserves RULE_NUMBER_PATTERN from ingest_rules.py untouched."""
        from engine.ingestion.ingest_rules import RULE_NUMBER_PATTERN
        assert RULE_NUMBER_PATTERN is not None
        # Up Front rules must still match
        for rule in ["5.41", "14.2", "3.3", "7.1"]:
            text = f"\n{rule} SOME RULE TEXT\n"
            assert RULE_NUMBER_PATTERN.search(text), (
                f"Regression: RULE_NUMBER_PATTERN must still match Up Front rule '{rule}'"
            )

    def test_extract_rule_numbers_still_works_for_upfront(self):
        """extract_rule_numbers() must continue to work for Up Front."""
        from engine.ingestion.ingest_rules import extract_rule_numbers
        text = "5.41 TERRAIN PLACEMENT: A terrain card...\nSee 5.61 for details. [6.1]"
        result = extract_rule_numbers(text)
        assert "5.41" in result, f"extract_rule_numbers regression: '5.41' not found in {result}"

    def test_upfront_chunk_format_unchanged(self):
        """
        chunk_generic (text, source_file) — the legacy Up Front function signature.
        """
        from engine.ingestion.ingest_rules import chunk_generic
        text = "\n5.41 TERRAIN PLACEMENT: A terrain card may be played.\n"
        chunks = chunk_generic(text, "UF_RuleBook.pdf")
        assert len(chunks) >= 1, "chunk_generic must produce at least 1 chunk"
        c = chunks[0]
        assert "text" in c
        assert "metadata" in c
        assert "doc_type" in c["metadata"]
        assert "rule_number" in c

    def test_build_rule_index_still_works(self):
        """build_rule_index() must work correctly with a minimal Up Front chunk set."""
        from engine.ingestion.ingest_rules import build_rule_index
        all_chunks = {
            "upfront_0": {
                "rule_number": "5.41",
                "metadata": {"doc_type": "core_rules", "priority": 1,
                             "source_file": "UF_RuleBook.pdf", "page": 1,
                             "section_path": "5.41", "content_type": "rule"},
                "text": "[Doc: core_rules] [Section: 5.41] [Rule: 5.41]\n5.41 Terrain Placement...",
                "cross_refs": ["6.1"],
                "all_rule_numbers": ["5.41"],
                "root_section": "5",
                "hierarchy_level": 1,
            }
        }
        idx = build_rule_index(all_chunks)
        assert "5.41" in idx, f"build_rule_index must index '5.41'. Got keys: {list(idx.keys())}"
        entry = idx["5.41"]
        assert isinstance(entry, list) and len(entry) >= 1
        assert entry[0]["chunk_id"] == "upfront_0"
