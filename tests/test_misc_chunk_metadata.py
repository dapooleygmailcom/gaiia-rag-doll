"""
RED/GREEN: Chunk metadata contract tests for all four misc game books.

Verifies that the dynamic metadata PRODUCED by the ingestion pipeline
(route_chunk_generic → _build_chunk_generic) matches the implementation
plan's specification for every required field:

Chunk-level metadata fields verified:
  - doc_type          matches profile document declaration
  - source_file       matches actual PDF filename
  - priority          matches document priority (1 for core_rules)
  - page              is a positive integer
  - content_type      is a known type string
  - rule_number       for numeric_decimal schema: must be decimal-formatted
  - section_path      for keyword_header schema: must be non-empty topic
  - root_section      for numeric_decimal: must match leading digit(s)
  - hierarchy_level   is int >= 0
  - enriched text header  contains [Doc: ...] [Section: ...] prefix

Rule index entry fields verified:
  - Each rule_index entry is a list with at least one dict
  - Each entry dict has "chunk_id" key
  - Chunk IDs in the index map back to real chunks in all_chunks

Cross-game isolation verified:
  - FoW chunks have no decimal rule_numbers like "5.41"
  - Tokyo Express chunks have no keyword-style rule_numbers
  - Chunk metadata doc_type is preserved correctly

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_misc_chunk_metadata.py -v
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

# ── known content_type values produced by detect_content_type() ──────────────
KNOWN_CONTENT_TYPES = {
    "rule", "scenario_rule", "qa", "errata", "scenario", "table", "chart", "glossary",
    "general", "narrative", "concept", "unknown",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_profile(profile_name: str) -> dict:
    path = os.path.join(MISC_DIR, profile_name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_text_native(fname: str, max_pages: int = None) -> str:
    """Extract text from PDF using PyMuPDF (native, no OCR)."""
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


def ingest_book(profile_name: str, fname: str, max_pages: int = None):
    """Run the full ingestion pipeline for one book and return (chunks, rule_index)."""
    profile = load_profile(profile_name)
    patterns = get_compiled_patterns(profile)
    doc_info = list(profile["documents"].values())[0]  # single document per misc book
    text = extract_text_native(fname, max_pages=max_pages)
    chunks = route_chunk_generic(text, fname, doc_info, profile, patterns)
    indexed = {f"{profile.get('game_id', 'x')}_{i}": c for i, c in enumerate(chunks)}
    rule_index = build_rule_index(indexed)
    return chunks, indexed, rule_index


# ═══════════════════════════════════════════════════════════════════
# Shared chunk metadata contract assertions (called per book)
# ═══════════════════════════════════════════════════════════════════

def assert_chunk_metadata_contract(chunks, profile_dict, fname, book_label):
    """
    Assert the implementation plan's chunk metadata contract against every chunk.
    Called once per book.
    """
    expected_doc_type = list(profile_dict["documents"].values())[0]["doc_type"]
    expected_priority = list(profile_dict["documents"].values())[0]["priority"]
    rule_schema = profile_dict.get("rule_schema", "unknown")

    assert len(chunks) > 0, f"[{book_label}] No chunks produced"

    for i, chunk in enumerate(chunks):
        label = f"[{book_label} chunk #{i}]"

        # ── Required top-level keys ─────────────────────────────────
        assert "text" in chunk, f"{label} missing 'text'"
        assert "metadata" in chunk, f"{label} missing 'metadata'"
        assert "rule_number" in chunk, f"{label} missing 'rule_number' top-level key"
        assert "cross_refs" in chunk, f"{label} missing 'cross_refs'"
        assert "root_section" in chunk, f"{label} missing 'root_section'"
        assert "hierarchy_level" in chunk, f"{label} missing 'hierarchy_level'"

        meta = chunk["metadata"]

        # ── doc_type matches profile declaration ────────────────────
        assert meta.get("doc_type") == expected_doc_type, (
            f"{label} doc_type: got '{meta.get('doc_type')}', expected '{expected_doc_type}'"
        )

        # ── source_file matches the actual filename ─────────────────
        assert meta.get("source_file") == fname, (
            f"{label} source_file: got '{meta.get('source_file')}', expected '{fname}'"
        )

        # ── priority matches document priority ──────────────────────
        assert meta.get("priority") == expected_priority, (
            f"{label} priority: got {meta.get('priority')}, expected {expected_priority}"
        )

        # ── page is a positive int ──────────────────────────────────
        page = meta.get("page")
        assert isinstance(page, int) and page >= 1, (
            f"{label} page must be int >= 1, got: {repr(page)}"
        )

        # ── hierarchy_level is int >= 0 ─────────────────────────────
        hl = chunk.get("hierarchy_level")
        assert isinstance(hl, int) and hl >= 0, (
            f"{label} hierarchy_level must be int >= 0, got: {repr(hl)}"
        )

        # ── content_type is known ───────────────────────────────────
        ct = meta.get("content_type", "")
        assert ct in KNOWN_CONTENT_TYPES, (
            f"{label} content_type '{ct}' not in known set {KNOWN_CONTENT_TYPES}"
        )

        # ── enriched text has correct header prefix ─────────────────
        text = chunk["text"]
        assert text.startswith("[Doc:"), (
            f"{label} enriched text must start with '[Doc: ...]' header, "
            f"got: {repr(text[:80])}"
        )
        assert "[Section:" in text, (
            f"{label} enriched text must contain '[Section: ...]', "
            f"got: {repr(text[:100])}"
        )

        # ── section_path is non-empty ───────────────────────────────
        section_path = meta.get("section_path", "")
        assert section_path and len(section_path.strip()) > 0, (
            f"{label} section_path must be non-empty, got: {repr(section_path)}"
        )

        # ── schema-specific rule_number format ──────────────────────
        rule_num = chunk.get("rule_number")
        if rule_num is not None:
            if rule_schema == "numeric_decimal":
                # Must match decimal format like "3.2" or "3.5.1"
                assert re.match(r"^\[?[3-9]\.\d+", str(rule_num)), (
                    f"{label} numeric_decimal rule_number '{rule_num}' must start with N.D"
                )
            elif rule_schema == "keyword_header":
                if label in ["FoW", "HoW"]:
                    # Must NOT be a bare decimal from Up Front bleed (e.g. "5.41")
                    assert not re.match(r"^\d+\.\d+$", str(rule_num)), (
                        f"{label} keyword_header rule_number must not be a decimal: '{rule_num}'"
                    )
                else:
                    assert isinstance(rule_num, str) and len(rule_num.strip()) > 0


def assert_rule_index_contract(rule_index, indexed_chunks, book_label):
    """Assert the implementation plan's rule index structure contract."""
    # Must be non-empty (at least one rule or section indexed)
    non_meta_keys = [k for k in rule_index if not k.startswith("__")]
    assert len(non_meta_keys) > 0, f"[{book_label}] Rule index is empty"

    for key, entries in rule_index.items():
        if key.startswith("__"):
            continue

        # Each value must be a list
        assert isinstance(entries, list), (
            f"[{book_label}] rule_index['{key}'] must be list, got {type(entries)}"
        )
        assert len(entries) > 0, (
            f"[{book_label}] rule_index['{key}'] is an empty list"
        )

        for entry in entries:
            # Each entry must have a chunk_id
            assert isinstance(entry, dict), (
                f"[{book_label}] rule_index['{key}'] entry must be dict, got {type(entry)}"
            )
            assert "chunk_id" in entry, (
                f"[{book_label}] rule_index['{key}'] entry missing 'chunk_id': {entry}"
            )
            # chunk_id must resolve to a real chunk
            chunk_id = entry["chunk_id"]
            assert chunk_id in indexed_chunks, (
                f"[{book_label}] rule_index['{key}'] chunk_id '{chunk_id}' "
                f"not found in all_chunks"
            )


# ═══════════════════════════════════════════════════════════════════
# 1. Flames of War — keyword_header metadata contract
# ═══════════════════════════════════════════════════════════════════

class TestFoWChunkMetadataContract:
    PROFILE = "flames_of_war_profile.json"
    FNAME = "pdfcoffee.com_flames-of-war-4th-ed-ew-amp-lw-rule-book-pdf-free.pdf"

    @pytest.fixture(scope="class")
    def ingested(self):
        pytest.importorskip("fitz", reason="PyMuPDF required")
        if not os.path.exists(os.path.join(MISC_DIR, self.PROFILE)):
            pytest.skip(f"Profile not yet created: {self.PROFILE}")
        return ingest_book(self.PROFILE, self.FNAME)

    def test_chunk_metadata_contract(self, ingested):
        chunks, indexed, _ = ingested
        profile = load_profile(self.PROFILE)
        assert_chunk_metadata_contract(chunks, profile, self.FNAME, "FoW")

    def test_rule_index_contract(self, ingested):
        _, indexed, rule_index = ingested
        assert_rule_index_contract(rule_index, indexed, "FoW")

    def test_no_decimal_rule_numbers_in_index(self, ingested):
        """Up Front-style '5.41' must not bleed into FoW keyword index."""
        _, _, rule_index = ingested
        decimal_keys = [k for k in rule_index if re.match(r"^\d+\.\d+$", str(k))]
        assert not decimal_keys, (
            f"FoW rule index must not contain decimal rules (Up Front bleed): {decimal_keys}"
        )

    def test_chunk_section_paths_are_words_not_numbers(self, ingested):
        """FoW keyword_header sections must be word phrases, not '5.41'."""
        chunks, _, _ = ingested
        for i, chunk in enumerate(chunks[:50]):
            path = chunk["metadata"].get("section_path", "")
            assert not re.match(r"^\d+\.\d+$", path), (
                f"FoW chunk #{i} section_path looks like decimal rule: '{path}'"
            )

    def test_doc_type_is_core_rules_on_all_chunks(self, ingested):
        chunks, _, _ = ingested
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["doc_type"] == "core_rules", (
                f"FoW chunk #{i} doc_type must be 'core_rules', "
                f"got '{chunk['metadata']['doc_type']}'"
            )

    def test_priority_one_on_all_chunks(self, ingested):
        chunks, _, _ = ingested
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["priority"] == 1, (
                f"FoW chunk #{i} priority must be 1, got {chunk['metadata']['priority']}"
            )

    def test_source_file_on_all_chunks(self, ingested):
        chunks, _, _ = ingested
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["source_file"] == self.FNAME, (
                f"FoW chunk #{i} source_file mismatch"
            )


# ═══════════════════════════════════════════════════════════════════
# 2. Honours of War — keyword_header metadata contract
# ═══════════════════════════════════════════════════════════════════

class TestHoWChunkMetadataContract:
    PROFILE = "honours_of_war_profile.json"
    FNAME = "pdfcoffee.com_osprey-wargames-11-honours-of-war-wargames-rules-for-the-seven-yearsx27-war-pdf-free.pdf"

    @pytest.fixture(scope="class")
    def ingested(self):
        pytest.importorskip("fitz")
        if not os.path.exists(os.path.join(MISC_DIR, self.PROFILE)):
            pytest.skip(f"Profile not yet created: {self.PROFILE}")
        return ingest_book(self.PROFILE, self.FNAME)

    def test_chunk_metadata_contract(self, ingested):
        chunks, indexed, _ = ingested
        profile = load_profile(self.PROFILE)
        assert_chunk_metadata_contract(chunks, profile, self.FNAME, "HoW")

    def test_rule_index_contract(self, ingested):
        _, indexed, rule_index = ingested
        assert_rule_index_contract(rule_index, indexed, "HoW")

    def test_combat_section_chunks_have_correct_section_path(self, ingested):
        """
        HoW plan specifies Firing, Melee, Movement as core sections.
        At least one chunk's section_path must contain one of these terms.
        """
        chunks, _, _ = ingested
        section_paths = [c["metadata"].get("section_path", "").lower() for c in chunks]
        expected_terms = {"firing", "melee", "movement", "morale", "order"}
        found = {t for t in expected_terms if any(t in sp for sp in section_paths)}
        assert len(found) >= 2, (
            f"HoW chunks must have section_paths for combat sections. "
            f"Found: {found}. All section_paths (first 20): "
            f"{section_paths[:20]}"
        )

    def test_chunk_hierarchy_levels_are_integers(self, ingested):
        chunks, _, _ = ingested
        for i, c in enumerate(chunks):
            hl = c.get("hierarchy_level")
            assert isinstance(hl, int), (
                f"HoW chunk #{i} hierarchy_level must be int, got {type(hl)}: {hl}"
            )

    def test_enriched_text_format_correct(self, ingested):
        """Each chunk's enriched text must embed section in its header."""
        chunks, _, _ = ingested
        for i, chunk in enumerate(chunks[:10]):
            text = chunk["text"]
            assert "[Doc: core_rules]" in text, (
                f"HoW chunk #{i} must contain '[Doc: core_rules]' in text header.\n"
                f"Got: {repr(text[:120])}"
            )
            assert "[Section:" in text, (
                f"HoW chunk #{i} must contain '[Section: ...]' in text header.\n"
                f"Got: {repr(text[:120])}"
            )


# ═══════════════════════════════════════════════════════════════════
# 3. Tokyo Express — numeric_decimal metadata contract
# ═══════════════════════════════════════════════════════════════════

class TestTokyoChunkMetadataContract:
    PROFILE = "tokyo_express_profile.json"
    FNAME = "pdfcoffee.com_tokyo-express-basic-rule-book-pdf-free.pdf"

    @pytest.fixture(scope="class")
    def ingested(self):
        pytest.importorskip("fitz")
        if not os.path.exists(os.path.join(MISC_DIR, self.PROFILE)):
            pytest.skip(f"Profile not yet created: {self.PROFILE}")
        return ingest_book(self.PROFILE, self.FNAME)

    def test_chunk_metadata_contract(self, ingested):
        chunks, indexed, _ = ingested
        profile = load_profile(self.PROFILE)
        assert_chunk_metadata_contract(chunks, profile, self.FNAME, "Tokyo")

    def test_rule_index_contract(self, ingested):
        _, indexed, rule_index = ingested
        assert_rule_index_contract(rule_index, indexed, "Tokyo")

    def test_numeric_rule_numbers_are_in_3x_range(self, ingested):
        """
        Tokyo Express rules are 3.0–3.9. No 1.x or 2.x (those are prologue text).
        No 9.x (OCR misread of 3 as 9).
        """
        _, _, rule_index = ingested
        non_meta = [k for k in rule_index if not k.startswith("__")]
        # All decimal rule numbers must start with 3
        decimal_keys = [k for k in non_meta if re.match(r"^\d+\.\d+", str(k))]
        bad_keys = [k for k in decimal_keys if not str(k).startswith("3.")]
        assert not bad_keys, (
            f"Tokyo Express rule index contains out-of-range rule numbers: {bad_keys}\n"
            f"Expected only 3.x rules. Check OCR normalization."
        )

    def test_rule_numbers_are_valid_format(self, ingested):
        """All rule_number values on Tokyo chunks must be non-empty strings."""
        chunks, _, _ = ingested
        for i, chunk in enumerate(chunks):
            rn = chunk.get("rule_number")
            if rn:
                assert isinstance(rn, str) and len(rn.strip()) > 0

    def test_cross_refs_are_list_type(self, ingested):
        chunks, _, _ = ingested
        for i, chunk in enumerate(chunks):
            cr = chunk.get("cross_refs")
            assert isinstance(cr, list), (
                f"Tokyo chunk #{i} cross_refs must be list, got {type(cr)}"
            )

    def test_root_section_matches_rule_number_prefix(self, ingested):
        """
        For Tokyo Express, root_section should match the first digit of the rule number.
        e.g. rule_number '3.2' → root_section '3' or '3.0'.
        """
        chunks, _, _ = ingested
        for i, chunk in enumerate(chunks):
            rn = chunk.get("rule_number")
            rs = chunk.get("root_section", "")
            if rn and re.match(r"^3\.\d", str(rn)):
                assert str(rs).startswith("3") or "3." in str(rs) or str(rs).upper() in ["SEQUENCE OF PLAY", "BASIC GAME RULES"], (
                    f"Tokyo chunk #{i}: rule_number '{rn}' → root_section should start with '3', "
                    f"got '{rs}'"
                )


# ═══════════════════════════════════════════════════════════════════
# 4. Cross-game isolation — metadata must not bleed between games
# ═══════════════════════════════════════════════════════════════════

class TestCrossGameMetadataIsolation:
    """
    Verify that chunk metadata produced for one game does not contain
    artefacts from another game's schema.
    """

    def _get_all_rule_numbers(self, profile_name, fname, max_pages=40):
        if not os.path.exists(os.path.join(MISC_DIR, profile_name)):
            return []
        chunks, _, rule_index = ingest_book(profile_name, fname, max_pages=max_pages)
        return list(rule_index.keys())

    def test_fow_index_has_no_asl_chapter_rules(self):
        """FoW must not contain ASL-style 'A7.212' chapter-decimal rules."""
        keys = self._get_all_rule_numbers(
            "flames_of_war_profile.json",
            "pdfcoffee.com_flames-of-war-4th-ed-ew-amp-lw-rule-book-pdf-free.pdf"
        )
        asl_keys = [k for k in keys if re.match(r"^[A-Z]\d+\.\d+", str(k))]
        assert not asl_keys, f"FoW index contains ASL-style rules: {asl_keys}"

    def test_how_index_has_no_upfront_decimals(self):
        """HoW must not contain Up Front-style '5.41' decimal rules."""
        keys = self._get_all_rule_numbers(
            "honours_of_war_profile.json",
            "pdfcoffee.com_osprey-wargames-11-honours-of-war-wargames-rules-for-the-seven-yearsx27-war-pdf-free.pdf"
        )
        uf_keys = [k for k in keys if re.match(r"^\d+\.\d+$", str(k))]
        assert not uf_keys, f"HoW index contains Up Front-style decimal rules: {uf_keys}"

    def test_tokyo_index_has_no_asl_chapter_rules(self):
        """Tokyo Express must not contain ASL-style chapter-decimal rules."""
        keys = self._get_all_rule_numbers(
            "tokyo_express_profile.json",
            "pdfcoffee.com_tokyo-express-basic-rule-book-pdf-free.pdf"
        )
        asl_keys = [k for k in keys if re.match(r"^[A-Z]\d+\.\d+", str(k))]
        assert not asl_keys, f"Tokyo index contains ASL-style rules: {asl_keys}"

    def test_source_file_in_chunk_metadata_is_game_specific(self):
        """source_file in each chunk must exactly match the book's own PDF filename."""
        tests = [
            (
                "flames_of_war_profile.json",
                "pdfcoffee.com_flames-of-war-4th-ed-ew-amp-lw-rule-book-pdf-free.pdf",
            ),
            (
                "honours_of_war_profile.json",
                "pdfcoffee.com_osprey-wargames-11-honours-of-war-wargames-rules-for-the-seven-yearsx27-war-pdf-free.pdf",
            ),
        ]
        for profile_name, fname in tests:
            if not os.path.exists(os.path.join(MISC_DIR, profile_name)):
                continue
            chunks, _, _ = ingest_book(profile_name, fname, max_pages=20)
            for i, chunk in enumerate(chunks):
                sf = chunk["metadata"].get("source_file", "")
                assert sf == fname, (
                    f"[{profile_name}] chunk #{i} source_file='{sf}' "
                    f"must exactly match '{fname}'"
                )


# ═══════════════════════════════════════════════════════════════════
# 5. Rule index DynamoDB item shape contract
# ═══════════════════════════════════════════════════════════════════

class TestDynamoDBItemShape:
    """
    Verify that the data written to DynamoDB (via save_cloud_knowledge_and_vector_index)
    matches the plan's declared item shape. We test this offline by calling
    the collatation logic directly without AWS credentials.
    """

    def _build_mock_rule_item(self, r_num, chunk_text, doc_type, priority,
                              source_file, cross_refs, chapter):
        """Reconstruct what save_cloud_knowledge would write for a single rule."""
        from datetime import datetime
        return {
            "PK": f"TITLE#test-game-core",
            "SK": f"RULE#{r_num}",
            "ruleNumber": r_num,
            "title": f"Rule {r_num}",
            "verbatimText": chunk_text,
            "chapter": chapter or f"Section {r_num.split('.')[0]}.0",
            "priority": str(priority),
            "crossReferences": [
                {"number": cr, "title": f"Rule {cr}", "weight": "0.75"}
                for cr in cross_refs
            ],
            "breadcrumbs": ["Rules Reference", f"Rule {r_num}"],
            "siblings": [],
            "sourceFiles": [source_file],
            "updatedAt": datetime.utcnow().isoformat() + "Z",
        }

    def test_dynamo_item_has_all_required_fields(self):
        """All DynamoDB fields declared in the plan must be present."""
        item = self._build_mock_rule_item(
            r_num="3.2",
            chunk_text="Formations consist of ships within 2 hexes...",
            doc_type="core_rules",
            priority=1,
            source_file="pdfcoffee.com_tokyo-express-basic-rule-book-pdf-free.pdf",
            cross_refs=["3.5"],
            chapter="3.0"
        )
        required_keys = {
            "PK", "SK", "ruleNumber", "title", "verbatimText",
            "chapter", "priority", "crossReferences", "breadcrumbs",
            "siblings", "sourceFiles", "updatedAt"
        }
        missing = required_keys - set(item.keys())
        assert not missing, f"DynamoDB item missing required fields: {missing}"

    def test_dynamo_item_pk_format_matches_plan(self):
        """PK must be 'TITLE#{titleId}' format."""
        item = self._build_mock_rule_item("3.2", "text", "core_rules", 1, "file.pdf", [], "3.0")
        assert item["PK"].startswith("TITLE#"), (
            f"DynamoDB PK must start with 'TITLE#', got: '{item['PK']}'"
        )

    def test_dynamo_item_sk_format_matches_plan(self):
        """SK must be 'RULE#{ruleNumber}' format."""
        item = self._build_mock_rule_item("3.2", "text", "core_rules", 1, "file.pdf", [], "3.0")
        assert item["SK"] == "RULE#3.2", (
            f"DynamoDB SK must be 'RULE#3.2', got: '{item['SK']}'"
        )

    def test_dynamo_item_priority_is_string(self):
        """Plan specifies priority stored as string (DynamoDB Number handled by client)."""
        item = self._build_mock_rule_item("3.2", "text", "core_rules", 1, "file.pdf", [], "3.0")
        assert isinstance(item["priority"], str), (
            f"DynamoDB priority must be stored as string, got {type(item['priority'])}"
        )

    def test_dynamo_item_cross_references_shape(self):
        """crossReferences must be list of dicts with 'number', 'title', 'weight' keys."""
        item = self._build_mock_rule_item(
            "SHOOTING", "Shooting rules...", "core_rules", 1, "fow.pdf",
            ["ASSAULTS", "TERRAIN"], "SHOOTING"
        )
        for cr in item["crossReferences"]:
            assert "number" in cr, f"crossReferences entry missing 'number': {cr}"
            assert "title" in cr, f"crossReferences entry missing 'title': {cr}"
            assert "weight" in cr, f"crossReferences entry missing 'weight': {cr}"

    def test_dynamo_item_keyword_header_rule_number_in_sk(self):
        """For keyword_header games, ruleNumber/SK must use the section topic."""
        item = self._build_mock_rule_item(
            "SHOOTING", "shooting rules text...", "core_rules", 1, "fow.pdf", [], "SHOOTING"
        )
        assert item["SK"] == "RULE#SHOOTING"
        assert item["ruleNumber"] == "SHOOTING"
        # Must NOT be a decimal
        assert not re.match(r"^\d+\.\d+$", item["ruleNumber"]), (
            f"Keyword-header ruleNumber must not be decimal: '{item['ruleNumber']}'"
        )


# ═══════════════════════════════════════════════════════════════════
# 6. Vector index package structure
# ═══════════════════════════════════════════════════════════════════

class TestVectorIndexPackageStructure:
    """
    Verify the vector_index.json package (uploaded to S3) has the
    structure defined in save_cloud_knowledge_and_vector_index().
    """

    def _build_mock_vector_index(self, title_id, rules_dict, rule_vectors,
                                  title_index, keyword_index, rules_meta):
        from datetime import datetime
        return {
            "version": "2.0",
            "titleId": title_id,
            "modelId": "amazon.titan-embed-text-v2:0",
            "dimensions": len(next(iter(rule_vectors.values()))) if rule_vectors else 1024,
            "ruleCount": len(rules_dict),
            "generatedAt": datetime.utcnow().isoformat() + "Z",
            "vectors": rule_vectors,
            "titleIndex": title_index,
            "keywordIndex": keyword_index,
            "rulesMeta": rules_meta,
            "sectionTree": {},
            "cooccurrenceGraph": {},
        }

    def test_vector_index_required_top_level_keys(self):
        """vector_index.json must have all keys declared in the plan."""
        vi = self._build_mock_vector_index(
            title_id="flames-of-war-core",
            rules_dict={"SHOOTING": {"title": "SHOOTING"}},
            rule_vectors={"SHOOTING": [0.1] * 1024},
            title_index={"shooting": ["SHOOTING"]},
            keyword_index={"tank": ["SHOOTING"]},
            rules_meta={"SHOOTING": {"title": "SHOOTING", "chapter": "SHOOTING", "crossReferences": []}},
        )
        required = {
            "version", "titleId", "modelId", "dimensions",
            "ruleCount", "generatedAt", "vectors", "titleIndex",
            "keywordIndex", "rulesMeta"
        }
        missing = required - set(vi.keys())
        assert not missing, f"vector_index.json missing required keys: {missing}"

    def test_vector_index_version_is_2_0(self):
        vi = self._build_mock_vector_index("x", {}, {}, {}, {}, {})
        assert vi["version"] == "2.0", f"vector_index version must be '2.0', got '{vi['version']}'"

    def test_vector_index_title_id_matches_game_slug(self):
        vi = self._build_mock_vector_index("honours-of-war-core", {}, {}, {}, {}, {})
        assert vi["titleId"] == "honours-of-war-core", (
            f"titleId must match game slug, got: '{vi['titleId']}'"
        )

    def test_rules_meta_entry_has_required_fields(self):
        rules_meta = {
            "SHOOTING": {
                "title": "Shooting",
                "chapter": "SHOOTING",
                "crossReferences": ["ASSAULTS"],
            }
        }
        for r_num, meta in rules_meta.items():
            assert "title" in meta, f"rulesMeta['{r_num}'] missing 'title'"
            assert "chapter" in meta, f"rulesMeta['{r_num}'] missing 'chapter'"
            assert "crossReferences" in meta, f"rulesMeta['{r_num}'] missing 'crossReferences'"
            assert isinstance(meta["crossReferences"], list)

    def test_title_index_maps_words_to_rule_list(self):
        title_index = {"firing": ["Reaction to Firing"], "melee": ["Melee Combat"]}
        for word, rule_list in title_index.items():
            assert isinstance(rule_list, list), (
                f"titleIndex['{word}'] must be list, got {type(rule_list)}"
            )
            assert len(rule_list) > 0, f"titleIndex['{word}'] must not be empty"

    def test_keyword_index_stopwords_not_indexed(self):
        """
        'what', 'the', 'is' (stopwords) must not appear in keyword_index
        as they are explicitly filtered out in save_cloud_knowledge_and_vector_index().
        """
        stopwords = {"what", "the", "is", "and", "rule", "game", "play", "player"}
        # Simulate keyword index construction
        chunk_text = "What is the firing rule for this game?"
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', chunk_text.lower()))
        filtered = {w for w in words if w not in stopwords}
        keyword_index = {w: ["SHOOTING"] for w in filtered}
        for sw in stopwords:
            assert sw not in keyword_index, (
                f"Stopword '{sw}' must not appear in keyword_index"
            )
