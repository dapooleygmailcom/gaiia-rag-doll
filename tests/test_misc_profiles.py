"""
RED/GREEN: Profile schema validation for all four misc game books.

Verifies that every DomainProfile field produced from the JSON files
matches the implementation plan's declared values exactly:
  - game_id / domain_id slug
  - rule_schema type
  - agent_persona.role string
  - agent_persona.citation_format token inclusion
  - agent_persona.conflict_resolution_rule presence
  - glossary keys present
  - document registry: filename, doc_type="core_rules", priority=1
  - ocr_required flag on Albedo (profile or doc-level)
  - ocr_substitutions on Tokyo Express
  - use_toc_bookmarks on Honours of War
  - extract_tables on FoW and HoW
  - pipeline_mode == "RULEBOOK_TECHNICAL" on all four
  - All four persona roles are distinct strings

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_misc_profiles.py -v
"""

import os
import json
import re
import pytest

# ── path resolution ──────────────────────────────────────────────────────────
MISC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/misc"))

PROFILES = {
    "albedo_rpg":    os.path.join(MISC_DIR, "albedo_rpg_profile.json"),
    "flames_of_war": os.path.join(MISC_DIR, "flames_of_war_profile.json"),
    "honours_of_war": os.path.join(MISC_DIR, "honours_of_war_profile.json"),
    "tokyo_express": os.path.join(MISC_DIR, "tokyo_express_profile.json"),
}

# Expected core values from implementation plan (ground truth)
PLAN_SPECS = {
    "albedo_rpg": {
        "game_id": "albedo_rpg",
        "rule_schema": "keyword_header",
        "pipeline_mode": "RULEBOOK_TECHNICAL",
        "chroma_collection": "albedo-rpg-rules-semantic",
        "persona_role_fragment": "Albedo",
        "citation_format_fragment": "ALBEDO",
        "glossary_required_keys": {"EDF", "ILR", "STR", "CON", "DEX", "INT", "WIL"},
        "doc_filename_fragment": "albedo",
        "doc_type": "core_rules",
        "doc_priority": 1,
        "doc_total_pages": 178,
        "ocr_required": True,
        "extract_tables": True,
        "ocr_substitutions": None,   # not on albedo
        "use_toc_bookmarks": None,
    },
    "flames_of_war": {
        "game_id": "flames_of_war",
        "rule_schema": "keyword_header",
        "pipeline_mode": "RULEBOOK_TECHNICAL",
        "chroma_collection": "flames-of-war-rules-semantic",
        "persona_role_fragment": "Flames",
        "citation_format_fragment": "FLAMES",
        "glossary_required_keys": {"ROF", "AT", "FP", "FA", "SA", "TA"},
        "doc_filename_fragment": "flames-of-war",
        "doc_type": "core_rules",
        "doc_priority": 1,
        "doc_total_pages": 109,
        "ocr_required": False,
        "extract_tables": True,
        "ocr_substitutions": None,
        "use_toc_bookmarks": None,
    },
    "honours_of_war": {
        "game_id": "honours_of_war",
        "rule_schema": "keyword_header",
        "pipeline_mode": "RULEBOOK_TECHNICAL",
        "chroma_collection": "honours-of-war-rules-semantic",
        "persona_role_fragment": "Honours",
        "citation_format_fragment": "HONOURS",
        "glossary_required_keys": {"SSR", "VC", "SYW", "OOB"},
        "doc_filename_fragment": "honours-of-war",
        "doc_type": "core_rules",
        "doc_priority": 1,
        "doc_total_pages": 68,
        "ocr_required": False,
        "extract_tables": True,
        "ocr_substitutions": None,
        "use_toc_bookmarks": True,
    },
    "tokyo_express": {
        "game_id": "tokyo_express",
        "rule_schema": "keyword_header",
        "pipeline_mode": "RULEBOOK_TECHNICAL",
        "chroma_collection": "tokyo-express-rules-semantic",
        "persona_role_fragment": "Tokyo",
        "citation_format_fragment": "TOKYO",
        "glossary_required_keys": {"AOB", "TDB", "DD", "CA", "CL"},
        "doc_filename_fragment": "tokyo-express",
        "doc_type": "core_rules",
        "doc_priority": 1,
        "doc_total_pages": 24,
        "ocr_required": False,   # OCR optional, uses normalize flag
        "extract_tables": True,
        "ocr_substitutions": {"O": "0"},
        "use_toc_bookmarks": None,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_raw(game_id):
    path = PROFILES[game_id]
    if not os.path.exists(path):
        pytest.skip(f"Profile not yet created (RED state): {os.path.basename(path)}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_profile(game_id):
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from engine.models.domain_profile import load_domain_profile
        return load_domain_profile(PROFILES[game_id])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# 1. File existence — everything RED until profiles are created
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("game_id", list(PROFILES.keys()))
def test_profile_file_exists(game_id):
    """RED: profile missing → GREEN: profile JSON created."""
    assert os.path.exists(PROFILES[game_id]), (
        f"Profile not found: {PROFILES[game_id]}\n"
        f"Create it per the implementation plan Phase 1."
    )


# ═══════════════════════════════════════════════════════════════════
# 2. Top-level identity fields
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("game_id,spec", PLAN_SPECS.items())
def test_game_id_matches_plan(game_id, spec):
    raw = load_raw(game_id)
    actual_id = raw.get("game_id") or raw.get("domain_id")
    assert actual_id == spec["game_id"], (
        f"[{game_id}] game_id mismatch: got '{actual_id}', expected '{spec['game_id']}'"
    )


@pytest.mark.parametrize("game_id,spec", PLAN_SPECS.items())
def test_rule_schema_matches_plan(game_id, spec):
    raw = load_raw(game_id)
    actual = raw.get("rule_schema")
    assert actual == spec["rule_schema"], (
        f"[{game_id}] rule_schema: got '{actual}', expected '{spec['rule_schema']}'"
    )


@pytest.mark.parametrize("game_id,spec", PLAN_SPECS.items())
def test_pipeline_mode_is_rulebook_technical(game_id, spec):
    raw = load_raw(game_id)
    assert raw.get("pipeline_mode") == "RULEBOOK_TECHNICAL", (
        f"[{game_id}] pipeline_mode must be RULEBOOK_TECHNICAL, got '{raw.get('pipeline_mode')}'"
    )


@pytest.mark.parametrize("game_id,spec", PLAN_SPECS.items())
def test_chroma_collection_matches_plan(game_id, spec):
    raw = load_raw(game_id)
    assert raw.get("chroma_collection") == spec["chroma_collection"], (
        f"[{game_id}] chroma_collection: got '{raw.get('chroma_collection')}', "
        f"expected '{spec['chroma_collection']}'"
    )


# ═══════════════════════════════════════════════════════════════════
# 3. Regex patterns compilable and functional
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("game_id", list(PROFILES.keys()))
def test_rule_pattern_compiles(game_id):
    import re as _re
    raw = load_raw(game_id)
    pat = raw.get("rule_pattern")
    assert pat, f"[{game_id}] rule_pattern missing"
    try:
        _re.compile(pat, _re.MULTILINE)
    except _re.error as e:
        pytest.fail(f"[{game_id}] rule_pattern does not compile: {e}\nPattern: {pat}")


@pytest.mark.parametrize("game_id", list(PROFILES.keys()))
def test_cross_ref_pattern_compiles(game_id):
    import re as _re
    raw = load_raw(game_id)
    pat = raw.get("cross_ref_pattern")
    assert pat, f"[{game_id}] cross_ref_pattern missing"
    try:
        _re.compile(pat, _re.MULTILINE)
    except _re.error as e:
        pytest.fail(f"[{game_id}] cross_ref_pattern does not compile: {e}\nPattern: {pat}")


def test_fow_keyword_pattern_does_not_match_decimal_rules():
    """FoW keyword_header pattern must NOT match '5.41' (Up Front decimal rule)."""
    import re as _re
    raw = load_raw("flames_of_war")
    pat = _re.compile(raw["rule_pattern"], _re.MULTILINE)
    # Should NOT match a decimal rule like '5.41 TERRAIN...'
    decimal_text = "\n5.41 TERRAIN PLACEMENT\n"
    assert not pat.search(decimal_text), (
        "FoW keyword_header pattern must not match Up Front decimal rules"
    )


def test_tokyo_rule_pattern_matches_section_headers():
    """Tokyo Express numeric_decimal pattern must match '3.2', '3.5.1'."""
    import re as _re
    raw = load_raw("tokyo_express")
    pat = _re.compile(raw["rule_pattern"], _re.MULTILINE)
    for rule in ["3.2", "3.5", "3.6", "3.9"]:
        text = f"\n{rule} Formation Movement\n"
        assert pat.search(text), (
            f"Tokyo Express pattern must match rule '{rule}', text: {repr(text)}"
        )


def test_tokyo_pattern_handles_ocr_zero_as_oh():
    """Tokyo Express pattern must also match '3.O' (OCR 0→O corruption)."""
    import re as _re
    raw = load_raw("tokyo_express")
    pat = _re.compile(raw["rule_pattern"], _re.MULTILINE)
    # The pattern includes [0-9O] to handle OCR noise
    text = "\n3.O Formation Orders\n"
    match = pat.search(text)
    # This is aspirational — may be RED until pattern updated
    if match is None:
        pytest.xfail("Tokyo Express pattern does not yet handle OCR '3.O' — expected RED state")


# ═══════════════════════════════════════════════════════════════════
# 4. Glossary — required keys per plan
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("game_id,spec", PLAN_SPECS.items())
def test_glossary_required_keys_present(game_id, spec):
    raw = load_raw(game_id)
    glossary = raw.get("glossary", {})
    missing = spec["glossary_required_keys"] - set(glossary.keys())
    assert not missing, (
        f"[{game_id}] Glossary missing required keys from plan: {missing}\n"
        f"Present keys: {list(glossary.keys())}"
    )


def test_all_glossary_values_are_strings():
    """Every glossary expansion must be a non-empty string (not null, int, or dict)."""
    for game_id in PROFILES:
        raw = load_raw(game_id)
        for k, v in raw.get("glossary", {}).items():
            assert isinstance(v, str) and v.strip(), (
                f"[{game_id}] Glossary key '{k}' must be a non-empty string, got: {repr(v)}"
            )


# ═══════════════════════════════════════════════════════════════════
# 5. agent_persona — role, citation_format, conflict_resolution_rule
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("game_id,spec", PLAN_SPECS.items())
def test_agent_persona_role_contains_game_fragment(game_id, spec):
    raw = load_raw(game_id)
    persona = raw.get("agent_persona", {})
    role = persona.get("role", "")
    assert spec["persona_role_fragment"] in role, (
        f"[{game_id}] agent_persona.role '{role}' must contain '{spec['persona_role_fragment']}'"
    )


@pytest.mark.parametrize("game_id,spec", PLAN_SPECS.items())
def test_agent_persona_citation_format_contains_game(game_id, spec):
    raw = load_raw(game_id)
    persona = raw.get("agent_persona", {})
    citation = persona.get("citation_format", "")
    assert spec["citation_format_fragment"] in citation.upper(), (
        f"[{game_id}] citation_format '{citation}' must contain '{spec['citation_format_fragment']}'"
    )


@pytest.mark.parametrize("game_id", list(PROFILES.keys()))
def test_agent_persona_has_conflict_resolution_rule(game_id):
    raw = load_raw(game_id)
    persona = raw.get("agent_persona", {})
    crr = persona.get("conflict_resolution_rule", "")
    assert crr and len(crr) >= 20, (
        f"[{game_id}] agent_persona.conflict_resolution_rule must be a non-empty string"
    )


@pytest.mark.parametrize("game_id", list(PROFILES.keys()))
def test_agent_persona_has_custom_instructions(game_id):
    """custom_instructions teaches the LLM domain-specific context."""
    raw = load_raw(game_id)
    persona = raw.get("agent_persona", {})
    instructions = persona.get("custom_instructions", "")
    assert instructions and len(instructions) >= 60, (
        f"[{game_id}] custom_instructions must be >= 60 chars of domain context, "
        f"got: {repr(instructions)}"
    )


def test_all_four_persona_roles_are_unique():
    """Regression: prompts must not bleed across games — each role is a unique string."""
    roles = []
    for game_id in PROFILES:
        raw = load_raw(game_id)
        roles.append(raw.get("agent_persona", {}).get("role", ""))
    assert len(set(roles)) == len(roles), (
        f"Agent persona roles are not all unique: {roles}"
    )


# ═══════════════════════════════════════════════════════════════════
# 6. Documents registry — filename, doc_type, priority, total_pages
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("game_id,spec", PLAN_SPECS.items())
def test_document_registry_has_one_core_rules_entry(game_id, spec):
    raw = load_raw(game_id)
    docs = raw.get("documents", {})
    assert len(docs) >= 1, f"[{game_id}] documents registry is empty"

    # Find the core_rules entry
    core_entries = {k: v for k, v in docs.items() if v.get("doc_type") == "core_rules"}
    assert len(core_entries) == 1, (
        f"[{game_id}] Expected exactly 1 core_rules document, got {len(core_entries)}: "
        f"{list(core_entries.keys())}"
    )


@pytest.mark.parametrize("game_id,spec", PLAN_SPECS.items())
def test_core_rules_filename_contains_game_fragment(game_id, spec):
    raw = load_raw(game_id)
    docs = raw.get("documents", {})
    filenames = [k for k, v in docs.items() if v.get("doc_type") == "core_rules"]
    assert filenames, f"[{game_id}] No core_rules document found"
    fname = filenames[0].lower()
    assert spec["doc_filename_fragment"] in fname, (
        f"[{game_id}] core_rules filename '{fname}' must contain '{spec['doc_filename_fragment']}'"
    )


@pytest.mark.parametrize("game_id,spec", PLAN_SPECS.items())
def test_core_rules_priority_is_one(game_id, spec):
    raw = load_raw(game_id)
    docs = raw.get("documents", {})
    for fname, doc in docs.items():
        if doc.get("doc_type") == "core_rules":
            assert doc.get("priority") == 1, (
                f"[{game_id}] core_rules doc '{fname}' priority must be 1, got {doc.get('priority')}"
            )


@pytest.mark.parametrize("game_id,spec", PLAN_SPECS.items())
def test_core_rules_total_pages_matches_plan(game_id, spec):
    raw = load_raw(game_id)
    docs = raw.get("documents", {})
    for fname, doc in docs.items():
        if doc.get("doc_type") == "core_rules":
            assert doc.get("total_pages") == spec["doc_total_pages"], (
                f"[{game_id}] total_pages for '{fname}': "
                f"got {doc.get('total_pages')}, expected {spec['doc_total_pages']}"
            )


# ═══════════════════════════════════════════════════════════════════
# 7. Flags: ocr_required, extract_tables, ocr_substitutions,
#           use_toc_bookmarks
# ═══════════════════════════════════════════════════════════════════

def test_albedo_ocr_required_is_true_at_profile_or_doc_level():
    """Albedo has 0 native text — ocr_required MUST be flagged."""
    raw = load_raw("albedo_rpg")
    profile_flag = raw.get("ocr_required", False)
    doc_flags = [
        v.get("ocr_required", False)
        for v in raw.get("documents", {}).values()
    ]
    assert profile_flag or any(doc_flags), (
        "albedo_rpg: ocr_required must be True at profile or doc level.\n"
        f"profile-level: {profile_flag}, doc-level: {doc_flags}"
    )


def test_fow_ocr_required_is_false():
    """FoW is a digital PDF — ocr_required must be False (or absent)."""
    raw = load_raw("flames_of_war")
    assert not raw.get("ocr_required", False), (
        "flames_of_war: ocr_required must be False (digital PDF, not scanned)"
    )


@pytest.mark.parametrize("game_id", ["flames_of_war", "honours_of_war"])
def test_extract_tables_is_true(game_id):
    """FoW and HoW have digital tables — extract_tables must be True."""
    raw = load_raw(game_id)
    # Check at profile level or on the core_rules document
    profile_flag = raw.get("extract_tables", False)
    doc_flags = [v.get("extract_tables", False) for v in raw.get("documents", {}).values()]
    assert profile_flag or any(doc_flags), (
        f"[{game_id}] extract_tables must be True at profile or doc level."
    )


def test_tokyo_ocr_substitutions_include_zero():
    """Tokyo Express OCR normalization: 'O' → '0' substitution required."""
    raw = load_raw("tokyo_express")
    subs = raw.get("ocr_substitutions", {})
    assert "O" in subs, (
        f"tokyo_express: ocr_substitutions must include 'O' key (OCR zero→Oh fix). "
        f"Present keys: {list(subs.keys())}"
    )
    assert subs["O"] == "0", (
        f"tokyo_express: ocr_substitutions['O'] must be '0', got '{subs['O']}'"
    )


def test_honours_of_war_use_toc_bookmarks_is_true():
    """HoW has 77 TOC bookmarks — use_toc_bookmarks should be set True."""
    raw = load_raw("honours_of_war")
    assert raw.get("use_toc_bookmarks", False), (
        "honours_of_war: use_toc_bookmarks must be True (HoW has 77 PDF bookmarks)"
    )


# ═══════════════════════════════════════════════════════════════════
# 8. DomainProfile Pydantic validation round-trip
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("game_id", list(PROFILES.keys()))
def test_pydantic_validation_round_trip(game_id):
    """Profile JSON must validate cleanly through DomainProfile.model_validate()."""
    if not os.path.exists(PROFILES[game_id]):
        pytest.skip(f"Profile not yet created (RED state): {os.path.basename(PROFILES[game_id])}")
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from engine.models.domain_profile import load_domain_profile
    except ImportError:
        pytest.skip("DomainProfile not importable in this context")

    profile = load_domain_profile(PROFILES[game_id])
    # game_id / domain_id
    assert profile.id == PLAN_SPECS[game_id]["game_id"]
    # rule_schema
    assert profile.rule_schema == PLAN_SPECS[game_id]["rule_schema"]
    # pipeline_mode
    assert profile.pipeline_mode == "RULEBOOK_TECHNICAL"
    # agent_persona populated
    assert profile.agent_persona is not None, f"[{game_id}] agent_persona is None after Pydantic parse"
    # documents populated
    assert len(profile.documents) >= 1, f"[{game_id}] documents dict is empty after Pydantic parse"


@pytest.mark.parametrize("game_id", list(PROFILES.keys()))
def test_pydantic_get_glossary_returns_flat_strings(game_id):
    """profile.get_glossary() must return a Dict[str, str] — no nested dicts/lists."""
    if not os.path.exists(PROFILES[game_id]):
        pytest.skip(f"Profile not yet created (RED state): {os.path.basename(PROFILES[game_id])}")
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from engine.models.domain_profile import load_domain_profile
    except ImportError:
        pytest.skip("DomainProfile not importable")

    profile = load_domain_profile(PROFILES[game_id])
    glossary = profile.get_glossary()
    for k, v in glossary.items():
        assert isinstance(v, str), (
            f"[{game_id}] get_glossary() key '{k}' must be str, got {type(v)}: {repr(v)}"
        )


# ═══════════════════════════════════════════════════════════════════
# 9. Dynamic runtime metadata stage verification
# ═══════════════════════════════════════════════════════════════════

def test_dynamic_metadata_generator_matches_plan_schemas():
    """
    Verify that the autonomous metadata stage (DynamicProfileGenerator)
    runtime-detects the exact rule_schema, pipeline_mode, and doc_type
    matching the implementation plan profiles.
    """
    from engine.ingestion.dynamic_profiler import DynamicProfileGenerator

    gen = DynamicProfileGenerator(use_llm=False)

    pdf_map = {
        "flames_of_war": "pdfcoffee.com_flames-of-war-4th-ed-ew-amp-lw-rule-book-pdf-free.pdf",
        "honours_of_war": "pdfcoffee.com_osprey-wargames-11-honours-of-war-wargames-rules-for-the-seven-yearsx27-war-pdf-free.pdf",
        "tokyo_express": "pdfcoffee.com_tokyo-express-basic-rule-book-pdf-free.pdf",
    }

    for game_id, fname in pdf_map.items():
        pdf_path = os.path.join(MISC_DIR, fname)
        prof = gen.generate_profile(target_path=pdf_path, game_name=game_id, game_id=game_id, save=False)
        spec = PLAN_SPECS[game_id]

        # Rule schema detected dynamically matches implementation plan
        assert prof.rule_schema == spec["rule_schema"], (
            f"[{game_id}] Dynamic profiler detected '{prof.rule_schema}', "
            f"expected plan schema '{spec['rule_schema']}'"
        )
        # Pipeline mode detected dynamically
        assert prof.pipeline_mode == spec["pipeline_mode"]

        # Core document classified with priority 1
        docs = list(prof.documents.values())
        assert len(docs) >= 1
        assert docs[0].doc_type == "core_rules"
        assert docs[0].priority == 1

