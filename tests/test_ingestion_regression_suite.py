"""
Comprehensive Automated Regression Suite for Ingestion & Schema Classification.
Validates zero regressions on Up Front, Cadet Handbook, and Honours of War,
and verifies granular codified rule extraction for Tokyo Express.
"""

import os
import sys
import json
import pytest

# Ensure backend/engine is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.ingestion.auto_discover import (
    sample_pdf_text,
    detect_rule_schema,
    KNOWN_SCHEMAS,
)
from engine.ingestion.dynamic_profiler import DynamicProfileGenerator
from engine.ingestion.ingest_rules import (
    load_profile,
    get_compiled_patterns,
    route_chunk_generic,
    build_rule_index,
    build_section_tree,
    build_ingestion_cooccurrence_graph,
    ingest_game,
)

UP_FRONT_PDF = r"c:\programming\aiia\gaiia-rag-doll\data\upfront\UF RuleBook updated.pdf"
CADET_PDF = r"c:\programming\aiia\gaiia-rag-doll\data\sfb\CadetHandbook.pdf"
HONOURS_PDF = r"c:\programming\aiia\gaiia-rag-doll\data\misc\pdfcoffee.com_osprey-wargames-11-honours-of-war-wargames-rules-for-the-seven-yearsx27-war-pdf-free.pdf"
TOKYO_PDF = r"c:\programming\aiia\gaiia-rag-doll\data\misc\pdfcoffee.com_tokyo-express-basic-rule-book-pdf-free.pdf"
FLAMES_PDF = r"c:\programming\aiia\gaiia-rag-doll\data\misc\pdfcoffee.com_flames-of-war-4th-ed-ew-amp-lw-rule-book-pdf-free.pdf"
ALBEDO_PDF = r"c:\programming\aiia\gaiia-rag-doll\data\misc\pdfcoffee.com_albedo-2nd-edition-compressed-pdf-free.pdf"


# ═══════════════════════════════════════════════════════════════════
# Layer 1: Schema Classification Invariants
# ═══════════════════════════════════════════════════════════════════

def test_up_front_schema_density_classification():
    """Up Front must always classify as numeric_decimal due to high non-TOC rule density."""
    assert os.path.exists(UP_FRONT_PDF), f"File missing: {UP_FRONT_PDF}"
    samples = sample_pdf_text(UP_FRONT_PDF)
    best_schema, scores = detect_rule_schema([t for _, t in samples])
    
    assert best_schema["name"] == "numeric_decimal", f"Expected numeric_decimal, got {best_schema['name']}"
    assert scores["numeric_decimal"] >= 20, f"Expected high numeric_decimal score, got {scores}"


def test_cadet_handbook_schema_classification():
    """Cadet Handbook must always classify as outline_parenthetical."""
    assert os.path.exists(CADET_PDF), f"File missing: {CADET_PDF}"
    samples = sample_pdf_text(CADET_PDF)
    best_schema, scores = detect_rule_schema([t for _, t in samples])
    
    assert best_schema["name"] == "outline_parenthetical", f"Expected outline_parenthetical, got {best_schema['name']}"
    assert scores["outline_parenthetical"] >= 5, f"Expected outline_parenthetical >= 5, got {scores}"


def test_honours_of_war_schema_classification():
    """Honours of War must always classify as keyword_header."""
    assert os.path.exists(HONOURS_PDF), f"File missing: {HONOURS_PDF}"
    samples = sample_pdf_text(HONOURS_PDF)
    best_schema, scores = detect_rule_schema([t for _, t in samples])
    
    assert best_schema["name"] == "keyword_header", f"Expected keyword_header, got {best_schema['name']}"
    assert scores["numeric_decimal"] == 0, f"Expected 0 numeric_decimal score, got {scores}"


def test_tokyo_express_schema_classification():
    """Tokyo Express has sparse decimal headers with dominant keyword headers -> keyword_header."""
    assert os.path.exists(TOKYO_PDF), f"File missing: {TOKYO_PDF}"
    samples = sample_pdf_text(TOKYO_PDF)
    best_schema, scores = detect_rule_schema([t for _, t in samples])
    
    assert best_schema["name"] == "keyword_header", f"Expected keyword_header, got {best_schema['name']}"
    assert scores["keyword_header"] > scores["numeric_decimal"], f"Expected keyword_header > numeric_decimal, got {scores}"


def test_flames_of_war_schema_classification():
    """Flames of War must classify as keyword_header."""
    assert os.path.exists(FLAMES_PDF), f"File missing: {FLAMES_PDF}"
    samples = sample_pdf_text(FLAMES_PDF)
    best_schema, scores = detect_rule_schema([t for _, t in samples])
    
    assert best_schema["name"] == "keyword_header", f"Expected keyword_header, got {best_schema['name']}"
    assert scores["keyword_header"] >= 5, f"Expected keyword_header >= 5, got {scores}"


def test_albedo_scanned_ocr_schema_classification():
    """Albedo 2nd Edition (100% scanned bitmap) must trigger OCR fallback and classify as keyword_header."""
    assert os.path.exists(ALBEDO_PDF), f"File missing: {ALBEDO_PDF}"
    samples = sample_pdf_text(ALBEDO_PDF)
    # Ensure text was extracted via OCR fallback
    total_chars = sum(len(t) for _, t in samples)
    assert total_chars >= 500, f"Expected OCR sampled characters >= 500, got {total_chars}"
    
    best_schema, scores = detect_rule_schema([t for _, t in samples])
    assert best_schema["name"] == "keyword_header", f"Expected keyword_header, got {best_schema['name']}"
    assert scores["keyword_header"] >= 5, f"Expected keyword_header >= 5, got {scores}"


# ═══════════════════════════════════════════════════════════════════
# Layer 2: Chunker & Ingestion Invariants
# ═══════════════════════════════════════════════════════════════════

def test_up_front_baseline_chunking_and_rules():
    """Up Front core rulebook must produce >= 800 chunks and granular decimal rules."""
    gen = DynamicProfileGenerator(use_llm=False)
    prof = gen.generate_profile(UP_FRONT_PDF, game_id="up_front", save=False)
    if hasattr(prof, "model_dump"):
        prof = prof.model_dump()
        
    assert prof["rule_schema"] == "numeric_decimal"
    
    import fitz
    doc = fitz.open(UP_FRONT_PDF)
    full_text = "\n".join([f"--- PAGE {i+1} ---\n" + doc[i].get_text() for i in range(len(doc))])
    doc.close()
    
    patterns = get_compiled_patterns(prof)
    chunks = route_chunk_generic(full_text, os.path.basename(UP_FRONT_PDF), prof["documents"][os.path.basename(UP_FRONT_PDF)], prof, patterns)
    
    assert len(chunks) >= 800, f"Expected >= 800 chunks for Up Front, got {len(chunks)}"
    
    rule_nums = set(c.get("rule_number") for c in chunks if c.get("rule_number"))
    assert "2.1" in rule_nums, "Expected Rule 2.1 in Up Front"
    assert "5.41" in rule_nums, "Expected Rule 5.41 in Up Front"
    assert "17.4" in rule_nums, "Expected Rule 17.4 in Up Front"
    assert len(rule_nums) >= 250, f"Expected >= 250 granular rules, got {len(rule_nums)}"


def test_cadet_handbook_baseline_chunking():
    """Cadet Handbook must produce >= 400 chunks and outline parenthetical rules."""
    gen = DynamicProfileGenerator(use_llm=False)
    prof = gen.generate_profile(CADET_PDF, game_id="star_fleet_battles", save=False)
    if hasattr(prof, "model_dump"):
        prof = prof.model_dump()
        
    assert prof["rule_schema"] == "outline_parenthetical"
    
    import fitz
    doc = fitz.open(CADET_PDF)
    full_text = "\n".join([f"--- PAGE {i+1} ---\n" + doc[i].get_text() for i in range(len(doc))])
    doc.close()
    
    patterns = get_compiled_patterns(prof)
    chunks = route_chunk_generic(full_text, os.path.basename(CADET_PDF), prof["documents"][os.path.basename(CADET_PDF)], prof, patterns)
    
    assert len(chunks) >= 400, f"Expected >= 400 chunks for Cadet Handbook, got {len(chunks)}"
    rule_nums = set(c.get("rule_number") for c in chunks if c.get("rule_number"))
    assert any("C1." in str(r) or "D2." in str(r) or "E2." in str(r) for r in rule_nums), f"Expected SFB rules, got {list(rule_nums)[:10]}"


def test_honours_of_war_baseline_chunking():
    """Honours of War must produce >= 150 chunks with valid visual headers."""
    gen = DynamicProfileGenerator(use_llm=False)
    prof = gen.generate_profile(HONOURS_PDF, game_id="honours_of_war", save=False)
    if hasattr(prof, "model_dump"):
        prof = prof.model_dump()
        
    assert prof["rule_schema"] == "keyword_header"
    
    import fitz
    doc = fitz.open(HONOURS_PDF)
    full_text = "\n".join([f"--- PAGE {i+1} ---\n" + doc[i].get_text() for i in range(len(doc))])
    doc.close()
    
    patterns = get_compiled_patterns(prof)
    chunks = route_chunk_generic(full_text, os.path.basename(HONOURS_PDF), prof["documents"][os.path.basename(HONOURS_PDF)], prof, patterns)
    
    assert len(chunks) >= 150, f"Expected >= 150 chunks for Honours of War, got {len(chunks)}"
    headers = set(c.get("rule_number") for c in chunks if c.get("rule_number"))
    assert "BRIGADE MORALE" in headers or any("MORALE" in str(h).upper() for h in headers), f"Expected Morale headers, got {list(headers)[:10]}"


def test_tokyo_express_enhanced_granular_chunking():
    """Tokyo Express must produce >= 40 distinct codified rules and populated SectionTree."""
    gen = DynamicProfileGenerator(use_llm=False)
    prof = gen.generate_profile(TOKYO_PDF, game_id="tokyo-express", save=False)
    if hasattr(prof, "model_dump"):
        prof = prof.model_dump()
        
    assert prof["rule_schema"] == "keyword_header"
    
    import fitz
    doc = fitz.open(TOKYO_PDF)
    full_text = "\n".join([f"--- PAGE {i+1} ---\n" + doc[i].get_text() for i in range(len(doc))])
    doc.close()
    
    patterns = get_compiled_patterns(prof)
    chunks = route_chunk_generic(full_text, os.path.basename(TOKYO_PDF), prof["documents"][os.path.basename(TOKYO_PDF)], prof, patterns)
    
    all_chunks_dict = {f"chunk_{i+1}": c for i, c in enumerate(chunks)}
    rule_index = build_rule_index(all_chunks_dict)
    
    non_meta_rules = [k for k in rule_index.keys() if not k.startswith("__")]
    assert len(non_meta_rules) >= 40, f"Expected >= 40 granular rules for Tokyo Express, got {len(non_meta_rules)}"
    assert any("3.1" in r for r in non_meta_rules), "Expected Section 3.1 rules"
    assert any("3.3" in r for r in non_meta_rules), "Expected Section 3.3 rules"
    
    sec_tree = build_section_tree(all_chunks_dict, game_id="tokyo-express")
    assert len(sec_tree.sections) >= 5, f"Expected >= 5 section tree branches, got {len(sec_tree.sections)}"


def test_tokyo_express_declarative_ocr_substitutions():
    """Verify profile-defined declarative OCR regex substitutions correctly fix scanned errors (9.2 -> 3.2)."""
    prof_path = os.path.join(os.path.dirname(__file__), "..", "data", "misc", "tokyo_express_profile.json")
    assert os.path.exists(prof_path)
    with open(prof_path, "r", encoding="utf-8") as f:
        prof = json.load(f)

    import fitz
    doc = fitz.open(TOKYO_PDF)
    full_text = "\n".join([f"--- PAGE {i+1} ---\n" + doc[i].get_text() for i in range(len(doc))])
    doc.close()

    patterns = get_compiled_patterns(prof)
    doc_info = prof["documents"][os.path.basename(TOKYO_PDF)]
    chunks = route_chunk_generic(full_text, os.path.basename(TOKYO_PDF), doc_info, prof, patterns)
    all_chunks_dict = {f"chunk_{i+1}": c for i, c in enumerate(chunks)}
    rule_index = build_rule_index(all_chunks_dict)

    non_meta_rules = [k for k in rule_index.keys() if not k.startswith("__")]
    assert any("3.2" in r for r in non_meta_rules), "Expected Section 3.2 resolved via profile declarative substitutions"


def test_flames_of_war_chunking():
    """Flames of War must produce >= 100 chunks and keyword headers."""
    gen = DynamicProfileGenerator(use_llm=False)
    prof = gen.generate_profile(FLAMES_PDF, game_id="flames-of-war", save=False)
    if hasattr(prof, "model_dump"):
        prof = prof.model_dump()
        
    assert prof["rule_schema"] == "keyword_header"
    
    import fitz
    doc = fitz.open(FLAMES_PDF)
    full_text = "\n".join([f"--- PAGE {i+1} ---\n" + doc[i].get_text() for i in range(len(doc))])
    doc.close()
    
    patterns = get_compiled_patterns(prof)
    chunks = route_chunk_generic(full_text, os.path.basename(FLAMES_PDF), prof["documents"][os.path.basename(FLAMES_PDF)], prof, patterns)
    
    assert len(chunks) >= 100, f"Expected >= 100 chunks for Flames of War, got {len(chunks)}"


def test_albedo_rpg_ocr_sample_chunking():
    """Albedo RPG sample (first 15 pages) through RapidOCR pipeline produces valid keyword_header chunks."""
    from engine.ingestion.ocr_processor import init_ocr, process_pdf_to_text
    
    ocr = init_ocr()
    full_text = process_pdf_to_text(ALBEDO_PDF, ocr, max_pages=15)
    assert len(full_text) >= 5000, f"Expected >= 5000 chars from 15 OCR pages, got {len(full_text)}"
    
    gen = DynamicProfileGenerator(use_llm=False)
    prof = gen.generate_profile(ALBEDO_PDF, game_id="albedo", save=False)
    if hasattr(prof, "model_dump"):
        prof = prof.model_dump()
        
    patterns = get_compiled_patterns(prof)
    chunks = route_chunk_generic(full_text, os.path.basename(ALBEDO_PDF), prof["documents"][os.path.basename(ALBEDO_PDF)], prof, patterns)
    
    assert len(chunks) >= 10, f"Expected >= 10 chunks from 15 OCR pages, got {len(chunks)}"
    rule_headers = [c.get("rule_number") for c in chunks if c.get("rule_number")]
    assert len(rule_headers) >= 3, f"Expected >= 3 rule headers, got {rule_headers}"


def test_dynamic_hierarchy_extraction_across_all_schemas():
    """Verify _extract_hierarchy_from_rule dynamically dispatches via profile parsing grammar without hardcoding."""
    from engine.ingestion.ingest_rules import _extract_hierarchy_from_rule
    from engine.models.domain_profile import DomainProfile, ParsingGrammar

    # 1. Breadcrumb / Keyword Header schema (e.g. Albedo, FoW, HoW)
    kh_prof = DomainProfile(
        game_name="Albedo RPG",
        game_id="albedo_rpg",
        parsing_grammar=ParsingGrammar(
            rule_schema="keyword_header",
            hierarchy_strategy="breadcrumb_path",
            hierarchy_delimiter=" > "
        )
    )
    root, parent, level = _extract_hierarchy_from_rule("COMBAT > Missile Combat > Penetration Resistance", profile=kh_prof)
    assert root == "COMBAT"
    assert parent == "Missile Combat"
    assert level == 3

    root, parent, level = _extract_hierarchy_from_rule("COMBAT > Armour Penetration", profile=kh_prof)
    assert root == "COMBAT"
    assert parent == "COMBAT"
    assert level == 2

    root, parent, level = _extract_hierarchy_from_rule("COMBAT", profile=kh_prof)
    assert root == "COMBAT"
    assert parent == ""
    assert level == 1

    # 2. Numeric Decimal schema (e.g. Up Front)
    nd_prof = DomainProfile(
        game_name="Up Front",
        game_id="up_front",
        parsing_grammar=ParsingGrammar(
            rule_schema="numeric_decimal",
            hierarchy_strategy="numeric_decimal",
            hierarchy_delimiter="."
        )
    )
    root, parent, level = _extract_hierarchy_from_rule("5.41", profile=nd_prof)
    assert root == "5.0"
    assert parent == "5.4"
    assert level == 3

    root, parent, level = _extract_hierarchy_from_rule("5.4", profile=nd_prof)
    assert root == "5.0"
    assert parent == "5.0"
    assert level == 2

    root, parent, level = _extract_hierarchy_from_rule("5.0", profile=nd_prof)
    assert root == "5.0"
    assert parent == ""
    assert level == 1

    # 3. Chapter Decimal schema (e.g. ASL)
    cd_prof = DomainProfile(
        game_name="Advanced Squad Leader",
        game_id="asl",
        parsing_grammar=ParsingGrammar(
            rule_schema="chapter_decimal",
            hierarchy_strategy="chapter_decimal",
            hierarchy_delimiter="."
        )
    )
    root, parent, level = _extract_hierarchy_from_rule("A7.21", profile=cd_prof)
    assert root == "A7.0"
    assert parent == "A7.2"
    assert level == 3

    # 4. Outline Parenthetical schema (e.g. SFB Cadet)
    op_prof = DomainProfile(
        game_name="Star Fleet Battles Cadet",
        game_id="sfb",
        parsing_grammar=ParsingGrammar(
            rule_schema="outline_parenthetical",
            hierarchy_strategy="outline_parenthetical",
            hierarchy_delimiter="."
        )
    )
    root, parent, level = _extract_hierarchy_from_rule("(D2.31)", profile=op_prof)
    assert root == "(D2.0)"
    assert parent == "(D2.3)"
    assert level == 3

    # 5. User-defined custom regex strategy
    custom_prof = DomainProfile(
        game_name="Custom Game",
        game_id="custom_game",
        parsing_grammar=ParsingGrammar(
            hierarchy_regex=r'^(?P<root>SEC-[A-Z])-(?P<parent>\d+)\.(?P<rule>\d+)$'
        )
    )
    root, parent, level = _extract_hierarchy_from_rule("SEC-A-10.3", profile=custom_prof)
    assert root == "SEC-A"
    assert parent == "10"
    assert level == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

