"""
Test Suite: Dynamic Profile Generation (TDD).

Validates autonomous profile generation against:
1. Up Front corpus (numeric decimal wargame standard)
2. Star Fleet Battles Cadet Handbook (single-file parenthetical outline standard)
3. SFB directory corpus (multi-document wargame with tables and supplements)
"""

import os
import re
import pytest

from engine.ingestion.dynamic_profiler import DynamicProfileGenerator
from engine.models.domain_profile import DomainProfile

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
UPFRONT_DIR = os.path.join(DATA_DIR, "upfront")
SFB_DIR = os.path.join(DATA_DIR, "sfb")
CADET_PDF = os.path.join(SFB_DIR, "CadetHandbook.pdf")


def test_dynamic_profile_upfront():
    """Verify autonomous profile generation on the Up Front corpus."""
    generator = DynamicProfileGenerator(use_llm=False)
    profile = generator.generate_profile(
        target_path=UPFRONT_DIR,
        game_name="Up Front",
        game_id="up_front_test",
        save=False,
    )

    assert isinstance(profile, DomainProfile)
    assert profile.domain_name == "Up Front"
    assert profile.rule_schema == "numeric_decimal"
    assert profile.scenario_format == "letter"

    # Verify rule pattern matches decimal rule numbers
    rule_pat = re.compile(profile.rule_pattern, re.MULTILINE)
    assert rule_pat.search("\n14.2 WEAPON JAMS\n") is not None
    assert rule_pat.search("\n5.41\n") is not None

    # Verify document classification
    docs = profile.documents
    assert len(docs) >= 8

    # Core rulebook classification & priority
    assert "UF RuleBook updated.pdf" in docs
    assert docs["UF RuleBook updated.pdf"].doc_type == "core_rules"
    assert docs["UF RuleBook updated.pdf"].priority == 1

    # Errata classification & priority
    assert "Up_Front_Errata_Pages.pdf" in docs
    assert docs["Up_Front_Errata_Pages.pdf"].doc_type == "errata"
    assert docs["Up_Front_Errata_Pages.pdf"].priority == 3

    # Scenarios classification
    assert "UF_Scenarios_2-4.pdf" in docs
    assert docs["UF_Scenarios_2-4.pdf"].doc_type == "scenarios"


def test_dynamic_profile_cadet_handbook_single_file():
    """Verify autonomous profile generation on a single PDF: CadetHandbook.pdf."""
    assert os.path.exists(CADET_PDF), f"Test PDF missing: {CADET_PDF}"

    generator = DynamicProfileGenerator(use_llm=False)
    profile = generator.generate_profile(
        target_path=CADET_PDF,
        game_name="Star Fleet Battles Cadet",
        game_id="sfb_cadet_test",
        save=False,
    )

    assert isinstance(profile, DomainProfile)
    assert profile.rule_schema == "outline_parenthetical"
    assert profile.scenario_format == "numeric"

    # Verify rule pattern matches standard and deep SFB parentheticals
    rule_pat = re.compile(profile.rule_pattern)
    assert rule_pat.match("(B2.0)") is not None
    assert rule_pat.match("(C3.1)") is not None
    assert rule_pat.match("(D6.13)") is not None
    # 4-digit sub-rule match
    assert rule_pat.match("(D3.3411)") is not None

    # Verify document classification for single PDF
    docs = profile.documents
    assert "CadetHandbook.pdf" in docs
    assert docs["CadetHandbook.pdf"].doc_type in ("core_rules", "supplement")
    assert docs["CadetHandbook.pdf"].priority == 1


def test_dynamic_profile_sfb_directory():
    """Verify autonomous profile generation on the full SFB directory."""
    generator = DynamicProfileGenerator(use_llm=False)
    profile = generator.generate_profile(
        target_path=SFB_DIR,
        game_name="Star Fleet Battles",
        game_id="sfb_test",
        save=False,
    )

    assert isinstance(profile, DomainProfile)
    assert profile.rule_schema == "outline_parenthetical"
    assert profile.scenario_format == "numeric"

    docs = profile.documents
    assert len(docs) >= 4

    # Cadet handbook priority 1
    assert "CadetHandbook.pdf" in docs
    assert docs["CadetHandbook.pdf"].priority == 1

    # Captain's Edition basic set rulebook
    rulebook_key = "pdfcoffee.com_001-5501-star-fleet-battles-captainx27s-edition-basic-set-rulebook-2012-pdf-free.pdf"
    if rulebook_key in docs:
        assert docs[rulebook_key].doc_type == "core_rules"
        assert docs[rulebook_key].priority == 1

    # SSD table book
    ssd_key = "pdfcoffee.com_c4a-tfg-5616-star-fleet-battles-ssd-book-pdf-free.pdf"
    if ssd_key in docs:
        assert docs[ssd_key].doc_type == "tables"
        assert docs[ssd_key].priority == 4


def test_pydantic_roundtrip():
    """Verify that generated profile dict round-trips through DomainProfile."""
    generator = DynamicProfileGenerator(use_llm=False)
    profile = generator.generate_profile(
        target_path=CADET_PDF,
        save=False,
    )

    dumped = profile.model_dump()
    reloaded = DomainProfile.model_validate(dumped)
    assert reloaded.domain_id == profile.domain_id
    assert reloaded.rule_schema == profile.rule_schema
