"""
Test Suite: Generic Policy & Legal Ingestion (TDD).

Validates that DomainProfile and policy_agent.py are completely generic,
deriving carrier names, document metadata, and comparison prompts from the DomainProfile.
"""

import os
import json
import pytest
from engine.models.domain_profile import load_domain_profile, DomainProfile, get_carrier_for_doc
from engine.retrieval.policy_agent import build_policy_comparison_prompt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
INSURANCE_PROFILE_PATH = os.path.join(PROJECT_ROOT, "data/home_insurance_profile.json")


def test_carrier_resolution_from_profile():
    """Verify that carrier/publisher names are resolved dynamically from DomainProfile."""
    profile = load_domain_profile(INSURANCE_PROFILE_PATH)
    
    carrier = get_carrier_for_doc("aami-home-contents-insurance-pds.pdf", profile)
    assert carrier == "AAMI"
    
    carrier_cba = get_carrier_for_doc("cba-home-pds.pdf", profile)
    assert carrier_cba == "CBA"

    carrier_tio = get_carrier_for_doc("POL1388TIO.pdf", profile)
    assert carrier_tio == "TIO"


def test_build_policy_comparison_prompt_from_profile():
    """Verify policy comparison prompt synthesis from Meta-Contract persona."""
    profile = load_domain_profile(INSURANCE_PROFILE_PATH)
    prompt = build_policy_comparison_prompt(
        profile,
        query="Does AAMI or CBA cover accidental flood damage?",
        context="Sample policy extracts for AAMI and CBA"
    )

    assert "Home & Contents" in prompt or "Insurance" in prompt
    assert "Does AAMI or CBA cover accidental flood damage?" in prompt
    assert "Sample policy extracts for AAMI and CBA" in prompt
