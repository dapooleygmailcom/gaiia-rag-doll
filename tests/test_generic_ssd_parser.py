"""
Test Suite: Generic Schema-Driven SSD / Vehicle Record Sheet Parser (TDD).

Validates that vision_ssd_parser.py is completely generic, building its extraction
prompts and validation rules dynamically from the DomainProfile Meta-Contract.
"""

import os
import json
import pytest
from engine.models.domain_profile import load_domain_profile, DomainProfile
from engine.ingestion.vision_ssd_parser import build_ssd_extraction_prompt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
RL_PROFILE_PATH = os.path.join(PROJECT_ROOT, "data/renegade_legion_profile.json")


def test_build_ssd_prompt_from_custom_profile():
    """Verify that SSD extraction prompts are dynamically generated from domain profile."""
    sample_ontology = {
        "entity_type": "starship",
        "name": "string",
        "grids": {
            "Shields": {"SF": "integer", "Width": "integer", "Depth": "integer"}
        },
        "weapons": [{"name": "string", "damage": "integer", "range": "integer"}]
    }

    hints = [
        "Pay special attention to Photon Torpedo tube counts",
        "Do not confuse Phaser-1 with Phaser-2"
    ]

    prompt = build_ssd_extraction_prompt(sample_ontology, hints=hints, domain_name="Star Fleet Battles")
    assert "Star Fleet Battles" in prompt
    assert "Photon Torpedo tube counts" in prompt
    assert "Phaser-1 with Phaser-2" in prompt
    assert "Shields" in prompt


def test_renegade_legion_profile_ssd_prompt_generation():
    """Verify that renegade_legion_profile.json drives prompt construction without hardcoding."""
    profile = load_domain_profile(RL_PROFILE_PATH)
    assert profile.name == "Renegade Legion"
    
    ontology_path = os.path.join(PROJECT_ROOT, "data/renegade_legion_ontology.json")
    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    hints = profile.structured_extraction.vlm_extraction_hints if profile.structured_extraction else []
    prompt = build_ssd_extraction_prompt(ontology, hints=hints, domain_name=profile.name)

    assert "Renegade Legion" in prompt
    assert "Armor" in prompt or "armor" in prompt
