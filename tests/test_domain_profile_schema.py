"""
Test Suite: Domain Profile / Meta-Contract Validation (TDD).

Validates that all profile configurations comply with the Pydantic DomainProfile
Meta-Contract schema and support custom extraction guidelines, ontologies, and agent personas.
"""

import os
import glob
import json
import pytest
from pydantic import ValidationError

from engine.models.domain_profile import (
    DomainProfile,
    DocumentMetadata,
    ParsingGrammar,
    StructuredExtractionConfig,
    OntologyConfig,
    AgentPersonaConfig,
    load_domain_profile,
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))


def test_load_all_existing_profiles_valid():
    """Verify that all JSON profile files in data/ load and validate against DomainProfile."""
    profile_files = glob.glob(os.path.join(DATA_DIR, "*_profile.json"))
    assert len(profile_files) >= 4, f"Expected at least 4 profiles in {DATA_DIR}, found {len(profile_files)}"
    
    for ppath in profile_files:
        profile = load_domain_profile(ppath)
        assert isinstance(profile, DomainProfile)
        assert profile.game_name or profile.domain_name
        assert profile.game_id or profile.domain_id
        assert isinstance(profile.documents, dict)


def test_custom_meta_contract_structured_extraction():
    """Verify structured extraction metadata parsing in DomainProfile."""
    sample_dict = {
        "domain_name": "Visual Catalog Archive",
        "domain_id": "visual_catalog",
        "data_dir": "data/PB",
        "chroma_collection": "rag-doll-visual-catalog",
        "pipeline_mode": "VISUAL_MEDIA",
        "documents": {
            "catalog_2006.pdf": {
                "doc_type": "pictorial",
                "priority": 1,
                "description": "2006 catalog scan"
            }
        },
        "structured_extraction": {
            "target_schema": {
                "page_type": "string",
                "model_name": "string",
                "year": "integer",
                "physical_attributes": {
                    "hair_color": "string",
                    "bodily_dimensions": "string"
                }
            },
            "vlm_extraction_hints": [
                "Extract model names from captions or headlines",
                "Extract hair color and styling"
            ]
        },
        "agent_persona": {
            "role": "Visual Media Archivist",
            "citation_format": "[Page {page_num}, {model_name}]"
        }
    }
    
    profile = DomainProfile.model_validate(sample_dict)
    assert profile.pipeline_mode == "VISUAL_MEDIA"
    assert profile.structured_extraction is not None
    assert "model_name" in profile.structured_extraction.target_schema
    assert len(profile.structured_extraction.vlm_extraction_hints) == 2
    assert profile.agent_persona.role == "Visual Media Archivist"


def test_domain_profile_validation_rejects_missing_required_fields():
    """Verify that empty or malformed profile dicts raise ValidationError."""
    with pytest.raises(ValidationError):
        # Missing required name and id
        DomainProfile.model_validate({"documents": {}})


def test_domain_profile_glossary_expansion():
    """Verify that acronym glossary is accessible and cleanly typed."""
    rl_path = os.path.join(DATA_DIR, "renegade_legion_profile.json")
    if os.path.exists(rl_path):
        profile = load_domain_profile(rl_path)
        glossary = profile.get_glossary()
        assert isinstance(glossary, dict)
        assert "DAP" in glossary
        assert glossary["DAP"] == "DOGFIGHT ARMOUR PIERCING MISSILE"
