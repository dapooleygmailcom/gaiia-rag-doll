"""
Test Suite: Generic Schema-Driven Visual Ingestion (TDD).

Validates that ingest_visual.py and media_agent.py are completely generic and
driven by the DomainProfile Meta-Contract rather than hardcoded magazine schemas.
"""

import os
import json
import pytest
from engine.models.domain_profile import load_domain_profile, DomainProfile
from engine.ingestion.ingest_visual import (
    synthesize_vector_text,
    build_vlm_prompt_from_profile,
    build_structuring_prompt_from_profile,
    process_visual_pdf,
)
from engine.retrieval.media_agent import MediaAgent, parse_query_intent_with_profile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
VISUAL_PROFILE_PATH = os.path.join(PROJECT_ROOT, "data/visual_media_profile.json")


def test_build_prompts_from_custom_profile():
    """Verify that VLM and structurer prompts are dynamically synthesized from profile."""
    custom_profile = DomainProfile.model_validate({
        "domain_name": "Vintage Car Catalog",
        "domain_id": "vintage_cars",
        "data_dir": "data/cars",
        "chroma_collection": "cars-catalog",
        "pipeline_mode": "VISUAL_MEDIA",
        "documents": {
            "catalog_1965.pdf": {"doc_type": "car_catalog", "priority": 1}
        },
        "structured_extraction": {
            "target_schema": {
                "make": "string",
                "model": "string",
                "engine": {
                    "horsepower": "integer",
                    "displacement": "string"
                },
                "exterior_color": "string"
            },
            "vlm_extraction_hints": [
                "Extract vehicle make and model name from header",
                "Extract engine horsepower and displacement if printed in specs"
            ]
        }
    })

    vlm_prompt = build_vlm_prompt_from_profile(custom_profile)
    assert "Extract vehicle make and model name from header" in vlm_prompt
    assert "horsepower" in vlm_prompt or "specs" in vlm_prompt

    struct_prompt = build_structuring_prompt_from_profile(
        custom_profile,
        vlm_description="A red 1965 Ford Mustang with 289ci V8 engine delivering 225 hp.",
        ocr_text="1965 MUSTANG SPECS"
    )
    assert "Vintage Car Catalog" in struct_prompt or "target schema" in struct_prompt.lower()
    assert "horsepower" in struct_prompt
    assert "displacement" in struct_prompt


def test_generic_synthesize_vector_text():
    """Verify that arbitrary nested dictionary data is serialized into rich embedding text."""
    custom_data = {
        "make": "Ford",
        "model": "Mustang GT",
        "year": 1965,
        "engine": {
            "horsepower": 225,
            "displacement": "289 CID V8"
        },
        "exterior_color": "Rangoon Red",
        "tags": ["muscle car", "classic", "v8", "coupe"]
    }

    text = synthesize_vector_text("Mustang_1965", 1, custom_data)
    assert "Mustang GT" in text
    assert "1965" in text
    assert "Rangoon Red" in text
    assert "289 CID V8" in text
    assert "muscle car" in text


def test_visual_media_profile_ingest_prompt_generation():
    """Verify that visual_media_profile.json generates complete glamour/pictorial prompts."""
    profile = load_domain_profile(VISUAL_PROFILE_PATH)
    assert profile.pipeline_mode == "VISUAL_MEDIA"
    assert profile.structured_extraction is not None

    vlm_prompt = build_vlm_prompt_from_profile(profile)
    assert "nudity level" in vlm_prompt.lower() or "model name" in vlm_prompt.lower()

    struct_prompt = build_structuring_prompt_from_profile(
        profile,
        vlm_description="Model standing on beach in swimwear.",
        ocr_text="Summer Special"
    )
    assert "bodily_dimensions" in struct_prompt
    assert "nudity_level" in struct_prompt
