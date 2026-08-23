"""
Test Suite: Universal Profiler, Visual Media Ingestion, and Media Agent.
"""

import os
import sys
import pytest
import fitz

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)

from engine.ingestion.universal_ingest import DocumentProfiler
from engine.ingestion.ingest_visual import extract_page_image, synthesize_vector_text, analyze_page_with_vlm
from engine.retrieval.media_agent import parse_query_intent, MediaAgent


def test_document_profiler_visual_detection():
    """Verify that Playboys_Vixens is autonomously profiled as VISUAL_MEDIA."""
    pdf_path = os.path.join(PROJECT_ROOT, "data/PB/Playboys_Vixens_2006-08_09.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test PDF not found at {pdf_path}")
        
    profile = DocumentProfiler.profile_pdf(pdf_path)
    assert profile["total_pages"] > 0
    assert profile["avg_visual_coverage"] > 0.3 or profile["avg_images_per_page"] > 0.5
    assert profile["detected_pipeline"] == "VISUAL_MEDIA"


def test_page_rasterization():
    """Verify that PyMuPDF renders high-quality page images."""
    pdf_path = os.path.join(PROJECT_ROOT, "data/PB/Playboys_Vixens_2006-08_09.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test PDF not found at {pdf_path}")
        
    doc = fitz.open(pdf_path)
    page = doc[0]
    pil_img, img_bytes, pix = extract_page_image(page, dpi=150)
    doc.close()
    
    assert pil_img.width > 500
    assert pil_img.height > 500
    assert len(img_bytes) > 10000


def test_synthesize_vector_text():
    """Verify that multi-attribute metadata is correctly woven into vector text."""
    sample_data = {
        "page_type": "Pictorial",
        "model_name": "Carmella DeCesare",
        "year": 2006,
        "physical_attributes": {
            "hair_color": "Brunette",
            "bodily_dimensions": "34D-24-34",
            "natural_status": "Enhanced",
            "was_playmate": True,
            "playmate_details": "Playmate of the Year 2004"
        },
        "presentation_and_styling": {
            "nudity_level": "Topless",
            "grooming": "Shaved",
            "wardrobe": "Silk sarong"
        },
        "visual_setting_and_theme": {
            "primary_theme": "Beach",
            "setting_description": "Tropical outdoor sunset on white sand",
            "tags": ["beach", "sunset", "glamour", "summer"]
        },
        "visual_narrative": "A sunlit beach photoshoot with ocean waves.",
        "ocr_text": "Summer Heat Spread"
    }
    
    text = synthesize_vector_text("PB_2006", 14, sample_data)
    assert "Carmella DeCesare" in text
    assert "Brunette" in text
    assert "34D-24-34" in text
    assert "Topless" in text
    assert "Shaved" in text
    assert "Beach" in text
    assert "Playmate" in text


def test_query_parser_heuristics():
    """Verify that the query parser extracts structured search parameters."""
    query = "Show me blonde beach pictorial spreads from 2006"
    parsed = parse_query_intent(query)
    
    filters = parsed.get("filters", {})
    assert filters.get("year") == 2006
    assert filters.get("hair_color") == "Blonde"
    assert filters.get("primary_theme") == "Beach"
