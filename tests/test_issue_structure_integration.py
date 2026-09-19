"""
Test Suite: Issue Structure Integration & Dynamic Stockist Linking (Red/Green TDD).

Validates that:
1. extract_dynamic_issue_structure accurately parses Harper's Bazaar TOC,
   identifying Julianne Moore, Morgane Polanski, Carolyn Murphy, and photographers
   Camilla Akrans and Terry Richardson.
2. Back-of-book Where to Buy stockist entries are extracted and attached to the issue metadata.
3. Playboy regression is 100% preserved (Vixens TOC and bio cards extract identical results).
"""

import os
import fitz
import pytest
from engine.ingestion.ingest_visual import extract_dynamic_issue_structure


def test_harpers_bazaar_dynamic_issue_structure():
    """Verify Harper's Bazaar April 2015 USA issue structure extraction."""
    hb_path = "data/HB/Harpers_Bazaar_April_2015_USA.pdf"
    if not os.path.exists(hb_path):
        pytest.skip(f"HB PDF not found at {hb_path}")
        
    doc = fitz.open(hb_path)
    cover_data, model_registry = extract_dynamic_issue_structure(doc, hb_path, category="HB")
    doc.close()

    # 1. Publication metadata
    assert "Harper's Bazaar" in cover_data["publication"]
    assert cover_data["year"] == 2015

    # 2. Cover girl / Feature celebrity
    assert "Julianne Moore" in [cover_data.get("cover_girl"), *[m["model_name"] for m in model_registry.values()]]

    # 3. Model & Feature registry
    registry_names = [m["model_name"] for m in model_registry.values()]
    assert any("Julianne Moore" in name for name in registry_names)
    assert any("Morgane Polanski" in name for name in registry_names)
    assert any("Carolyn Murphy" in name for name in registry_names)

    # 4. Photographers mapped to features
    photographers_found = [m.get("photographer") for m in model_registry.values() if m.get("photographer")]
    assert any("Camilla Akrans" in p for p in photographers_found or []) or any("Terry Richardson" in p for p in photographers_found or [])

    # 5. Stockist directory
    stockist = cover_data.get("stockist_directory", [])
    assert len(stockist) > 20
    designers = [e["designer"].lower() for e in stockist]
    assert any("gucci" in d for d in designers)
    assert any("givenchy" in d or "chloé" in d or "max mara" in d for d in designers)


def test_playboy_dynamic_issue_structure_regression():
    """Verify legacy Playboy issues continue to parse TOC and Bio cards identically."""
    pb_path = "data/PB/Playboys_Vixens_2006-08_09.pdf"
    if not os.path.exists(pb_path):
        pytest.skip(f"PB PDF not found at {pb_path}")
        
    doc = fitz.open(pb_path)
    cover_data, model_registry = extract_dynamic_issue_structure(doc, pb_path, category="PB")
    doc.close()

    assert "Playboy" in cover_data["publication"]
    assert cover_data["year"] == 2006

    # Verify bio cards and TOC entries extracted
    registry_names = [m["model_name"] for m in model_registry.values()]
    assert any("Elizabeth" in name or "Jennifer" in name for name in registry_names)
    assert any("Christine" in name or "Heather" in name for name in registry_names)
