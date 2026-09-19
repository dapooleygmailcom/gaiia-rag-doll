"""
Test Suite: Unified Master Periodical Index Format (TDD).

Validates that data/visual_catalog_index.json uses a common, unified structure
across ALL magazine types (PB, HB, etc.):
- Shared 'photographers' registry indexing shoots across both genres (e.g. Terry Richardson).
- Shared 'models' registry indexing both glamour models and fashion celebrities.
- 'designers_and_brands' registry indexing fashion houses (Givenchy, Chanel, Dior) and prices.
- 'wardrobe_accessories' registry indexing clothing and accessories (scarf, dress, jacket, lingerie).
- Preserves legacy PB vital stats for glamour models while seamlessly enriching fashion entries.
"""

import os
import json
import pytest
from engine.ingestion.ingest_visual import update_master_index_with_catalog


def test_unified_master_index_structure(tmp_path):
    """Verify that update_master_index_with_catalog populates the unified schema across genres."""
    custom_index_path = os.path.join(tmp_path, "test_unified_index.json")
    
    # Sample HB entries
    hb_entries = [
        {
            "page_number": 0,
            "page_type": "Cover",
            "model_name": "Julianne Moore",
            "is_cover_girl": True,
            "document_id": "Harpers_Bazaar_April_2015_USA",
            "publication_metadata": {
                "publication": "Harper's Bazaar",
                "magazine_title": "Harper's Bazaar USA",
                "issue_date": "April 2015",
                "year": 2015
            },
            "production_and_credits": {
                "photographer": "Camilla Akrans",
                "fashion_editor_stylist": "Elissa Santisi"
            },
            "wardrobe_items": [
                {"designer": "Dior", "item": "top", "price_usd": 3500.0}
            ]
        },
        {
            "page_number": 280,
            "page_type": "Feature_Pictorial",
            "model_name": "Julianne Moore",
            "is_cover_girl": False,
            "document_id": "Harpers_Bazaar_April_2015_USA",
            "production_and_credits": {
                "photographer": "Camilla Akrans",
                "fashion_editor_stylist": "Elissa Santisi"
            },
            "wardrobe_items": [
                {"designer": "Givenchy by Riccardo Tisci", "item": "jacket", "price_usd": 4650.0},
                {"designer": "Saint Laurent by Hedi Slimane", "item": "scarf", "price_usd": 325.0}
            ]
        },
        {
            "page_number": 284,
            "page_type": "Feature_Pictorial",
            "model_name": "Featured Cast",
            "document_id": "Harpers_Bazaar_April_2015_USA",
            "production_and_credits": {
                "photographer": "Terry Richardson"
            },
            "wardrobe_items": [
                {"designer": "Lanvin", "item": "blazer", "price_usd": 3265.0}
            ]
        }
    ]

    # Sample PB entries
    pb_entries = [
        {
            "page_number": 0,
            "page_type": "Cover",
            "model_name": "Elizabeth JoAnne",
            "is_cover_girl": True,
            "document_id": "Playboys_Vixens_2006-08_09",
            "publication_metadata": {
                "publication": "Playboy Special Editions",
                "magazine_title": "Playboy's Vixens",
                "issue_date": "August/September 2006",
                "year": 2006
            },
            "production_and_credits": {
                "photographer": "Gen Mishino"
            },
            "physical_attributes": {
                "measurements": "36DD-22-36",
                "height": "5'11\"",
                "hair_color": "Brown"
            }
        },
        {
            "page_number": 50,
            "page_type": "Feature_Pictorial",
            "model_name": "Carmella DeCesare",
            "is_cover_girl": False,
            "document_id": "Playboys_Vixens_2006-08_09",
            "production_and_credits": {
                "photographer": "Terry Richardson"
            },
            "wardrobe_items": [
                {"designer": "Agent Provocateur", "item": "lingerie", "price_usd": 250.0}
            ],
            "physical_attributes": {
                "measurements": "34D-24-34",
                "height": "5'8\"",
                "hair_color": "Brunette"
            }
        }
    ]

    # 1. Update index with HB issue
    idx_data = update_master_index_with_catalog(
        hb_entries, "Harpers_Bazaar_April_2015_USA", index_path=custom_index_path
    )
    # 2. Update index with PB issue
    idx_data = update_master_index_with_catalog(
        pb_entries, "Playboys_Vixens_2006-08_09", index_path=custom_index_path
    )

    # Assert photographers shared seamlessly (Terry Richardson has shoots in BOTH PB and HB)
    tr_slug = "terry_richardson"
    assert tr_slug in idx_data["photographers"]
    tr_shoots = idx_data["photographers"][tr_slug]["shoots"]
    assert len(tr_shoots) == 2
    assert any("Harpers" in s["document_id"] for s in tr_shoots)
    assert any("Playboys" in s["document_id"] for s in tr_shoots)

    # Assert models contains both Julianne Moore and Elizabeth JoAnne
    assert "julianne_moore" in idx_data["models"]
    assert "elizabeth_joanne" in idx_data["models"]
    assert idx_data["models"]["elizabeth_joanne"]["appearances"][0]["physical_attributes"]["measurements"] == "36DD-22-36"

    # Assert designers_and_brands contains Givenchy and Dior
    assert "designers_and_brands" in idx_data
    assert "givenchy" in idx_data["designers_and_brands"]
    assert idx_data["designers_and_brands"]["givenchy"]["mentions"][0]["price_usd"] == 4650.0

    # Assert wardrobe_accessories contains scarf and lingerie
    assert "wardrobe_accessories" in idx_data
    assert "scarf" in idx_data["wardrobe_accessories"]
    assert any(item["page_number"] == 280 for item in idx_data["wardrobe_accessories"]["scarf"])
    assert "lingerie" in idx_data["wardrobe_accessories"]
