"""
Test Suite: Unified Media Agent Query Parser & Multi-Periodical Retrieval (TDD).

Validates that:
1. parse_query_intent extracts wardrobe accessories (e.g. scarf), designers (Givenchy),
   photographers across genres (Terry Richardson), and models/celebrities.
2. MediaAgent.search queries the unified master index for photographers, designers,
   wardrobe accessories, and models across publications.
3. format_retrieval_response suppresses empty vital stats placeholders for fashion
   publications and renders rich wardrobe/designer/price details.
4. Legacy PB searches (Carmella DeCesare, Arny Freytag) continue to work seamlessly.
"""

import os
import json
import pytest
from engine.retrieval.media_agent import (
    parse_query_intent,
    extract_rule_based_filters,
    MediaAgent,
)


def test_fashion_query_intent_parsing():
    """Verify rule-based and intent parsing for fashion queries."""
    # 1. Wardrobe / Accessory query
    q_scarf = "Show me all shots with scarves"
    parsed_scarf = parse_query_intent(q_scarf)
    filters_scarf = parsed_scarf.get("filters", {})
    assert filters_scarf.get("wardrobe_item") == "scarf"

    # 2. Photographer query across genres
    q_photog = "Find photoshoots by Terry Richardson"
    parsed_photog = parse_query_intent(q_photog)
    filters_photog = parsed_photog.get("filters", {})
    assert filters_photog.get("photographer") == "Terry Richardson"

    # 3. Designer and price query
    q_des = "Find jackets by Givenchy under $5000"
    parsed_des = parse_query_intent(q_des)
    filters_des = parsed_des.get("filters", {})
    assert filters_des.get("designer_brand") == "Givenchy"
    assert filters_des.get("wardrobe_item") == "jacket"
    assert filters_des.get("price_max") == 5000.0

    # 4. Celebrity cover query
    q_cover = "Who was the cover girl for Harper's Bazaar April 2015?"
    parsed_cover = parse_query_intent(q_cover)
    filters_cover = parsed_cover.get("filters", {})
    assert filters_cover.get("is_cover_girl") is True
    assert filters_cover.get("year") == 2015


def test_playboy_query_intent_regression():
    """Verify legacy Playboy queries continue to parse with exact fidelity."""
    q_pb = "Find all pictorials featuring Carmella DeCesare with blonde hair in 2006"
    parsed_pb = parse_query_intent(q_pb)
    filters_pb = parsed_pb.get("filters", {})
    assert filters_pb.get("model_name") == "Carmella DeCesare"
    assert filters_pb.get("hair_color") == "Blonde"
    assert filters_pb.get("year") == 2006


def test_unified_media_agent_search(tmp_path):
    """Verify MediaAgent searches unified photographers, designers, accessories, and models."""
    custom_index = {
        "models": {
            "julianne_moore": {
                "model_name": "Julianne Moore",
                "appearances": [
                    {
                        "document_id": "Harpers_Bazaar_April_2015_USA",
                        "magazine_title": "Harper's Bazaar USA",
                        "page_number": 280,
                        "spread_pages": [280, 281],
                        "photographer": "Camilla Akrans",
                        "stylist": "Elissa Santisi",
                        "wardrobe_items": [{"designer": "Givenchy", "item": "jacket", "price_usd": 4650.0}]
                    }
                ]
            },
            "carmella_decesare": {
                "model_name": "Carmella DeCesare",
                "appearances": [
                    {
                        "document_id": "Playboys_Vixens_2006-08_09",
                        "magazine_title": "Playboy's Vixens",
                        "page_number": 50,
                        "physical_attributes": {"measurements": "34D-24-34", "height": "5'8\"", "hair_color": "Brunette"},
                        "photographer": "Terry Richardson"
                    }
                ]
            }
        },
        "photographers": {
            "terry_richardson": {
                "photographer": "Terry Richardson",
                "shoots": [
                    {"document_id": "Harpers_Bazaar_April_2015_USA", "magazine_title": "Harper's Bazaar USA", "page_number": 284},
                    {"document_id": "Playboys_Vixens_2006-08_09", "magazine_title": "Playboy's Vixens", "page_number": 50}
                ]
            }
        },
        "designers_and_brands": {
            "givenchy": {
                "brand_name": "Givenchy",
                "mentions": [
                    {"document_id": "Harpers_Bazaar_April_2015_USA", "page_number": 280, "item": "jacket", "price_usd": 4650.0}
                ]
            }
        },
        "wardrobe_accessories": {
            "scarf": [
                {"document_id": "Harpers_Bazaar_April_2015_USA", "page_number": 285, "designer": "Saint Laurent", "item": "scarf", "price_usd": 325.0}
            ]
        },
        "issues": {}
    }

    agent = MediaAgent()
    agent.master_index = custom_index

    # 1. Search for photographer across publications
    results_photog = agent.search("Find shoots by Terry Richardson")
    output_photog = agent.format_retrieval_response("Find shoots by Terry Richardson", results_photog)
    assert "Terry Richardson" in output_photog
    assert "Harper's Bazaar USA" in output_photog
    assert "Playboy's Vixens" in output_photog

    # 2. Search for designer
    results_des = agent.search("Find Givenchy pieces")
    output_des = agent.format_retrieval_response("Find Givenchy pieces", results_des)
    assert "Givenchy" in output_des
    assert "4650" in output_des

    # 3. Search for wardrobe accessory
    results_scarf = agent.search("Show me all shots with scarves")
    output_scarf = agent.format_retrieval_response("Show me all shots with scarves", results_scarf)
    assert "scarf" in output_scarf.lower()
    assert "Saint Laurent" in output_scarf

    # 4. Search for fashion model vs glamour model formatting
    results_hb = agent.search("Julianne Moore")
    output_hb = agent.format_retrieval_response("Julianne Moore", results_hb)
    assert "Julianne Moore" in output_hb
    assert "Vital Stats: Measurements: Unspecified" not in output_hb  # Vital stats placeholder suppressed!
    assert "Camilla Akrans" in output_hb

    results_pb = agent.search("Carmella DeCesare")
    output_pb = agent.format_retrieval_response("Carmella DeCesare", results_pb)
    assert "Carmella DeCesare" in output_pb
    assert "34D-24-34" in output_pb  # PB vital stats rendered!
