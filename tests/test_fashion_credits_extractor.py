"""
Test Suite: Fashion Credits & Stockist Directory Extraction (TDD).

Validates that:
1. extract_stockist_directory_entries accurately parses designers, garment items,
   prices, page references, and stockist contacts from the 'Where to Buy' index.
2. extract_fashion_and_creative_credits extracts photographer, fashion editor/stylist,
   hair, makeup, and garment mentions from spread captions.
3. Legacy PB credits (e.g. 'PHOTOGRAPHY BY JOSH RYAN') continue to parse seamlessly.
"""

import pytest
from engine.ingestion.ingest_visual import (
    extract_stockist_directory_entries,
    extract_fashion_and_creative_credits,
)


def test_extract_stockist_directory_entries():
    """Verify parsing of multi-item designer credits with prices from Where to Buy index."""
    stockist_snippet = (
        "Page 277 Gucci gown, $5,200. gucci.com. Lisa Eisner Jewelry cuff, $3,200. Maxfield, L.A.; 310-274-8800. "
        "Page 279 Chloé blouse, $995. Max Mara pants, $645. 212-879-6100. Cartier ring, $116,000. 800-CARTIER. "
        "What's Hot Now Page 280 Givenchy by Riccardo Tisci jacket, $4,650, and brooch, price upon request. 305-576-6250. "
        "T by Alexander Wang skirt, $700. Saint Laurent by Hedi Slimane hat, $990, and belt, $690. 212-980-2970. "
        "Page 285 Valentino jacket, $23,000. thefursalon.com. Dolce & Gabbana dress, $3,295, and briefs, $295. "
        "Saint Laurent by Hedi Slimane scarf, $325, and shoes, $1,295."
    )

    entries = extract_stockist_directory_entries(stockist_snippet)
    assert len(entries) >= 6

    # Test Gucci gown
    gucci = next((e for e in entries if "gucci" in e["designer"].lower() and "gown" in e["item"].lower()), None)
    assert gucci is not None
    assert gucci["page_ref"] == 277
    assert gucci["price_usd"] == 5200.0

    # Test Cartier ring
    cartier = next((e for e in entries if "cartier" in e["designer"].lower()), None)
    assert cartier is not None
    assert cartier["page_ref"] == 279
    assert cartier["price_usd"] == 116000.0

    # Test Givenchy jacket
    givenchy = next((e for e in entries if "givenchy" in e["designer"].lower() and "jacket" in e["item"].lower()), None)
    assert givenchy is not None
    assert givenchy["page_ref"] == 280
    assert givenchy["price_usd"] == 4650.0

    # Test Valentino jacket
    valentino = next((e for e in entries if "valentino" in e["designer"].lower()), None)
    assert valentino is not None
    assert valentino["page_ref"] == 285
    assert valentino["price_usd"] == 23000.0

    # Test multi-item stockist clauses: Saint Laurent scarf ($325) and belt ($690)
    saint_laurent_scarf = next((e for e in entries if "saint laurent" in e["designer"].lower() and "scarf" in e["item"].lower()), None)
    assert saint_laurent_scarf is not None
    assert saint_laurent_scarf["price_usd"] == 325.0

    saint_laurent_belt = next((e for e in entries if "saint laurent" in e["designer"].lower() and "belt" in e["item"].lower()), None)
    assert saint_laurent_belt is not None
    assert saint_laurent_belt["price_usd"] == 690.0


def test_extract_fashion_and_creative_credits():
    """Verify parsing of photographer, stylist, hair, makeup, and wardrobe from caption."""
    caption = (
        "JULIANNE MOORE: THE MOORE, THE BETTER\n"
        "THIS PAGE: Jacket and brooch, Givenchy by Riccardo Tisci. Skirt, Alexander Wang.\n"
        "Photographs by Camilla Akrans. Fashion Editor: Elissa Santisi. "
        "Hair: Marcus Francis; makeup: Elaine Offers; manicure: April Foreman."
    )

    credits = extract_fashion_and_creative_credits(caption)
    assert credits["photographer"] == "Camilla Akrans"
    assert credits["fashion_editor_stylist"] == "Elissa Santisi"
    assert credits["hair"] == "Marcus Francis"
    assert credits["makeup"] == "Elaine Offers"
    assert len(credits["wardrobe_mentions"]) >= 2
    assert any("givenchy" in w.lower() for w in credits["wardrobe_mentions"])


def test_playboy_credits_regression():
    """Verify legacy Playboy photography credits parse cleanly."""
    pb_caption = "PHOTOGRAPHY BY ARNY FREYTAG. Miss December 1998."
    credits = extract_fashion_and_creative_credits(pb_caption)
    assert credits["photographer"] == "Arny Freytag"
