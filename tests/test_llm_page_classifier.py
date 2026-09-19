"""
Test Suite: Generalized & LLM-Driven Page Archetype Classifier (TDD).

Validates that classify_page_archetype accurately identifies page roles for both
high-fashion magazines (Harper's Bazaar) and glamour magazines (Playboy):
- Cover & Back Cover
- Advertisements (including minimal-text luxury fashion ads like Dior/Chanel)
- Table of Contents (including segmented multi-page TOCs)
- Fashion Editorials & Feature Pictorials
- Structured Directories (PB Bio Cards & HB 'Where to Buy' stockist indices)
- Shopping Showcases (Collage grids with prices)
"""

import pytest
from engine.ingestion.ingest_visual import classify_page_archetype


def test_cover_and_back_cover():
    """Verify boundaries are always identified as Cover and Back_Cover."""
    assert classify_page_archetype(0, 330, "HARPER'S BAZAAR SPRING FASHION") == "Cover"
    assert classify_page_archetype(329, 330, "CHANEL ADVERTISEMENT BACK COVER") == "Back_Cover"


def test_luxury_minimal_advertisements():
    """Verify minimal-text luxury ads (Dior, Chanel, Fendi) classify as Advertisement, not Feature_Pictorial."""
    dior_ad = "800.929.Dior (3467) Dior.com"
    chanel_ad = "CHANEL BOUTIQUES 800.550.0005 chanel.com"
    fendi_ad = "FENDI BOUTIQUES FENDI.COM"
    estee_ad = "esteelauder.com © 2015 Estée Lauder Inc YOU'RE FLAWLESS"
    
    assert classify_page_archetype(5, 330, dior_ad) == "Advertisement"
    assert classify_page_archetype(20, 330, chanel_ad) == "Advertisement"
    assert classify_page_archetype(33, 330, fendi_ad) == "Advertisement"
    assert classify_page_archetype(1, 330, estee_ad) == "Advertisement"


def test_segmented_table_of_contents():
    """Verify multi-page segmented TOCs in HB classify as Table_of_Contents."""
    toc_highlights = (
        "68 HIGHLIGHTS CONTINUED ON PAGE 72 APRIL 2015\n"
        "THE BAZAAR A suede spin on Burberry's iconic trench\n"
        "THE LIST 24 hours with designer Isabel Marant\n"
        "THE NEWS The best from Paris couture shows\n"
        "THE BEAUTY BAZAAR Secrets to great skin"
    )
    toc_fashion = (
        "CONTINUED ON PAGE 78 APRIL 2015\n"
        "FASHION\n"
        "JULIANNE MOORE: THE MOORE, THE BETTER 276\n"
        "By Laura Brown Photographs by Camilla Akrans\n"
        "WHAT'S HOT NOW 280 Photographs by Terry Richardson\n"
        "FABULOUS AT EVERY AGE PORTFOLIO 288 Photographs by Mark Abrahams\n"
        "HOTTEST SHOES OF THE SEASON 308 Photographs by Dan Forbes"
    )
    assert classify_page_archetype(79, 330, toc_highlights) == "Table_of_Contents"
    assert classify_page_archetype(83, 330, toc_fashion) == "Table_of_Contents"


def test_stockist_directory():
    """Verify 'Where to Buy' and stockist directories classify as Stockist_Directory or Structured_Grid_Directory."""
    stockist_text = (
        "320 WHERE TO BUY\n"
        "Harper's Bazaar April 2015 issue no. 3632\n"
        "Page 277 Gucci gown, $5,200. gucci.com. Lisa Eisner Jewelry cuff, $3,200.\n"
        "Page 279 Chloé blouse, $995. Max Mara pants, $645. Cartier ring, $116,000. 800-CARTIER.\n"
        "Page 280 Givenchy by Riccardo Tisci jacket, $4,650, and brooch. 305-576-6250.\n"
        "Saint Laurent by Hedi Slimane turban, $2,035. Tom Ford dress, prices upon request."
    )
    result = classify_page_archetype(323, 330, stockist_text)
    assert result in ["Stockist_Directory", "Structured_Grid_Directory"]


def test_playboy_bio_cards_regression():
    """Verify PB vital stats bio cards continue to classify as Structured_Grid_Directory."""
    pb_stats = (
        "BARE FACTS VITAL STATS\n"
        "Elizabeth JoAnne 36DD-22-36 5'11\" 135 lbs\n"
        "Heather Rene 34D-24-35 5'3\" 115 lbs\n"
        "Cynthia Kaye 34C-24-34 5'6\" 120 lbs"
    )
    assert classify_page_archetype(85, 100, pb_stats) == "Structured_Grid_Directory"


def test_fashion_editorial():
    """Verify narrative pictorial spreads classify as Feature_Pictorial."""
    editorial = (
        "JULIANNE MOORE: THE MOORE, THE BETTER\n"
        "Julianne Moore has just arrived back at her hotel in L.A. after an Oscar nominees luncheon.\n"
        "THIS PAGE: Jacket and brooch, Givenchy by Riccardo Tisci. Skirt, Alexander Wang.\n"
        "Photographs by Camilla Akrans. Fashion Editor: Elissa Santisi."
    )
    assert classify_page_archetype(281, 330, editorial) == "Feature_Pictorial"
