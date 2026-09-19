"""
Test Suite: Entity Cleaning, OCR Normalization, and Vision-Guided Subject Thumbnail Generation.
"""

import os
import json
import pytest
from PIL import Image
import numpy as np
from engine.ingestion.ingest_visual import (
    normalize_ocr_text,
    clean_model_name,
    crop_and_save_subject_thumbnail,
    crop_and_save_headshot,
    extract_bio_cards_from_text,
    slugify
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


def test_normalize_ocr_text_unglues_words():
    """Verify that OCR text with glued words and numbers is cleanly un-glued."""
    raw1 = "DutchbeautySaskiaLinssenhatesto stay inone place long"
    norm1 = normalize_ocr_text(raw1)
    assert "Dutch beauty" in norm1
    assert "Saskia Linssen" in norm1
    assert "hates to" in norm1 or "hatesto" in norm1

    raw2 = "Brisbane-bornTeresaLinnaneisblessed with good luck"
    norm2 = normalize_ocr_text(raw2)
    assert "Teresa Linnane" in norm2

    raw3 = "SultryFrancesca Nicodemi was Miss April1992 in theGreekeditionofPLAYBOY."
    norm3 = normalize_ocr_text(raw3)
    assert "Francesca Nicodemi" in norm3
    assert "April 1992" in norm3


def test_clean_model_name_filters_and_normalizes():
    """Verify filtering of false positives, stopwords, demonyms, and newlines."""
    # False positives that must be rejected (return None)
    assert clean_model_name("When Teresa") is None
    assert clean_model_name("Miss April") is None
    assert clean_model_name("Miss July") is None
    assert clean_model_name("Cover Girl") is None
    assert clean_model_name("Il Stats") is None
    assert clean_model_name("Nj Stats") is None
    assert clean_model_name("Hugh Hefner") is None
    assert clean_model_name("The Calendar") is None
    assert clean_model_name("Playmate Of The Month") is None
    assert clean_model_name("She Originally") is None
    assert clean_model_name("Adores Sushi And") is None
    assert clean_model_name("Personal Credo") is None
    assert clean_model_name("That Is Not An Unreal") is None
    assert clean_model_name("Kelly James Studies Business And Computers With An Eye Toward Desktop Publ") is None
    assert clean_model_name("SingleWord") is None
    assert clean_model_name("") is None
    assert clean_model_name(None) is None

    # Demonym and adjective stripping
    assert clean_model_name("Dutch beauty Saskia Linssen") == "Saskia Linssen"
    assert clean_model_name("Brisbane-born Teresa Linnane") == "Teresa Linnane"
    assert clean_model_name("Sultry Francesca Nicodemi") == "Francesca Nicodemi"
    assert clean_model_name("Polish-born Renata Gounis") == "Renata Gounis"
    assert clean_model_name("Mansion\nPennelope Jimenez") == "Pennelope Jimenez"
    assert clean_model_name("Lisa\nMatthews") == "Lisa Matthews"
    assert clean_model_name("Helle Michaelsen") == "Helle Michaelsen"
    assert clean_model_name("Maria Solasso") == "Maria Solasso"
    assert clean_model_name("Marta Caracciolo") == "Marta Caracciolo"
    assert clean_model_name("Pia Contestabili") == "Pia Contestabili"
    assert clean_model_name("Patricia Aguirre") == "Patricia Aguirre"


def test_extract_bio_cards_filters_directory_headers():
    """Verify extract_bio_cards_from_text doesn't create cards for 'Il Stats' or directory headers."""
    sample_text = """
    IL STATS
    Height: 5'6"
    Weight: 115 lbs
    34C-24-34
    Photographed by Josh Ryan
    pages 10-14
    """
    cards = extract_bio_cards_from_text(sample_text, 1)
    # 'Il Stats' is rejected by clean_model_name
    assert len(cards) == 0


def test_crop_and_save_subject_thumbnail_fallback(tmp_path):
    """Verify fallback cropping generates valid image file with aesthetic proportions."""
    img = Image.new("RGB", (1000, 1500), color=(200, 200, 200))
    out_path = str(tmp_path / "thumb_fallback.jpg")

    res_path = crop_and_save_subject_thumbnail(img, out_path)
    assert os.path.exists(res_path)

    saved_img = Image.open(res_path)
    # Target center crop (12% to 88% width = 760px, 8% to 85% height = 1155px)
    assert saved_img.size == (760, 1155)


def test_crop_and_save_subject_thumbnail_positional(tmp_path):
    """Verify positional routing crops left vs right vs top vs bottom halves accurately."""
    img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
    
    out_left = str(tmp_path / "thumb_left.jpg")
    crop_and_save_subject_thumbnail(img, out_left, positional_hint="left")
    img_l = Image.open(out_left)
    assert img_l.size[0] < 500  # Left half width

    out_right = str(tmp_path / "thumb_right.jpg")
    crop_and_save_subject_thumbnail(img, out_right, positional_hint="right")
    img_r = Image.open(out_right)
    assert img_r.size[0] < 500  # Right half width


def test_backwards_compatible_crop_and_save_headshot(tmp_path):
    """Verify crop_and_save_headshot acts as alias for crop_and_save_subject_thumbnail."""
    img = Image.new("RGB", (500, 500), color=(100, 150, 200))
    out_path = str(tmp_path / "headshot_alias.jpg")

    res = crop_and_save_headshot(img, out_path)
    assert os.path.exists(res)


def test_crop_model_thumbnail_from_spread(tmp_path):
    """Verify crop_model_thumbnail_from_spread fallback and file creation."""
    import fitz
    from engine.ingestion.ingest_visual import crop_model_thumbnail_from_spread

    # Create dummy 3-page doc in memory
    doc = fitz.open()
    for _ in range(3):
        p = doc.new_page(width=400, height=600)
        p.draw_rect(fitz.Rect(50, 50, 350, 550), color=(0.8, 0.8, 0.8), fill=(0.9, 0.9, 0.9))
        
    out_path = str(tmp_path / "spread_thumb.jpg")
    res = crop_model_thumbnail_from_spread(doc, 1, 2, out_path, model_name="Test Model")
    assert os.path.exists(res)
    with Image.open(res) as im:
        assert im.size[0] > 0 and im.size[1] > 0

