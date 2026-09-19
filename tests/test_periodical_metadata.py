"""
Test Suite: Generalized Filename & Periodical Metadata Extraction (TDD).

Validates that parse_filename_metadata in ingest_visual.py extracts publication,
title, year, issue_date, and clean_stem accurately for both high-fashion magazines
(Harper's Bazaar, etc.) and glamour publications (Playboy Special Editions, Germany).
"""

import os
import pytest
from engine.ingestion.ingest_visual import parse_filename_metadata


def test_harpers_bazaar_filename_metadata():
    """Verify Harper's Bazaar filenames are parsed accurately without Playboy prefix."""
    hb_path = "data/HB/Harpers_Bazaar_April_2015_USA.pdf"
    meta = parse_filename_metadata(hb_path)
    
    assert meta["publication"] == "Harper's Bazaar"
    assert "Harper's Bazaar" in meta["magazine_title"]
    assert "Playboy" not in meta["magazine_title"]
    assert meta["year"] == 2015
    assert "April 2015" in meta["issue_date"]
    assert meta["clean_stem"] == "Harpers_Bazaar_April_2015_USA"


def test_playboy_special_editions_regression():
    """Verify legacy Playboy filenames retain exact metadata output."""
    pb_vixens = "data/PB/Playboys_Vixens_2006-08_09.pdf"
    meta = parse_filename_metadata(pb_vixens)
    
    assert meta["publication"] == "Playboy Special Editions"
    assert meta["magazine_title"] == "Playboy's Vixens"
    assert meta["year"] == 2006
    assert "2006" in meta["issue_date"]
    assert meta["clean_stem"] == "Playboys_Vixens_2006-08_09"


def test_playboy_germany_regression():
    """Verify German Playboy editions retain exact metadata output."""
    pb_ger = "data/PB/Playboy_Germany_2011-01_Pdf_Xxx_Adult_Magazine.pdf"
    meta = parse_filename_metadata(pb_ger)
    
    assert meta["publication"] == "Playboy Germany"
    assert meta["year"] == 2011
    assert "2011" in meta["issue_date"]
    assert "Playboy" in meta["magazine_title"]
    assert meta["clean_stem"] == "Playboy_Germany_2011-01"


def test_playboy_calendar_regression():
    """Verify calendar issue metadata extraction."""
    pb_cal = "data/PB/Playboy_2005_Swimsuit_Calendar.pdf"
    meta = parse_filename_metadata(pb_cal)
    
    assert meta["publication"] == "Playboy Special Editions"
    assert meta["year"] == 2005
    assert meta["clean_stem"] == "Playboy_2005_Swimsuit_Calendar"
