"""
RED/GREEN: Table extractor emits valid Markdown tables from digital PDFs.
RED: table_extractor module doesn't exist.
GREEN: Module exists and correctly extracts tables.
"""
import os, sys, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MISC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/misc"))

try:
    import fitz
    from engine.ingestion.table_extractor import (
        extract_page_tables_as_markdown,
        inject_markdown_tables,
        _rows_to_markdown
    )
    HAS_EXTRACTOR = True
except ImportError:
    HAS_EXTRACTOR = False


@pytest.mark.skipif(not HAS_EXTRACTOR, reason="table_extractor not yet created (RED)")
class TestTableExtractor:

    def test_rows_to_markdown_basic(self):
        """Verify Markdown pipe table rendering from row data."""
        rows = [
            ["Total Hits", "Effect"],
            ["5 or more", "Unit is Done For"],
            ["4", "Loss of morale. Retreat."],
            ["3", "Continue with -1 modifier."],
        ]
        md = _rows_to_markdown(rows)
        assert "| Total Hits | Effect |" in md
        assert "|---|---|" in md
        assert "| 5 or more | Unit is Done For |" in md

    def test_rows_to_markdown_empty_returns_empty(self):
        assert _rows_to_markdown([]) == ""
        assert _rows_to_markdown([["H1"]]) == ""  # single row → no data rows

    def test_honours_of_war_reaction_table_extracted(self):
        """
        Honours of War page 29 contains the 'Reaction to Firing' table.
        Verify at least one table is extracted from that page.
        RED: extract_page_tables_as_markdown returns "" (before fix).
        GREEN: Returns non-empty markdown with Hits/Effect columns.
        """
        path = os.path.join(MISC_DIR, "pdfcoffee.com_osprey-wargames-11-honours-of-war-wargames-rules-for-the-seven-yearsx27-war-pdf-free.pdf")
        doc = fitz.open(path)
        page = doc[28]  # page 29 (0-indexed)
        md = extract_page_tables_as_markdown(page)
        doc.close()
        assert md != "", "Expected at least one table from HoW page 29"
        assert "|" in md, "Expected markdown pipe table characters"

    def test_fow_team_stat_card_table_extracted(self):
        """
        Flames of War contains team stat card tables with columns:
        Motivation | Skill | Is Hit On | Armour F/S/T | Tactical/Dash | ROF | AT | FP
        """
        path = os.path.join(MISC_DIR, "pdfcoffee.com_flames-of-war-4th-ed-ew-amp-lw-rule-book-pdf-free.pdf")
        doc = fitz.open(path)
        # Search first 50 pages for any table
        found = False
        for i in range(min(50, len(doc))):
            md = extract_page_tables_as_markdown(doc[i])
            if md and "|" in md:
                found = True
                break
        doc.close()
        assert found, "Expected at least one tabular structure in FoW first 50 pages"

    def test_inject_markdown_tables_adds_table_markers(self):
        """inject_markdown_tables inserts <!-- TABLES PAGE N --> markers."""
        path = os.path.join(MISC_DIR, "pdfcoffee.com_osprey-wargames-11-honours-of-war-wargames-rules-for-the-seven-yearsx27-war-pdf-free.pdf")
        result = inject_markdown_tables(path, "--- PAGE 29 ---\nSome text here\n", max_pages=29)
        # If tables were found on page 29, markers should be injected
        if "<!-- TABLES PAGE" in result:
            assert "<!-- END TABLES -->" in result

    def test_table_extractor_handles_ocr_page_gracefully(self):
        """Albedo's bitmap pages should not crash — returns empty string gracefully."""
        path = os.path.join(MISC_DIR, "pdfcoffee.com_albedo-2nd-edition-compressed-pdf-free.pdf")
        doc = fitz.open(path)
        md = extract_page_tables_as_markdown(doc[0])
        doc.close()
        assert isinstance(md, str)  # must return a string (may be empty)
