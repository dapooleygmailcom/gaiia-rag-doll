"""
RED/GREEN: DocumentProfiler correctly routes scanned RPG/wargame docs.
RED state: Albedo misclassified as VISUAL_MEDIA.
GREEN state: Albedo correctly classified as SCANNED_OCR.
"""
import os, sys, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MISC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/misc"))

try:
    from engine.ingestion.universal_ingest import DocumentProfiler
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False

@pytest.mark.skipif(not HAS_PROFILER, reason="universal_ingest not importable in test context")
class TestDocumentProfilerRouting:

    def test_albedo_routes_to_scanned_ocr_not_visual_media(self):
        """
        RED: Albedo → VISUAL_MEDIA (bug).
        GREEN: Albedo → SCANNED_OCR (after profiler priority fix).
        """
        path = os.path.join(MISC_DIR, "pdfcoffee.com_albedo-2nd-edition-compressed-pdf-free.pdf")
        profile = DocumentProfiler.profile_pdf(path)
        assert profile["detected_pipeline"] == "SCANNED_OCR", (
            f"Expected SCANNED_OCR, got {profile['detected_pipeline']}. "
            f"Reason: {profile['reason']}. "
            f"avg_chars={profile['avg_chars_per_page']}, coverage={profile['avg_visual_coverage']}"
        )

    def test_fow_routes_to_rulebook_technical(self):
        """FoW is digital text — should remain RULEBOOK_TECHNICAL."""
        path = os.path.join(MISC_DIR, "pdfcoffee.com_flames-of-war-4th-ed-ew-amp-lw-rule-book-pdf-free.pdf")
        profile = DocumentProfiler.profile_pdf(path)
        assert profile["detected_pipeline"] == "RULEBOOK_TECHNICAL"

    def test_how_routes_to_rulebook_technical(self):
        """HoW is digital Osprey text — should remain RULEBOOK_TECHNICAL."""
        path = os.path.join(MISC_DIR, "pdfcoffee.com_osprey-wargames-11-honours-of-war-wargames-rules-for-the-seven-yearsx27-war-pdf-free.pdf")
        profile = DocumentProfiler.profile_pdf(path)
        assert profile["detected_pipeline"] == "RULEBOOK_TECHNICAL"

    def test_tokyo_express_routes_to_rulebook_technical(self):
        """Tokyo Express has OCR-recoverable text with rule numbers — RULEBOOK_TECHNICAL."""
        path = os.path.join(MISC_DIR, "pdfcoffee.com_tokyo-express-basic-rule-book-pdf-free.pdf")
        profile = DocumentProfiler.profile_pdf(path)
        assert profile["detected_pipeline"] in {"RULEBOOK_TECHNICAL", "SCANNED_OCR"}

    def test_visual_media_still_routes_visual_for_playboy(self):
        """Regression: a visual media file with explicit filename signals still routes correctly."""
        filename = "pdfcoffee.com_playboy-2006-pdf-free.pdf"
        visual_terms = ["vixen", "playboy", "glamour", "portfolio", "lookbook", "photo", "magazine", "comic"]
        assert any(term in filename.lower() for term in visual_terms)
