"""
OCR Processor for the Rules Lawyer pipeline.

Hybrid text extraction: native PyMuPDF text first, RapidOCR fallback for scanned pages.
Includes duplicate detection via fuzzy string matching.
"""

import os
import difflib
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
import numpy as np
from PIL import Image

DATA_DIR = "data/upfront"
OUTPUT_DIR = "data/upfront_text"

# Minimum character threshold — pages below this are flagged for OCR
MIN_CHARS_THRESHOLD = 50

# Duplicate detection similarity threshold
DUPLICATE_THRESHOLD = 0.90

# OCR DPI for rendering scanned pages
OCR_DPI = 300


def init_ocr():
    """Initialize the RapidOCR engine."""
    ocr = RapidOCR()
    return ocr


def extract_native_text(doc, page_idx):
    """Extract text from a PDF page using native PyMuPDF extraction."""
    page = doc[page_idx]
    text = page.get_text()
    return text.strip()


def extract_ocr_text(ocr, doc, page_idx):
    """Extract text from a scanned PDF page using RapidOCR."""
    page = doc[page_idx]
    # Render page to pixmap at high DPI
    mat = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
    pix = page.get_pixmap(matrix=mat)

    # Convert pixmap to PIL Image for RapidOCR
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_array = np.array(img)

    # Run OCR
    result, _ = ocr(img_array)

    if result is None:
        return ""

    # RapidOCR returns list of [bbox, text, confidence]
    lines = []
    for detection in result:
        text = detection[1]
        lines.append(text)

    return "\n".join(lines)


def process_pdf(pdf_path, ocr):
    """
    Process a single PDF: extract text from all pages using native or OCR.
    Returns (full_text, stats_dict).
    """
    fname = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)

    pages_text = []
    stats = {
        "file": fname,
        "total_pages": len(doc),
        "native_pages": 0,
        "ocr_pages": 0,
        "empty_pages": 0,
        "total_chars": 0
    }

    for i in range(len(doc)):
        # Try native text extraction first
        native_text = extract_native_text(doc, i)

        if len(native_text) >= MIN_CHARS_THRESHOLD:
            pages_text.append(f"--- PAGE {i + 1} ---\n{native_text}")
            stats["native_pages"] += 1
            stats["total_chars"] += len(native_text)
        else:
            # Fall back to OCR
            ocr_text = extract_ocr_text(ocr, doc, i)
            if len(ocr_text.strip()) > 0:
                pages_text.append(f"--- PAGE {i + 1} [OCR] ---\n{ocr_text}")
                stats["ocr_pages"] += 1
                stats["total_chars"] += len(ocr_text)
            else:
                pages_text.append(f"--- PAGE {i + 1} [EMPTY] ---")
                stats["empty_pages"] += 1

    doc.close()
    full_text = "\n\n".join(pages_text)
    return full_text, stats


def check_duplicate(text_a, text_b, pages_to_compare=3):
    """
    Compare first N pages of two extracted texts using fuzzy string matching.
    Returns (is_duplicate, similarity_ratio).
    """
    # Extract first N pages from each text
    def get_first_pages(text, n):
        pages = text.split("--- PAGE ")
        # Filter out empty first element from split
        pages = [p for p in pages if p.strip()]
        return " ".join(pages[:n])

    sample_a = get_first_pages(text_a, pages_to_compare)
    sample_b = get_first_pages(text_b, pages_to_compare)

    if not sample_a or not sample_b:
        return False, 0.0

    # Normalize whitespace for comparison
    sample_a = " ".join(sample_a.split()).lower()
    sample_b = " ".join(sample_b.split()).lower()

    ratio = difflib.SequenceMatcher(None, sample_a, sample_b).ratio()
    return ratio >= DUPLICATE_THRESHOLD, ratio


def process_all():
    """
    Process all PDFs in the upfront data directory.
    Outputs .txt files and performs duplicate detection.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")])
    print(f"Found {len(pdf_files)} PDF files in {DATA_DIR}")
    print("=" * 70)

    ocr = init_ocr()
    all_results = {}
    all_stats = []

    for idx, fname in enumerate(pdf_files, 1):
        pdf_path = os.path.join(DATA_DIR, fname)
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {fname}")

        full_text, stats = process_pdf(pdf_path, ocr)
        all_results[fname] = full_text
        all_stats.append(stats)

        # Write output text file
        txt_name = fname.replace(".pdf", ".txt")
        txt_path = os.path.join(OUTPUT_DIR, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"  Pages: {stats['total_pages']} "
              f"(native={stats['native_pages']}, ocr={stats['ocr_pages']}, empty={stats['empty_pages']})")
        print(f"  Total chars: {stats['total_chars']}")
        print(f"  Output: {txt_path}")

    # --- Duplicate Detection ---
    print("\n" + "=" * 70)
    print("DUPLICATE DETECTION: up-front-rules.pdf vs Up_Front.pdf")
    print("=" * 70)

    scanned = "up-front-rules.pdf"
    native = "Up_Front.pdf"

    if scanned in all_results and native in all_results:
        is_dup, ratio = check_duplicate(all_results[scanned], all_results[native])
        print(f"  Fuzzy similarity (first 3 pages): {ratio:.2%}")
        if is_dup:
            print(f"  RESULT: DUPLICATE detected (>{DUPLICATE_THRESHOLD:.0%} threshold)")
            print(f"  ACTION: Skipping {scanned} — using {native} (native text) instead")
            # Remove the duplicate text file
            dup_txt = os.path.join(OUTPUT_DIR, scanned.replace(".pdf", ".txt"))
            if os.path.exists(dup_txt):
                os.rename(dup_txt, dup_txt + ".duplicate")
                print(f"  Renamed output to: {dup_txt}.duplicate")
        else:
            print(f"  RESULT: NOT a duplicate (<{DUPLICATE_THRESHOLD:.0%} threshold)")
            print(f"  ACTION: Treating as separate document version — both will be indexed")
    else:
        missing = []
        if scanned not in all_results:
            missing.append(scanned)
        if native not in all_results:
            missing.append(native)
        print(f"  WARNING: Could not compare — missing files: {', '.join(missing)}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"{'File':<55} {'Pages':>5} {'Native':>7} {'OCR':>5} {'Empty':>6} {'Chars':>8}")
    print("-" * 86)
    for s in all_stats:
        print(f"{s['file']:<55} {s['total_pages']:>5} {s['native_pages']:>7} "
              f"{s['ocr_pages']:>5} {s['empty_pages']:>6} {s['total_chars']:>8}")

    return all_results, all_stats


if __name__ == "__main__":
    process_all()
