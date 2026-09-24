"""
OCR Processor for the Rules Lawyer pipeline.

Hybrid text extraction: native PyMuPDF text first, RapidOCR fallback for scanned pages.
Includes duplicate detection via fuzzy string matching.
"""

import os
import re
import difflib
import fitz  # PyMuPDF
try:
    from rapidocr_onnxruntime import RapidOCR
except (ImportError, ModuleNotFoundError):
    RapidOCR = None

import numpy as np
from PIL import Image

DATA_DIR = "data/upfront"
OUTPUT_DIR = "data/upfront_text"

# Minimum character threshold — pages below this are flagged for OCR
MIN_CHARS_THRESHOLD = 50

# Duplicate detection similarity threshold
DUPLICATE_THRESHOLD = 0.90

# OCR DPI for rendering scanned pages (150 DPI provides high accuracy for small fonts and tables)
OCR_DPI = int(os.environ.get("OCR_DPI", "150"))


def init_ocr():
    """Initialize the RapidOCR engine."""
    if RapidOCR is None:
        raise ImportError("rapidocr_onnxruntime is not installed in the current Python environment.")
    ocr = RapidOCR(use_cls=False)
    return ocr


def extract_native_text(doc, page_idx):
    """Extract text from a PDF page using native PyMuPDF extraction."""
    page = doc[page_idx]
    text = page.get_text()
    return text.strip()


def sort_ocr_detections_by_reading_order(detections, page_width, page_height):
    """
    Sort RapidOCR detections into natural multi-column reading order.
    Detects 1-column, 2-column, or 3-column page layouts and orders text top-to-bottom
    within each column, while keeping full-width headers/footers in place.
    """
    if not detections:
        return []

    boxes = []
    for d in detections:
        bbox = d[0]
        text = d[1] if len(d) > 1 else ""
        if not text or not text.strip():
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width = x_max - x_min
        height = y_max - y_min
        boxes.append({
            "text": text.strip(),
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "x_mid": (x_min + x_max) / 2.0,
            "y_mid": (y_min + y_max) / 2.0,
            "width": width,
            "height": height,
        })

    if not boxes:
        return []

    # Threshold for full-width spanning elements (titles, page headers, footers)
    spanning_threshold = page_width * 0.65

    # Split into spanning (headers/footers) vs column content
    non_spanning = [b for b in boxes if b["width"] < spanning_threshold]
    headers = [b for b in boxes if b["width"] >= spanning_threshold and b["y_mid"] < page_height * 0.2]
    footers = [b for b in boxes if b["width"] >= spanning_threshold and b["y_mid"] >= page_height * 0.8]
    mid_spanning = [b for b in boxes if b["width"] >= spanning_threshold and page_height * 0.2 <= b["y_mid"] < page_height * 0.8]

    if not non_spanning:
        boxes.sort(key=lambda b: (b["y_min"], b["x_min"]))
        return [b["text"] for b in boxes]

    # Check for 3-column layout (density in 3 distinct horizontal zones)
    col3_1 = [b for b in non_spanning if b["x_mid"] < page_width * 0.35]
    col3_2 = [b for b in non_spanning if page_width * 0.35 <= b["x_mid"] < page_width * 0.66]
    col3_3 = [b for b in non_spanning if b["x_mid"] >= page_width * 0.66]
    is_3_col = len(col3_1) >= 3 and len(col3_2) >= 3 and len(col3_3) >= 3

    # Check for 2-column layout
    col2_1 = [b for b in non_spanning if b["x_mid"] < page_width * 0.50]
    col2_2 = [b for b in non_spanning if b["x_mid"] >= page_width * 0.50]
    is_2_col = not is_3_col and len(col2_1) >= 4 and len(col2_2) >= 4

    if is_3_col:
        col3_1.sort(key=lambda b: b["y_min"])
        col3_2.sort(key=lambda b: b["y_min"])
        col3_3.sort(key=lambda b: b["y_min"])
        headers.sort(key=lambda b: b["y_min"])
        mid_spanning.sort(key=lambda b: b["y_min"])
        footers.sort(key=lambda b: b["y_min"])
        sorted_boxes = headers + col3_1 + col3_2 + col3_3 + mid_spanning + footers
    elif is_2_col:
        col2_1.sort(key=lambda b: b["y_min"])
        col2_2.sort(key=lambda b: b["y_min"])
        headers.sort(key=lambda b: b["y_min"])
        mid_spanning.sort(key=lambda b: b["y_min"])
        footers.sort(key=lambda b: b["y_min"])
        sorted_boxes = headers + col2_1 + col2_2 + mid_spanning + footers
    else:
        # 1-column layout: sort primarily by Y (grouped into line bands) then X
        boxes.sort(key=lambda b: (round(b["y_min"] / 10), b["x_min"]))
        sorted_boxes = boxes

    return [b["text"] for b in sorted_boxes]


def extract_ocr_text(ocr, doc, page_idx, dpi=None):
    """Extract text from a scanned PDF page using RapidOCR with column layout sorting."""
    page = doc[page_idx]
    eff_dpi = dpi or OCR_DPI
    # Render page to pixmap at specified DPI
    mat = fitz.Matrix(eff_dpi / 72, eff_dpi / 72)
    pix = page.get_pixmap(matrix=mat)

    # Convert pixmap to PIL Image for RapidOCR
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_array = np.array(img)

    # Run OCR with rotation classifier disabled (rulebook scans are upright)
    result, _ = ocr(img_array, use_cls=False)

    if result is None:
        return ""

    lines = sort_ocr_detections_by_reading_order(result, pix.width, pix.height)
    return "\n".join(lines)


def _process_single_page_worker(ocr, pdf_path, page_idx, dpi):
    """Worker function for concurrent page processing with dedicated doc handle."""
    doc = fitz.open(pdf_path)
    try:
        native_text = extract_native_text(doc, page_idx)
        if len(native_text) >= MIN_CHARS_THRESHOLD:
            return page_idx, "native", native_text, len(native_text)
        
        ocr_text = extract_ocr_text(ocr, doc, page_idx, dpi=dpi)
        if len(ocr_text.strip()) > 0:
            return page_idx, "ocr", ocr_text, len(ocr_text)
        return page_idx, "empty", "", 0
    finally:
        doc.close()


def process_pdf(pdf_path, ocr, max_pages=None, dpi=None, max_workers=None):
    """
    Process a single PDF: extract text from all pages using native or OCR.
    Uses multi-threaded parallel execution across vCPUs for scanned pages.
    Returns (full_text, stats_dict).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    fname = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    total_doc_pages = len(doc)
    doc.close()

    limit = min(total_doc_pages, max_pages) if max_pages else total_doc_pages
    eff_dpi = dpi or OCR_DPI
    eff_workers = max_workers or 6

    stats = {
        "file": fname,
        "total_pages": total_doc_pages,
        "processed_pages": limit,
        "native_pages": 0,
        "ocr_pages": 0,
        "empty_pages": 0,
        "total_chars": 0
    }

    print(f"  [OCR Processor] Processing {limit}/{total_doc_pages} pages for {fname} (DPI={eff_dpi}, workers={eff_workers})...")

    # Concurrent page extraction with progress logging
    page_indices = list(range(limit))
    results = []
    with ThreadPoolExecutor(max_workers=eff_workers) as executor:
        future_to_idx = {
            executor.submit(_process_single_page_worker, ocr, pdf_path, idx, eff_dpi): idx
            for idx in page_indices
        }
        done_count = 0
        for future in as_completed(future_to_idx):
            res = future.result()
            results.append(res)
            done_count += 1
            if done_count % 25 == 0 or done_count == limit:
                print(f"  [OCR Processor Progress] {done_count}/{limit} pages extracted...")

    # Sort results deterministically by page index
    results.sort(key=lambda r: r[0])

    pages_text = []
    for page_idx, ptype, text, char_count in results:
        stats["total_chars"] += char_count
        if ptype == "native":
            stats["native_pages"] += 1
            pages_text.append(f"--- PAGE {page_idx + 1} ---\n{text}")
        elif ptype == "ocr":
            stats["ocr_pages"] += 1
            pages_text.append(f"--- PAGE {page_idx + 1} [OCR] ---\n{text}")
        else:
            stats["empty_pages"] += 1
            pages_text.append(f"--- PAGE {page_idx + 1} [EMPTY] ---")

    print(f"  [OCR Processor Complete] {limit} pages -> native={stats['native_pages']}, ocr={stats['ocr_pages']}, empty={stats['empty_pages']}, chars={stats['total_chars']}")
    full_text = "\n\n".join(pages_text)
    return full_text, stats



def process_pdf_to_text(pdf_path: str, ocr, max_pages: int = None,
                        ocr_substitutions: dict = None,
                        ocr_regex_substitutions: list = None,
                        dpi: int = None) -> str:
    """
    OCR-extract all pages in a PDF and return a single concatenated text
    string with PAGE markers — suitable for direct use in the ingestion pipeline.
    """
    try:
        full_text, stats = process_pdf(pdf_path, ocr, max_pages=max_pages, dpi=dpi)
    except Exception as e:
        print(f"  [process_pdf_to_text] process_pdf failed for {pdf_path}: {e}")
        return ""

    # Trim to max_pages by PAGE markers
    if max_pages:
        lines = full_text.split("\n")
        trimmed = []
        page_count = 0
        for line in lines:
            if line.startswith("--- PAGE "):
                page_count += 1
                if page_count > max_pages:
                    break
            trimmed.append(line)
        full_text = "\n".join(trimmed)

    # Apply profile-defined regex substitutions
    if ocr_regex_substitutions and isinstance(ocr_regex_substitutions, list):
        for entry in ocr_regex_substitutions:
            pat = entry.get("pattern")
            rep = entry.get("replacement", "")
            if pat:
                full_text = re.sub(pat, rep, full_text)

    # Apply profile-defined literal substitutions
    if ocr_substitutions and isinstance(ocr_substitutions, dict):
        for bad, good in ocr_substitutions.items():
            if len(bad) > 0:
                full_text = full_text.replace(bad, good)

    return full_text


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
