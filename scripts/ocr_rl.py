"""
OCR runner for data/rl — processes scanned PDFs one at a time, slowly.
Outputs .txt files alongside the PDFs in data/rl_text/.
Native text pages use PyMuPDF; scanned pages fall back to RapidOCR at 300 DPI.
"""

import os
import sys
import time
import fitz
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

DATA_DIR   = "data/rl"
OUTPUT_DIR = "data/rl_text"
MIN_CHARS  = 50
DPI        = 300

# Files confirmed to need OCR (from the pre-flight check)
OCR_TARGETS = [
    "2nd-ACR.pdf",
    "Interceptor_Overlays.pdf",
    "Interceptor_and_Leviathan_Skies.pdf",
    "RL-Int_Sheet.pdf",
    "TOG_Grav_Vehicles.pdf",
    "pdfcoffee.com_128283435-renegade-legion-renegade-fighter-briefing-pdf-free.pdf",
    "pdfcoffee.com_renegade-legion-distant-firepdf-pdf-free.pdf",
    "pdfcoffee.com_renegade-legion-legionnairepdf-pdf-free.pdf",
    "pdfcoffee.com_renegade-legion-prefect-pdf-free.pdf",
    "pdfcoffee.com_renegade-legion-tog-fighter-briefingpdf-pdf-free.pdf",
]

# Pause between pages (seconds) — keeps CPU load low
PAGE_PAUSE = 0.5


def process_file(pdf_path, ocr):
    fname = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    total = len(doc)
    pages_out = []
    native_count = ocr_count = empty_count = 0

    for i in range(total):
        native = doc[i].get_text().strip()

        if len(native) >= MIN_CHARS:
            pages_out.append(f"--- PAGE {i+1} ---\n{native}")
            native_count += 1
        else:
            page = doc[i]
            mat = fitz.Matrix(DPI / 72, DPI / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            result, _ = ocr(np.array(img))

            if result:
                text = "\n".join(det[1] for det in result)
                pages_out.append(f"--- PAGE {i+1} [OCR] ---\n{text}")
                ocr_count += 1
            else:
                pages_out.append(f"--- PAGE {i+1} [EMPTY] ---")
                empty_count += 1

        # Progress report every 10 pages
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"    Page {i+1}/{total}  (native={native_count}, ocr={ocr_count}, empty={empty_count})")
            sys.stdout.flush()

        time.sleep(PAGE_PAUSE)

    doc.close()
    return "\n\n".join(pages_out), native_count, ocr_count, empty_count


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Initialising RapidOCR engine...")
    ocr = RapidOCR()
    print(f"Ready. Processing {len(OCR_TARGETS)} files from {DATA_DIR}/\n")
    print("=" * 70)

    for idx, fname in enumerate(OCR_TARGETS, 1):
        pdf_path = os.path.join(DATA_DIR, fname)
        txt_path = os.path.join(OUTPUT_DIR, fname.replace(".pdf", ".txt"))

        if os.path.exists(txt_path):
            print(f"[{idx}/{len(OCR_TARGETS)}] SKIP (already done): {fname}")
            continue

        print(f"\n[{idx}/{len(OCR_TARGETS)}] {fname}")
        sys.stdout.flush()

        if not os.path.exists(pdf_path):
            print(f"  WARNING: file not found, skipping.")
            continue

        start = time.time()
        full_text, native, ocr_pages, empty = process_file(pdf_path, ocr)
        elapsed = time.time() - start

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        chars = len(full_text)
        mins = elapsed / 60
        print(f"  Done in {mins:.1f} min — {chars:,} chars extracted")
        print(f"  Output: {txt_path}")
        print(f"  Sleeping 5s before next file...")
        sys.stdout.flush()
        time.sleep(5)

    print("\n" + "=" * 70)
    print("OCR COMPLETE — all output in data/rl_text/")
    print("Next step: python engine/ingestion/auto_discover.py data/rl \"Renegade Legion\" renegade_legion")


if __name__ == "__main__":
    main()
