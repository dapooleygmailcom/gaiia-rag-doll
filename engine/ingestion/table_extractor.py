"""
Table Extractor — Gaiia RAG Doll.

Extracts tabular data from PDF pages using PyMuPDF find_tables() (Tier 1)
with a spatial block-alignment fallback (Tier 2). Emits standard Markdown
pipe tables that the chunker can identify and preserve verbatim.

Supported game table types:
  - Combat resolution tables  (hits → effects)
  - Weapon stat cards / unit profiles
  - Hit location charts       (D100 → body region)
  - Torpedo hit probability matrices
  - Unit/Army list attributes
  - Terrain effect tables

Usage:
    from engine.ingestion.table_extractor import (
        extract_page_tables_as_markdown,
        inject_markdown_tables,
    )
"""

from __future__ import annotations

import re
import fitz  # PyMuPDF


# ─── Tier 1: PyMuPDF find_tables() ───────────────────────────────────────────

def extract_page_tables_as_markdown(page: fitz.Page) -> str:
    """
    Extract all tables from a PyMuPDF page and return them concatenated as
    Markdown pipe tables (or empty string if no tables are found).

    Uses PyMuPDF's built-in find_tables() heuristic for digital PDFs.
    Falls back to _spatial_block_table_fallback() for pages where the
    heuristic finds nothing or the page has only image content.

    Args:
        page: An open fitz.Page object.

    Returns:
        Markdown string (possibly empty).
    """
    try:
        finder = page.find_tables()
        if not finder.tables:
            return _spatial_block_table_fallback(page)

        md_parts = []
        for tbl in finder.tables:
            rows = tbl.extract()
            if not rows or len(rows) < 2:
                continue
            md = _rows_to_markdown(rows)
            if md:
                md_parts.append(md)

        result = "\n\n".join(md_parts)
        if not result:
            return _spatial_block_table_fallback(page)
        return result

    except AttributeError:
        # PyMuPDF < 1.23 — find_tables() not available
        return _spatial_block_table_fallback(page)
    except Exception:
        return ""


def _rows_to_markdown(rows: list) -> str:
    """
    Convert a list-of-lists row structure to a Markdown pipe table.

    Args:
        rows: List of rows, each row a list of cell strings.
              First row is treated as the header.

    Returns:
        Markdown string, or empty string if rows is empty or too short.
    """
    if not rows or len(rows) < 2:
        return ""

    # Determine column count from widest row
    ncols = max(len(row) for row in rows)
    if ncols < 1:
        return ""

    # Normalize: clean and pad all cells
    cleaned = []
    for row in rows:
        cleaned_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
        # Pad to ncols
        cleaned_row += [""] * (ncols - len(cleaned_row))
        cleaned.append(cleaned_row)

    lines = []
    # Header
    lines.append("| " + " | ".join(cleaned[0]) + " |")
    # Separator
    lines.append("|" + "|".join(["---"] * ncols) + "|")
    # Data rows
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ─── Tier 2: Spatial block alignment fallback ─────────────────────────────────

def _spatial_block_table_fallback(page: fitz.Page) -> str:
    """
    Spatial column detection fallback for pages where find_tables() finds nothing.

    Groups text spans into column clusters by their X position, then reconstructs
    rows from Y proximity. Only returns a non-empty string if clear multi-column
    structure is detected (≥2 columns with X gap > 50pt), preventing false positives
    on flowing text.

    Args:
        page: An open fitz.Page object.

    Returns:
        Markdown string (possibly empty).
    """
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return ""

    text_spans = []
    for block in blocks:
        if block.get("type") != 0:  # type 0 = text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                if txt:
                    x0 = span["bbox"][0]
                    y0 = span["bbox"][1]
                    text_spans.append((x0, y0, txt))

    if len(text_spans) < 6:
        return ""

    # Quantise X positions to 15pt grid and sort
    x_buckets = sorted(set(round(x / 15) * 15 for x, _, _ in text_spans))
    if len(x_buckets) < 2:
        return ""

    # Compute gaps between consecutive X buckets
    gaps = [x_buckets[i + 1] - x_buckets[i] for i in range(len(x_buckets) - 1)]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    # Only continue if there are clear wide column gaps (table-like structure)
    if avg_gap < 50:
        return ""

    # Detect column boundaries at large gaps
    col_starts = [x_buckets[0]]
    for i, gap in enumerate(gaps):
        if gap >= avg_gap * 1.3:
            col_starts.append(x_buckets[i + 1])

    if len(col_starts) < 2:
        return ""

    # Assign each span to the nearest column
    def nearest_col(x):
        return min(range(len(col_starts)), key=lambda i: abs(col_starts[i] - x))

    # Group spans by row (Y quantised to 8pt grid)
    row_map: dict[int, dict[int, list[str]]] = {}
    for x, y, txt in text_spans:
        row_key = round(y / 8) * 8
        col_idx = nearest_col(x)
        row_map.setdefault(row_key, {}).setdefault(col_idx, []).append(txt)

    if len(row_map) < 2:
        return ""

    # Reconstruct rows sorted by Y
    ncols = len(col_starts)
    sorted_rows = []
    for row_key in sorted(row_map.keys()):
        cells = row_map[row_key]
        row = [" ".join(cells.get(c, [])) for c in range(ncols)]
        sorted_rows.append(row)

    return _rows_to_markdown(sorted_rows)


# ─── Table injection into page-marked text ────────────────────────────────────

def inject_markdown_tables(pdf_path: str, base_text: str,
                            max_pages: int = None) -> str:
    """
    Re-open a PDF and inject Markdown table blocks into a page-marked text string.

    For each page that has extractable tables, appends a
    ``<!-- TABLES PAGE N -->`` block immediately after the page's text block.
    The ``<!-- END TABLES -->`` marker closes the block. This tagging allows the
    downstream chunker to identify and preserve table content.

    Args:
        pdf_path:   Absolute path to the PDF file.
        base_text:  Existing extracted text with ``--- PAGE N ---`` markers.
        max_pages:  Optional limit on how many pages to process.

    Returns:
        Modified text string with table blocks injected, or base_text unchanged
        if no tables are found or the PDF cannot be opened.
    """
    try:
        doc = fitz.open(pdf_path)
        limit = min(max_pages or len(doc), len(doc))

        # Build page → Markdown tables map
        table_map: dict[int, str] = {}
        for i in range(limit):
            md = extract_page_tables_as_markdown(doc[i])
            if md:
                table_map[i + 1] = md   # page numbers are 1-indexed
        doc.close()

        if not table_map:
            return base_text

        # Inject table blocks after each --- PAGE N --- section
        result_lines: list[str] = []
        current_page: int | None = None
        injected: set[int] = set()

        for line in base_text.split("\n"):
            result_lines.append(line)
            page_match = re.match(r"--- PAGE (\d+)", line)
            if page_match:
                current_page = int(page_match.group(1))

            # Inject after first blank line following a page header
            elif (
                line == ""
                and current_page is not None
                and current_page in table_map
                and current_page not in injected
            ):
                result_lines.append(
                    f"\n<!-- TABLES PAGE {current_page} -->\n"
                    f"{table_map[current_page]}\n"
                    f"<!-- END TABLES -->"
                )
                injected.add(current_page)

        return "\n".join(result_lines)

    except Exception as e:
        print(f"  [TableExtractor] inject_markdown_tables failed for {pdf_path}: {e}")
        return base_text
