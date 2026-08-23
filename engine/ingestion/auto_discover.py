"""
Auto-Discovery Pipeline — Agnostic RAG Doll.

Analyzes a directory of PDFs and produces a game_profile.json that drives
the generic ingestion and agent pipelines. No OCR — native text only.

Output profile schema:
    game_name, game_id, data_dir, text_dir, chroma_collection,
    rule_index_file, rule_schema, rule_pattern, cross_ref_pattern,
    scenario_format, scenario_pattern, documents{}

Usage:
    python auto_discover.py data/asl "Advanced Squad Leader"
    python auto_discover.py data/upfront "Up Front"       (re-generates UF profile)
"""

import os
import re
import json
import fitz   # PyMuPDF
import ollama

# ═══════════════════════════════════════════════════════════════════
# Known Rule Schemas
# ═══════════════════════════════════════════════════════════════════

KNOWN_SCHEMAS = [
    {
        "name": "chapter_decimal",
        "description": "Chapter-letter + decimal (ASL: A7.212, D5.6)",
        # Matches on a word boundary so e.g. "A7.212" is found in body text
        "rule_pattern": r"(?:^|\s)([A-Z]\d{1,2}\.\d{1,4})\b",
        "cross_ref_pattern": (
            r"\(([A-Z]\d{1,2}\.\d{1,4})\)|"
            r"\[EXC:\s*([A-Z]\d{1,2}\.\d{1,4})[^\]]*\]|"
            r"(?:see|See|SEE)\s+([A-Z]\d{1,2}\.\d{1,4})"
        ),
    },
    {
        "name": "numeric_decimal",
        "description": "Numeric decimal anchored to line-start (Up Front: 5.41, 17.4)",
        "rule_pattern": r"(?:^|\n)\s*(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\b",
        "cross_ref_pattern": (
            r"\[(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\]|"
            r"(?:see|See|SEE)\s+(?:\[)?(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)(?:\])?|"
            r"(?:rule|Rule|RULE)\s+(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)|"
            r"EXC:\s*\[(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\]"
        ),
    },
    {
        "name": "outline_parenthetical",
        "description": "Parenthetical outline (SFB: (D2.31), (C3.1))",
        "rule_pattern": r"\(([A-Z]\d{1,2}\.\d{1,3})\)",
        "cross_ref_pattern": r"\(([A-Z]\d{1,2}\.\d{1,3})\)",
    },
    {
        "name": "keyword_header",
        "description": "Visual bold headers instead of numbers (Warhammer 40K)",
        "rule_pattern": r"(?:^|\n)([A-Z\s]{4,})\n",
        "cross_ref_pattern": r"(?:see|See|SEE)\s+(?:page\s+\d+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    },
]


# ═══════════════════════════════════════════════════════════════════
# Scenario Format Signatures
# ═══════════════════════════════════════════════════════════════════

SCENARIO_FORMATS = [
    {
        "name": "letter",
        "description": "Lettered scenarios A-Z (Up Front: Scenario A, B, C...)",
        "pattern": r"(?:^|\n)\s*([A-K])[.:\s]+[A-Z]",
        "detect": r"Scenario\s+[A-K]\b|^[A-K][.:]\s+[A-Z]",
    },
    {
        "name": "numeric",
        "description": "Numbered scenarios (ASL: Scenario 1, ASL 14...)",
        "pattern": r"(?:^|\n)\s*(?:Scenario\s+|ASL\s+)?(\d+)[.:\s]+",
        "detect": r"ASL\s+\d+|Scenario\s+\d+|^(?:Scenario\s+)?\d+[.:]",
    },
    {
        "name": "named",
        "description": "Title-named scenarios (no numbering scheme)",
        "pattern": r"(?:^|\n)\s*([A-Z][A-Z\s]+)\n",
        "detect": r"SCENARIO\s+DESIGN|Special\s+Scenario\s+Rules|SSR",
    },
]


# ═══════════════════════════════════════════════════════════════════
# Document Type Heuristics
# ═══════════════════════════════════════════════════════════════════

# Filename pattern → (doc_type, priority)
FILENAME_HEURISTICS = [
    (r"errata",              "errata",           3),
    (r"hasl.*errata",        "errata",           3),
    (r"scenario.*errata",    "scenario_errata",  4),
    (r"scenario.*balance",   "scenario_balance", 5),
    (r"version.*tracker",    "version_tracker",  2),
    (r"qa|q&a|q_a",         "qa",               6),
    (r"journal",             "journal",          6),
    (r"variant|house.*rule", "variant",          7),
    (r"tournament",          "tournament",       7),
    (r"codex",               "codex",            2),
    (r"primer|intro|what.is","primer",           9),
    # Scenario files
    (r"scenario[a-z]$",      "scenarios",        8),
    (r"scenario[a-z0-9]+$",  "scenarios",        8),
    (r"^ap\d+",              "scenarios",        8),
    (r"^pp\d+",              "scenarios",        8),
    (r"^scenario[ta-z]\d*$", "scenarios",        8),
    (r"scenarios.*\d",       "scenarios",        8),
    (r"1st.*edition|1st.*ed","core_rules_v1",    9),
    (r"2nd.*edition|2nd.*ed","core_rules",       1),
    (r"replacement.*page",   "errata",           3),
    (r"vb$",                 "errata",           3),   # e.g. A15-A16vB
]

# Title/content keywords → (doc_type, priority)
CONTENT_HEURISTICS = [
    (r"published errata",                "errata",           3),
    (r"balance errata",                  "scenario_balance", 5),
    (r"scenario.*errata",               "scenario_errata",  4),
    (r"questions and answers",           "qa",               6),
    (r"clarifications and errata",       "qa",               6),
    (r"version tracker",                 "version_tracker",  2),
    (r"2nd edition.*rules|rules.*2nd",  "core_rules",       1),
    (r"1st edition.*rules|rules.*1st",  "core_rules_v1",    9),
    (r"introduction to the (1st|2nd)",  "core_rules",       1),
    (r"asl scenario\s+\w+\d+",          "scenarios",        8),
    (r"turn record chart",               "scenarios",        8),
    (r"asl journal",                     "journal",          6),
    (r"codex",                           "codex",            2),
    (r"variant|house rule|optional rule","variant",          7),
    (r"replacement page",               "errata",           3),
]

# doc_type priority defaults (fallback when not already set)
PRIORITY_DEFAULTS = {
    "core_rules":     1,
    "codex":          2,
    "version_tracker":2,
    "errata":         3,
    "scenario_errata":4,
    "scenario_balance":5,
    "qa":             6,
    "journal":        6,
    "scenarios":      8,
    "tournament":     7,
    "variant":        7,
    "core_rules_v1":  9,
    "primer":         9,
    "supplement":     8,
    "unknown":        9,
}


# ═══════════════════════════════════════════════════════════════════
# PDF Sampling
# ═══════════════════════════════════════════════════════════════════

def sample_pdf_text(pdf_path, page_indices=None, max_chars_per_page=2000):
    """
    Extract text from specified pages of a PDF using native extraction.
    Returns list of (page_idx, text) tuples.
    """
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        if page_indices is None:
            # Sample front, middle, and near-end
            candidates = [0, 1, 2, 3, min(10, total_pages - 1),
                          total_pages // 4, total_pages // 2]
            page_indices = sorted(set(p for p in candidates if 0 <= p < total_pages))

        results = []
        for idx in page_indices:
            if 0 <= idx < total_pages:
                text = doc[idx].get_text().strip()
                results.append((idx, text[:max_chars_per_page]))

        doc.close()
        return results
    except Exception as e:
        print(f"  Warning: could not sample {os.path.basename(pdf_path)}: {e}")
        return []


def get_title_text(pdf_path, max_chars=1500):
    """Extract text from the first usable page of a PDF."""
    samples = sample_pdf_text(pdf_path, page_indices=[0, 1, 2, 3])
    for _, text in samples:
        if len(text) > 50:
            return text[:max_chars]
    return ""


# ═══════════════════════════════════════════════════════════════════
# Rule Schema Detection
# ═══════════════════════════════════════════════════════════════════

def detect_rule_schema(sampled_texts):
    """
    Run each known schema's pattern against sampled text.
    Returns the schema dict with highest hit count, plus all scores.
    """
    combined_text = "\n".join(sampled_texts)
    scores = {}

    for schema in KNOWN_SCHEMAS:
        try:
            pattern = re.compile(schema["rule_pattern"], re.MULTILINE)
            matches = pattern.findall(combined_text)
            scores[schema["name"]] = len(matches)
        except re.error:
            scores[schema["name"]] = 0

    best_name = max(scores, key=scores.get)
    best_schema = next(s for s in KNOWN_SCHEMAS if s["name"] == best_name)

    return best_schema, scores


# ═══════════════════════════════════════════════════════════════════
# Document Classification
# ═══════════════════════════════════════════════════════════════════

def classify_by_filename(filename):
    """
    Classify a document by its filename using heuristics.
    Returns (doc_type, priority, edition) or None.
    """
    fname_lower = filename.lower().replace(" ", "_").replace("-", "_")
    # Remove extension
    fname_stem = os.path.splitext(fname_lower)[0]
    
    # Try to extract edition number
    edition = None
    ed_match = re.search(r'(\d+)(?:st|nd|rd|th|e)?\s*(?:edition|ed|e\b)', fname_stem)
    if not ed_match:
        # e.g. "new40k" or "9e" or "10e"
        ed_match = re.search(r'(\d+)(?:e)\b', fname_stem)
    
    if ed_match:
        edition = int(ed_match.group(1))

    for pattern, doc_type, priority in FILENAME_HEURISTICS:
        if re.search(pattern, fname_stem):
            return doc_type, priority, edition

    return None, None, edition


def classify_by_content(title_text):
    """
    Classify a document by its content using keyword heuristics.
    Returns (doc_type, priority, edition) or None.
    """
    text_lower = title_text.lower()
    
    edition = None
    ed_match = re.search(r'(\d+)(?:st|nd|rd|th|e)?\s*(?:edition|ed)', text_lower)
    if ed_match:
        edition = int(ed_match.group(1))
        
    for pattern, doc_type, priority in CONTENT_HEURISTICS:
        if re.search(pattern, text_lower):
            return doc_type, priority, edition

    return None, None, edition


def classify_by_llm(filename, title_text, game_name):
    """
    Use llama3.1:8b to classify a document when heuristics fail.
    Returns (doc_type, priority) or falls back to ("unknown", 9).
    """
    prompt = f"""You are classifying a document in a wargame rules RAG system for "{game_name}".

Filename: {filename}
First page content:
{title_text[:800]}

Classify this document as ONE of these types:
- core_rules: The main/primary rulebook
- errata: Official corrections and updates to rules
- scenario_errata: Corrections specific to scenarios
- scenario_balance: Balance adjustments for scenarios
- version_tracker: Change log between rulebook editions
- qa: Q&A / clarifications document
- journal: Magazine or journal containing rules/scenarios
- scenarios: Scenario cards or packs
- variant: Variant or house rules
- tournament: Tournament-specific rules
- supplement: Supplementary rules module
- unknown: Cannot be determined

Respond with ONLY a JSON object (no explanation):
{{"doc_type": "<type>", "priority": <1-9>, "description": "<brief one-line description>"}}

Priority guide: core_rules=1, version_tracker=2, errata=3, scenario_errata=4, scenario_balance=5, qa=6, journal=6, scenarios=7-8, variant=7, tournament=7, unknown=9"""

    try:
        response = ollama.generate(model="llama3.1:8b", prompt=prompt)
        raw = response["response"].strip()
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            doc_type = result.get("doc_type", "unknown")
            priority = int(result.get("priority", PRIORITY_DEFAULTS.get(doc_type, 9)))
            description = result.get("description", "")
            return doc_type, priority, description
    except Exception as e:
        print(f"    LLM classification failed: {e}")

    return "unknown", 9, ""


def classify_document(filename, title_text, game_name, use_llm=True):
    """
    Classify a document, trying heuristics first then LLM fallback.
    Returns dict with doc_type, priority, description.
    """
    # 1. Try filename heuristics
    doc_type, priority, edition_file = classify_by_filename(filename)

    # 2. Try content heuristics
    edition_content = None
    if not doc_type:
        doc_type, priority, edition_content = classify_by_content(title_text)

    edition = edition_file or edition_content

    # 3. LLM fallback
    description = ""
    if not doc_type and use_llm and title_text:
        doc_type, priority, description = classify_by_llm(filename, title_text, game_name)

    # 4. Final fallback
    if not doc_type:
        doc_type = "unknown"
        priority = 9

    if not priority:
        priority = PRIORITY_DEFAULTS.get(doc_type, 9)

    # Build a default description if none
    if not description:
        description = f"{doc_type.replace('_', ' ').title()} document"

    return {
        "doc_type": doc_type,
        "priority": priority,
        "description": description,
        "edition": edition
    }


# ═══════════════════════════════════════════════════════════════════
# Scenario Format Detection
# ═══════════════════════════════════════════════════════════════════

def detect_scenario_format(sampled_texts):
    """
    Detect the scenario identification format used in this game's documents.
    Returns (format_name, pattern_string, detection_confidence).
    """
    combined = "\n".join(sampled_texts)
    scores = {}

    for fmt in SCENARIO_FORMATS:
        try:
            detect_re = re.compile(fmt["detect"], re.IGNORECASE | re.MULTILINE)
            hits = len(detect_re.findall(combined))
            scores[fmt["name"]] = hits
        except re.error:
            scores[fmt["name"]] = 0

    best_name = max(scores, key=scores.get)
    best_fmt = next(f for f in SCENARIO_FORMATS if f["name"] == best_name)

    # Confidence: 0-1 based on relative dominance
    total = sum(scores.values()) or 1
    confidence = scores[best_name] / total

    return best_fmt["name"], best_fmt["pattern"], confidence, scores


# ═══════════════════════════════════════════════════════════════════
# Max Pages Heuristic
# ═══════════════════════════════════════════════════════════════════

def suggest_max_pages(filename, file_size_mb, total_pages, doc_type):
    """
    Suggest a max_pages limit for ingestion to protect against massive files.
    Returns None (no limit) for small files, or an integer for large ones.
    """
    # Skip files over 50MB that are lower-priority (old editions, journals)
    if file_size_mb > 50 and doc_type in ("core_rules_v1", "journal", "unknown"):
        return 0  # 0 = skip entirely

    # Cap very large core rulebooks at 200 pages (covers chapters A-E)
    if file_size_mb > 100 and doc_type == "core_rules":
        return 200

    # Cap large scenario compilations
    if file_size_mb > 15 and doc_type == "scenarios":
        return 80

    # Cap large supplementary docs
    if file_size_mb > 15 and doc_type in ("journal", "version_tracker", "supplement"):
        return 50

    return None  # No limit


def extract_glossary_hybrid(pdf_path):
    """Extract a glossary from the rulebook using a hybrid Regex/LLM approach."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + " "
        doc.close()
    except Exception as e:
        print(f"  Warning: could not read {os.path.basename(pdf_path)} for glossary: {e}")
        return {}

    # Common acronym definition patterns
    patterns = [
        r'([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,4})\s*\(([A-Z]{2,5}[a-z]?)\)',
        r'([A-Z]{2,5}[a-z]?)\s*\(([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,4})\)',
        r'([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,4})\s*\[([A-Z]{2,5}[a-z]?)\]',
        r'\b([A-Z]{2,5}[a-z]?):\s*([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,4})'
    ]
    
    candidates = set()
    for pat_str in patterns:
        pat = re.compile(pat_str)
        for match in pat.finditer(text):
            g1, g2 = match.groups()
            g1 = g1.strip()
            g2 = g2.strip()
            if len(g1) <= 6 and g1.isupper() and len(g2) > len(g1):
                candidates.add(f"{g1}: {g2}")
            elif len(g2) <= 6 and g2.isupper() and len(g1) > len(g2):
                candidates.add(f"{g2}: {g1}")

    if not candidates:
        return {}

    cand_list = sorted(list(candidates))
    cand_text = "\n".join(cand_list)
    
    prompt = f'''You are a wargaming expert. Here is a list of candidate acronym definitions extracted from a rulebook via regex.
Many of them are garbage or false positives. Filter this list and return ONLY a valid JSON dictionary of the true game-specific abbreviations and their meanings.
Return ONLY JSON, no markdown formatting or other text.

Candidates:
{cand_text}
'''
    try:
        import ollama
        import json
        import ast
        res = ollama.generate(model='llama3.1:8b', prompt=prompt)
        raw = res['response'].strip()
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # LLMs sometimes use single quotes; literal_eval can parse python dict strings
                return ast.literal_eval(json_str)
        else:
            return {}
    except Exception as e:
        print(f"  Warning: LLM glossary extraction failed: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════
# Main Discovery Function
# ═══════════════════════════════════════════════════════════════════

def discover(data_dir, game_name, game_id=None, use_llm=True, output_path=None):
    """
    Analyze a directory of PDFs and produce a game profile.

    Args:
        data_dir:    Path to directory containing game PDFs.
        game_name:   Human-readable game name (e.g., "Advanced Squad Leader").
        game_id:     Short ID for paths (e.g., "asl"). Auto-derived if None.
        use_llm:     Use LLM for ambiguous document classification.
        output_path: Where to save the profile JSON. Auto-derived if None.

    Returns:
        profile dict
    """
    if not game_id:
        game_id = re.sub(r'[^a-z0-9]', '_', game_name.lower())[:20].strip('_')

    if not output_path:
        output_path = f"data/{game_id}_profile.json"

    print("=" * 60)
    print(f"AUTO-DISCOVERY: {game_name}")
    print(f"Directory: {data_dir}")
    print("=" * 60)

    pdf_files = sorted([
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print(f"ERROR: No PDF files found in {data_dir}")
        return None

    print(f"\nFound {len(pdf_files)} PDF files")

    # ─── Phase 1: Sample texts for schema detection ───
    print("\n[1/4] Sampling documents for rule schema detection...")
    all_sampled = []
    doc_samples = {}  # filename → title_text

    for fname in pdf_files:
        fpath = os.path.join(data_dir, fname)
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        samples = sample_pdf_text(fpath, page_indices=[0, 1, 5, 10, 20])
        texts = [text for _, text in samples if len(text) > 30]
        doc_samples[fname] = texts[0] if texts else ""
        all_sampled.extend(texts)

    # ─── Phase 2: Detect rule schema ───
    print("\n[2/4] Detecting rule numbering schema...")
    schema, schema_scores = detect_rule_schema(all_sampled)
    print(f"  Schema scores: {schema_scores}")
    print(f"  Selected: {schema['name']} — {schema['description']}")

    # ─── Phase 3: Classify each document ───
    print(f"\n[3/4] Classifying {len(pdf_files)} documents...")
    documents = {}

    for fname in pdf_files:
        fpath = os.path.join(data_dir, fname)
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        try:
            doc = fitz.open(fpath)
            total_pages = len(doc)
            doc.close()
        except Exception:
            total_pages = 0

        title_text = doc_samples.get(fname, "")
        classification = classify_document(fname, title_text, game_name, use_llm=use_llm)
        doc_type = classification["doc_type"]

        max_pg = suggest_max_pages(fname, size_mb, total_pages, doc_type)

        documents[fname] = {
            "doc_type": doc_type,
            "priority": classification["priority"],
            "description": classification["description"],
            "size_mb": round(size_mb, 2),
            "total_pages": total_pages,
            "max_pages": max_pg,
        }

        skip_note = f" [SKIP: {max_pg}p limit]" if max_pg == 0 else (
            f" [cap: {max_pg}p]" if max_pg else "")
        print(f"  [{classification['priority']}] {fname[:50]}  -> {doc_type}{skip_note}")

    # ─── Phase 4: Detect scenario format ───
    print("\n[4/4] Detecting scenario format...")
    scenario_texts = [
        doc_samples.get(f, "") for f in pdf_files
        if "scenario" in f.lower() or f.startswith(("ap", "pp", "Scenario"))
    ]
    scenario_fmt, scenario_pattern, scenario_confidence, fmt_scores = detect_scenario_format(
        scenario_texts + all_sampled[:10]
    )
    print(f"  Format scores: {fmt_scores}")
    print(f"  Selected: {scenario_fmt} (confidence: {scenario_confidence:.0%})")

    # ─── Phase 5: Glossary Extraction ───
    print("\n[5/5] Extracting Glossary (Hybrid Regex-to-LLM)...")
    core_rules_path = None
    for fname, info in documents.items():
        if info["doc_type"] == "core_rules":
            core_rules_path = os.path.join(data_dir, fname)
            break
            
    glossary = {}
    if core_rules_path and use_llm:
        glossary = extract_glossary_hybrid(core_rules_path)
        print(f"  Extracted {len(glossary)} acronyms.")

    # ─── Build profile ───
    profile = {
        "game_name": game_name,
        "game_id": game_id,
        "data_dir": data_dir,
        "text_dir": f"data/{game_id}_text",
        "chroma_collection": f"{game_id}-rules-semantic",
        "rule_index_file": f"data/{game_id}_rule_index.json",
        "cooccurrence_graph_file": f"data/{game_id}_cooccurrence_graph.json",
        "section_tree_file": f"data/{game_id}_section_tree.json",
        "rule_schema": schema["name"],
        "rule_pattern": schema["rule_pattern"],
        "cross_ref_pattern": schema["cross_ref_pattern"],
        "scenario_format": scenario_fmt,
        "scenario_pattern": scenario_pattern,
        "schema_detection_scores": schema_scores,
        "documents": documents,
        "glossary": glossary,
    }

    # ─── Save profile ───
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"PROFILE SAVED: {output_path}")
    print(f"  Rule schema:     {schema['name']}")
    print(f"  Scenario format: {scenario_fmt}")
    print(f"  Documents:       {len(documents)}")
    active = sum(1 for d in documents.values() if d.get("max_pages") != 0)
    print(f"  Active (non-skip): {active}")
    print(f"{'=' * 60}")

    return profile


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python auto_discover.py <data_dir> <game_name> [game_id]")
        print("  e.g: python auto_discover.py data/asl \"Advanced Squad Leader\" asl")
        sys.exit(1)

    data_dir = sys.argv[1]
    game_name = sys.argv[2]
    game_id = sys.argv[3] if len(sys.argv) > 3 else None
    no_llm = "--no-llm" in sys.argv

    if not os.path.isdir(data_dir):
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)

    profile = discover(data_dir, game_name, game_id=game_id, use_llm=not no_llm)
