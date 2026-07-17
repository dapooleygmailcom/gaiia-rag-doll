"""
Rules Lawyer Ingestion Pipeline.

Classifies, chunks, and indexes Up Front wargame rule documents into ChromaDB
with a parallel JSON rule-number lookup index.

Document types are prioritized for temporal supersession handling.
Chunking is rule-number-aware, preserving Q&A blocks and cross-references.
"""

import os
import re
import json
import fitz  # PyMuPDF
import chromadb
import ollama

DATA_DIR = "data/upfront"
TEXT_DIR = "data/upfront_text"
CHROMA_DB_DIR = "data/chroma"
CHROMA_COLLECTION = "upfront-rules-semantic"
RULE_INDEX_FILE = "data/upfront_rule_index.json"

# ═══════════════════════════════════════════════════════════════════
# Document Classification
# ═══════════════════════════════════════════════════════════════════

DOCUMENT_CLASSIFICATION = {
    "UF RuleBook updated.pdf": {
        "doc_type": "integrated_rules",
        "priority": 1,
        "description": "Definitive merged reference — core rules interleaved with Q&A/errata"
    },
    "Up_Front.pdf": {
        "doc_type": "core_rules",
        "priority": 2,
        "description": "Original 1983 core rules"
    },
    "Up_Front_Errata_Pages.pdf": {
        "doc_type": "errata",
        "priority": 3,
        "description": "Standalone errata organized by rule number"
    },
    "UF_Scenarios_2-4.pdf": {
        "doc_type": "scenario_errata",
        "priority": 4,
        "description": "Scenario pack Q&A and errata"
    },
    "UF_Scenarios_2-5.pdf": {
        "doc_type": "scenario_errata",
        "priority": 4,
        "description": "Scenario pack update"
    },
    "Upfront_Scenarios_1.pdf": {
        "doc_type": "scenarios",
        "priority": 5,
        "description": "Scenarios A-J with setup, OOBs, victory conditions"
    },
    "Up_Front-_Experimental_house_and_variant_rules.pdf": {
        "doc_type": "variant",
        "priority": 6,
        "description": "Variant/house rules (non-official)"
    },
    "origins_87_tournament_rules.pdf": {
        "doc_type": "tournament",
        "priority": 7,
        "description": "Origins 1987 tournament rules"
    },
    "What_is_UpFront.pdf": {
        "doc_type": "primer",
        "priority": 8,
        "description": "Beginner overview"
    },
    # Potential duplicate — will be skipped if flagged by OCR processor
    "up-front-rules.pdf": {
        "doc_type": "core_rules",
        "priority": 2,
        "description": "Scanned core rules (may be duplicate of Up_Front.pdf)"
    }
}

# ═══════════════════════════════════════════════════════════════════
# Rule Number Parsing
# ═══════════════════════════════════════════════════════════════════

# Matches rule numbers like: 5.41, 17.4, 6.5, 11.12, 3.3, 29.5
# At the start of a line or after specific prefixes
RULE_NUMBER_PATTERN = re.compile(
    r'(?:^|\n)\s*(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\b',
    re.MULTILINE
)

# Matches cross-references like: [5.61], EXC: [15.2], see 11.12, rule 3.3
CROSS_REF_PATTERN = re.compile(
    r'\[(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\]|'
    r'(?:see|See|SEE)\s+(?:\[)?(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)(?:\])?|'
    r'(?:rule|Rule|RULE)\s+(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)|'
    r'EXC:\s*\[(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\]',
    re.MULTILINE
)

# Section header pattern (e.g., "5. MOVEMENT & RANGE DETERMINATION")
SECTION_HEADER_PATTERN = re.compile(
    r'(?:^|\n)\s*(\d{1,2})\.\s+([A-Z][A-Z\s&\-]+)',
    re.MULTILINE
)

# Q&A block detection
QA_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:Q\.|A\.|CLARIFICATION:|VARIANT:)',
    re.MULTILINE
)

# Scenario letter pattern
SCENARIO_PATTERN = re.compile(
    r'(?:^|\n)\s*([A-K])[\.:]\s+',
    re.MULTILINE
)

# Separator used in the integrated rulebook
ERRATA_SEPARATOR = re.compile(r'[•·]\.{4,}')


def extract_rule_numbers(text):
    """Extract all rule numbers from a chunk of text."""
    matches = RULE_NUMBER_PATTERN.findall(text)
    return list(set(matches))


def extract_cross_references(text):
    """Extract all cross-referenced rule numbers from text."""
    refs = set()
    for match in CROSS_REF_PATTERN.finditer(text):
        # Each group captures a different pattern variant
        for group in match.groups():
            if group:
                refs.add(group)
    return list(refs)


def detect_content_type(text):
    """Classify the content type of a text chunk."""
    text_upper = text.strip().upper()
    
    if QA_PATTERN.search(text):
        if "CLARIFICATION:" in text.upper():
            return "clarification"
        if "VARIANT:" in text.upper():
            return "variant"
        return "qa"
    
    if SCENARIO_PATTERN.match(text):
        return "scenario_rule"
    
    return "rule"


def detect_scenario(text):
    """Detect if a chunk is specific to a scenario (A-K)."""
    # Look for scenario identifiers
    scenario_matches = re.findall(
        r'(?:^|\n)\s*([A-K])[\.:]\s+|'
        r'[Ss]cenario\s+([A-K])\b',
        text
    )
    scenarios = set()
    for match in scenario_matches:
        for group in match:
            if group:
                scenarios.add(group)
    return sorted(scenarios) if scenarios else None


def build_section_path(section_num, section_name, subsection_num=None):
    """Build a hierarchical section path string."""
    path = f"{section_num}. {section_name.strip()}"
    if subsection_num:
        path += f" > {subsection_num}"
    return path


# ═══════════════════════════════════════════════════════════════════
# Chunking Strategies
# ═══════════════════════════════════════════════════════════════════

def chunk_integrated_rules(text, source_file):
    """
    Chunk the integrated rulebook (UF RuleBook updated.pdf).
    
    This document interleaves original rules with inline errata (marked with •............).
    We chunk by rule number, keeping Q&A blocks atomic.
    """
    chunks = []
    
    # Split by rule number boundaries
    # Look for lines that start with a rule number pattern
    lines = text.split('\n')
    
    current_section = "Introduction"
    current_subsection = ""
    current_rule = None
    accumulated_lines = []
    current_page = 1
    
    for line in lines:
        # Track page markers from OCR/text output
        page_match = re.match(r'--- PAGE (\d+)', line)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        
        # Detect section headers (e.g., "5. MOVEMENT & RANGE DETERMINATION")
        section_match = SECTION_HEADER_PATTERN.match(line)
        if section_match:
            # Save current accumulated chunk
            if accumulated_lines and len(' '.join(accumulated_lines)) >= 100:
                chunk = _build_chunk(
                    accumulated_lines, source_file, "integrated_rules", 1,
                    current_section, current_subsection, current_rule,
                    current_page
                )
                chunks.append(chunk)
                accumulated_lines = []
            
            current_section = f"{section_match.group(1)}. {section_match.group(2).strip()}"
            current_subsection = ""
            current_rule = section_match.group(1) + ".0"
        
        # Detect rule number at start of line
        rule_match = re.match(r'^\s*(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\s', line)
        if rule_match:
            new_rule = rule_match.group(1)
            
            # Save previous rule's accumulated text
            if accumulated_lines and len(' '.join(accumulated_lines)) >= 80:
                chunk = _build_chunk(
                    accumulated_lines, source_file, "integrated_rules", 1,
                    current_section, current_subsection, current_rule,
                    current_page
                )
                chunks.append(chunk)
                accumulated_lines = []
            
            current_rule = new_rule
            current_subsection = new_rule
        
        # Skip empty lines but accumulate content
        if line.strip():
            accumulated_lines.append(line.strip())
        
        # Split if chunk is getting too large (but respect Q&A atomicity)
        joined = ' '.join(accumulated_lines)
        if len(joined) > 3000:
            # Try to split at a natural boundary (Q&A or errata separator)
            split_point = _find_split_point(accumulated_lines)
            if split_point > 0:
                chunk = _build_chunk(
                    accumulated_lines[:split_point], source_file, "integrated_rules", 1,
                    current_section, current_subsection, current_rule,
                    current_page
                )
                chunks.append(chunk)
                # Keep overlap
                accumulated_lines = accumulated_lines[max(0, split_point - 2):]
            else:
                # Force split
                chunk = _build_chunk(
                    accumulated_lines, source_file, "integrated_rules", 1,
                    current_section, current_subsection, current_rule,
                    current_page
                )
                chunks.append(chunk)
                accumulated_lines = accumulated_lines[-2:]
    
    # Final chunk
    if accumulated_lines and len(' '.join(accumulated_lines)) >= 50:
        chunk = _build_chunk(
            accumulated_lines, source_file, "integrated_rules", 1,
            current_section, current_subsection, current_rule,
            current_page
        )
        chunks.append(chunk)
    
    return chunks


def chunk_errata(text, source_file):
    """
    Chunk the standalone errata document.
    
    Errata is organized by rule number with Q&A format.
    Each rule number + its Q&A block becomes one chunk.
    """
    chunks = []
    lines = text.split('\n')
    
    current_rule = None
    accumulated_lines = []
    current_page = 1
    
    for line in lines:
        page_match = re.match(r'--- PAGE (\d+)', line)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        
        # Detect rule number headers in errata
        rule_match = re.match(r'^\s*\*?(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\s', line)
        if rule_match:
            new_rule = rule_match.group(1)
            
            if accumulated_lines and current_rule:
                chunk = _build_chunk(
                    accumulated_lines, source_file, "errata", 3,
                    f"Errata for Rule {current_rule}", current_rule, current_rule,
                    current_page
                )
                chunks.append(chunk)
                accumulated_lines = []
            
            current_rule = new_rule
        
        if line.strip():
            accumulated_lines.append(line.strip())
        
        # Errata Q&A blocks can be long — split at 2500 chars
        if len(' '.join(accumulated_lines)) > 2500:
            chunk = _build_chunk(
                accumulated_lines, source_file, "errata", 3,
                f"Errata for Rule {current_rule}", current_rule, current_rule,
                current_page
            )
            chunks.append(chunk)
            accumulated_lines = accumulated_lines[-2:]
    
    if accumulated_lines:
        chunk = _build_chunk(
            accumulated_lines, source_file, "errata", 3,
            f"Errata for Rule {current_rule}" if current_rule else "Errata",
            current_rule, current_rule, current_page
        )
        chunks.append(chunk)
    
    return chunks


def chunk_scenarios(text, source_file):
    """
    Chunk scenario documents.
    
    Each scenario (A-K) with its setup, special rules, and victory conditions
    becomes one or more chunks.
    """
    chunks = []
    lines = text.split('\n')
    
    current_scenario = None
    current_section = "Scenarios"
    accumulated_lines = []
    current_page = 1
    
    for line in lines:
        page_match = re.match(r'--- PAGE (\d+)', line)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        
        # Detect scenario headers (e.g., "A: MEETING OF PATROLS" or "C: ASSAULTING A FORTIFICATION")
        scenario_match = re.match(r'^\s*([A-K])[\.:]\s*(.*)', line)
        if scenario_match:
            if accumulated_lines and current_scenario:
                chunk = _build_chunk(
                    accumulated_lines, source_file,
                    DOCUMENT_CLASSIFICATION.get(source_file, {}).get("doc_type", "scenarios"),
                    DOCUMENT_CLASSIFICATION.get(source_file, {}).get("priority", 5),
                    f"Scenario {current_scenario}", current_scenario, None,
                    current_page, scenario=current_scenario
                )
                chunks.append(chunk)
                accumulated_lines = []
            
            current_scenario = scenario_match.group(1)
            current_section = f"Scenario {current_scenario}: {scenario_match.group(2).strip()}"
        
        if line.strip():
            accumulated_lines.append(line.strip())
        
        if len(' '.join(accumulated_lines)) > 2500:
            chunk = _build_chunk(
                accumulated_lines, source_file,
                DOCUMENT_CLASSIFICATION.get(source_file, {}).get("doc_type", "scenarios"),
                DOCUMENT_CLASSIFICATION.get(source_file, {}).get("priority", 5),
                current_section, current_scenario, None,
                current_page, scenario=current_scenario
            )
            chunks.append(chunk)
            accumulated_lines = accumulated_lines[-2:]
    
    if accumulated_lines:
        chunk = _build_chunk(
            accumulated_lines, source_file,
            DOCUMENT_CLASSIFICATION.get(source_file, {}).get("doc_type", "scenarios"),
            DOCUMENT_CLASSIFICATION.get(source_file, {}).get("priority", 5),
            current_section, current_scenario, None,
            current_page, scenario=current_scenario
        )
        chunks.append(chunk)
    
    return chunks


def chunk_generic(text, source_file):
    """
    Generic chunker for documents without specialized structure
    (variant rules, tournament rules, primer).
    
    Uses section-boundary detection and size limits.
    """
    doc_info = DOCUMENT_CLASSIFICATION.get(source_file, {
        "doc_type": "unknown", "priority": 9
    })
    doc_type = doc_info["doc_type"]
    priority = doc_info["priority"]
    
    chunks = []
    lines = text.split('\n')
    
    current_section = doc_type.replace("_", " ").title()
    accumulated_lines = []
    current_page = 1
    
    for line in lines:
        page_match = re.match(r'--- PAGE (\d+)', line)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        
        if line.strip():
            accumulated_lines.append(line.strip())
        
        if len(' '.join(accumulated_lines)) > 2000:
            chunk = _build_chunk(
                accumulated_lines, source_file, doc_type, priority,
                current_section, None, None, current_page
            )
            chunks.append(chunk)
            accumulated_lines = accumulated_lines[-2:]
    
    if accumulated_lines and len(' '.join(accumulated_lines)) >= 50:
        chunk = _build_chunk(
            accumulated_lines, source_file, doc_type, priority,
            current_section, None, None, current_page
        )
        chunks.append(chunk)
    
    return chunks


def chunk_core_rules(text, source_file):
    """
    Chunk the original core rules document (Up_Front.pdf).
    Similar to integrated_rules but without errata interleaving.
    """
    # The core rules have the same structure as integrated but without •............ markers
    chunks = []
    lines = text.split('\n')
    
    current_section = "Introduction"
    current_subsection = ""
    current_rule = None
    accumulated_lines = []
    current_page = 1
    
    for line in lines:
        page_match = re.match(r'--- PAGE (\d+)', line)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        
        # Detect section headers
        section_match = SECTION_HEADER_PATTERN.match(line)
        if section_match:
            if accumulated_lines and len(' '.join(accumulated_lines)) >= 100:
                chunk = _build_chunk(
                    accumulated_lines, source_file, "core_rules", 2,
                    current_section, current_subsection, current_rule,
                    current_page
                )
                chunks.append(chunk)
                accumulated_lines = []
            
            current_section = f"{section_match.group(1)}. {section_match.group(2).strip()}"
            current_subsection = ""
            current_rule = section_match.group(1) + ".0"
        
        # Detect rule numbers
        rule_match = re.match(r'^\s*(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\s', line)
        if rule_match:
            new_rule = rule_match.group(1)
            
            if accumulated_lines and len(' '.join(accumulated_lines)) >= 80:
                chunk = _build_chunk(
                    accumulated_lines, source_file, "core_rules", 2,
                    current_section, current_subsection, current_rule,
                    current_page
                )
                chunks.append(chunk)
                accumulated_lines = []
            
            current_rule = new_rule
            current_subsection = new_rule
        
        if line.strip():
            accumulated_lines.append(line.strip())
        
        if len(' '.join(accumulated_lines)) > 2500:
            chunk = _build_chunk(
                accumulated_lines, source_file, "core_rules", 2,
                current_section, current_subsection, current_rule,
                current_page
            )
            chunks.append(chunk)
            accumulated_lines = accumulated_lines[-2:]
    
    if accumulated_lines and len(' '.join(accumulated_lines)) >= 50:
        chunk = _build_chunk(
            accumulated_lines, source_file, "core_rules", 2,
            current_section, current_subsection, current_rule,
            current_page
        )
        chunks.append(chunk)
    
    return chunks


# ═══════════════════════════════════════════════════════════════════
# Chunk Builder & Helpers
# ═══════════════════════════════════════════════════════════════════

def _build_chunk(lines, source_file, doc_type, priority, section_path,
                 subsection, rule_number, page, scenario=None):
    """Build a chunk dictionary with full metadata."""
    text = '\n'.join(lines)
    
    # Extract rule numbers mentioned in the text
    rule_numbers = extract_rule_numbers(text)
    
    # Use the primary rule number (from parsing context) or the first detected one
    primary_rule = rule_number
    if not primary_rule and rule_numbers:
        primary_rule = rule_numbers[0]
    
    # Extract cross-references
    cross_refs = extract_cross_references(text)
    # Remove self-references
    if primary_rule and primary_rule in cross_refs:
        cross_refs.remove(primary_rule)
    
    # Detect content type
    content_type = detect_content_type(text)
    
    # Detect scenario if not explicitly provided
    detected_scenarios = detect_scenario(text)
    if not scenario and detected_scenarios:
        scenario = detected_scenarios[0]
    
    # Build the enriched text with section context header
    header = f"[Doc: {doc_type}] [Section: {section_path}]"
    if primary_rule:
        header += f" [Rule: {primary_rule}]"
    if scenario:
        header += f" [Scenario: {scenario}]"
    
    enriched_text = f"{header}\n{text}"
    
    # Build metadata (ChromaDB only supports str, int, float, bool)
    metadata = {
        "doc_type": doc_type,
        "source_file": source_file,
        "section_path": section_path or "",
        "content_type": content_type,
        "page": page,
        "priority": priority,
    }
    
    # ChromaDB metadata must be flat — store rule_number and scenario as strings
    if primary_rule:
        metadata["rule_number"] = primary_rule
    if scenario:
        metadata["scenario"] = scenario
    if cross_refs:
        metadata["cross_refs"] = ",".join(cross_refs)
    if subsection:
        metadata["subsection"] = subsection
    
    return {
        "text": enriched_text,
        "metadata": metadata,
        "rule_number": primary_rule,
        "cross_refs": cross_refs,
        "all_rule_numbers": rule_numbers
    }


def _find_split_point(lines):
    """Find a natural split point in accumulated lines (Q&A boundary, errata separator)."""
    # Look for errata separators or Q&A boundaries from the middle outward
    mid = len(lines) // 2
    best = -1
    
    for i in range(mid, len(lines)):
        line = lines[i]
        if ERRATA_SEPARATOR.search(line):
            best = i
            break
        if re.match(r'^\s*(Q\.|A\.)', line) and i > mid:
            best = i
            break
    
    if best == -1:
        # Try before the midpoint
        for i in range(mid - 1, max(0, mid - 10), -1):
            line = lines[i]
            if ERRATA_SEPARATOR.search(line):
                best = i
                break
    
    return best


# ═══════════════════════════════════════════════════════════════════
# Routing: File → Chunker
# ═══════════════════════════════════════════════════════════════════

CHUNKER_MAP = {
    "integrated_rules": chunk_integrated_rules,
    "core_rules": chunk_core_rules,
    "errata": chunk_errata,
    "scenario_errata": chunk_errata,  # Same format as errata
    "scenarios": chunk_scenarios,
    "variant": chunk_generic,
    "tournament": chunk_generic,
    "primer": chunk_generic,
}


# ═══════════════════════════════════════════════════════════════════
# Main Ingestion Pipeline
# ═══════════════════════════════════════════════════════════════════

def get_text_for_file(fname):
    """Get extractable text for a file — prefer pre-processed .txt if available."""
    txt_name = fname.replace(".pdf", ".txt")
    txt_path = os.path.join(TEXT_DIR, txt_name)
    
    # Check for duplicate flag
    dup_path = txt_path + ".duplicate"
    if os.path.exists(dup_path):
        print(f"  SKIPPED: {fname} flagged as duplicate by OCR processor")
        return None
    
    # Use pre-processed text if available
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    # Fall back to native PDF extraction
    pdf_path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(pdf_path):
        print(f"  ERROR: {fname} not found")
        return None
    
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text().strip()
        if text:
            pages.append(f"--- PAGE {i + 1} ---\n{text}")
    doc.close()
    
    return '\n\n'.join(pages)


def setup_vector_db():
    """Initialize ChromaDB collection for rules."""
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Clear existing collection
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print(f"  Cleared existing '{CHROMA_COLLECTION}' collection")
    except Exception:
        pass
    
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"  Created collection '{CHROMA_COLLECTION}'")
    return collection


def build_rule_index(all_chunks):
    """Build JSON rule-number lookup index from all chunks."""
    index = {}
    
    for chunk_id, chunk in all_chunks.items():
        rule_num = chunk.get("rule_number")
        all_rules = chunk.get("all_rule_numbers", [])
        
        # Index by primary rule number
        if rule_num:
            if rule_num not in index:
                index[rule_num] = []
            index[rule_num].append({
                "chunk_id": chunk_id,
                "doc_type": chunk["metadata"]["doc_type"],
                "priority": chunk["metadata"]["priority"],
                "source_file": chunk["metadata"]["source_file"],
                "content_type": chunk["metadata"]["content_type"],
            })
        
        # Also index by all mentioned rule numbers
        for r in all_rules:
            if r != rule_num:
                if r not in index:
                    index[r] = []
                index[r].append({
                    "chunk_id": chunk_id,
                    "doc_type": chunk["metadata"]["doc_type"],
                    "priority": chunk["metadata"]["priority"],
                    "source_file": chunk["metadata"]["source_file"],
                    "content_type": chunk["metadata"]["content_type"],
                    "secondary": True  # Not the primary rule of this chunk
                })
    
    # Sort each rule's entries by priority (lowest = highest authority)
    for rule_num in index:
        index[rule_num].sort(key=lambda x: x["priority"])
    
    return index


def ingest_all(dry_run=False):
    """
    Main ingestion pipeline.
    
    Args:
        dry_run: If True, parse and classify without embedding or indexing.
    """
    if not dry_run:
        collection = setup_vector_db()
    
    all_chunks = {}  # chunk_id -> chunk_data
    total_indexed = 0
    
    print(f"\n{'='*70}")
    print("INGESTION PIPELINE — Up Front Rules Lawyer")
    print(f"{'='*70}")
    
    # Process files in priority order
    sorted_files = sorted(
        DOCUMENT_CLASSIFICATION.items(),
        key=lambda x: x[1]["priority"]
    )
    
    for fname, doc_info in sorted_files:
        doc_type = doc_info["doc_type"]
        priority = doc_info["priority"]
        
        print(f"\n[P{priority}] {fname}")
        print(f"     Type: {doc_type} — {doc_info['description']}")
        
        # Get text content
        text = get_text_for_file(fname)
        if text is None:
            continue
        
        if len(text.strip()) < 50:
            print(f"     SKIPPED: No extractable text (length={len(text.strip())})")
            continue
        
        # Route to appropriate chunker
        chunker = CHUNKER_MAP.get(doc_type, chunk_generic)
        chunks = chunker(text, fname)
        
        print(f"     Chunks: {len(chunks)}")
        
        if dry_run:
            # Print sample chunks
            for i, chunk in enumerate(chunks[:3]):
                rule = chunk.get("rule_number", "N/A")
                refs = chunk.get("cross_refs", [])
                ctype = chunk["metadata"]["content_type"]
                text_preview = chunk["text"][:120].replace('\n', ' ')
                print(f"       [{i+1}] rule={rule} type={ctype} refs={refs}")
                print(f"           \"{text_preview}...\"")
            if len(chunks) > 3:
                print(f"       ... and {len(chunks) - 3} more chunks")
            continue
        
        # Embed and index chunks
        batch_ids = []
        batch_embeddings = []
        batch_metadatas = []
        batch_documents = []
        
        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_id = f"{doc_type}_{fname.replace('.pdf', '').replace(' ', '_').lower()}_chunk_{chunk_idx}"
            
            # Generate embedding
            try:
                response = ollama.embeddings(model="nomic-embed-text", prompt=chunk["text"])
                embedding = response["embedding"]
            except Exception as e:
                print(f"     ERROR embedding chunk {chunk_idx}: {e}")
                continue
            
            batch_ids.append(chunk_id)
            batch_embeddings.append(embedding)
            batch_metadatas.append(chunk["metadata"])
            batch_documents.append(chunk["text"])
            
            # Store for rule index
            all_chunks[chunk_id] = chunk
            
            # Batch upsert every 20 chunks
            if len(batch_ids) >= 20:
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                batch_ids, batch_embeddings, batch_metadatas, batch_documents = [], [], [], []
        
        # Upsert remaining
        if batch_ids:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )
        
        total_indexed += len(chunks)
        print(f"     Indexed: {len(chunks)} chunks")
    
    if not dry_run:
        # Build and save rule number index
        print(f"\n{'='*70}")
        print("Building rule-number lookup index...")
        rule_index = build_rule_index(all_chunks)
        
        with open(RULE_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(rule_index, f, indent=2)
        
        print(f"  Rule numbers indexed: {len(rule_index)}")
        print(f"  Saved to: {RULE_INDEX_FILE}")
        
        # Print top-level stats
        print(f"\n{'='*70}")
        print("INGESTION COMPLETE")
        print(f"  Total chunks indexed: {total_indexed}")
        print(f"  Unique rule numbers: {len(rule_index)}")
        print(f"  ChromaDB collection: {CHROMA_COLLECTION}")
        print(f"  Rule index: {RULE_INDEX_FILE}")
        
        # Show some sample rule lookups
        print(f"\n  Sample rule index entries:")
        sample_rules = list(rule_index.keys())[:5]
        for r in sample_rules:
            entries = rule_index[r]
            sources = [f"{e['doc_type']}(P{e['priority']})" for e in entries[:3]]
            print(f"    Rule {r}: {', '.join(sources)}")
    else:
        print(f"\n{'='*70}")
        print("DRY RUN COMPLETE")
        print(f"  Total chunks parsed: {sum(1 for _ in all_chunks) if all_chunks else 'N/A (dry run)'}")


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    profile_arg = next((a for a in sys.argv[1:] if a.endswith("_profile.json")), None)

    if profile_arg:
        # Generic game ingestion
        from engine.ingestion.ingest_rules import ingest_game
        ingest_game(profile_arg, dry_run=dry_run)
    else:
        # Up Front (legacy)
        if dry_run:
            print("Running in DRY RUN mode (no embedding or indexing)")
        ingest_all(dry_run=dry_run)


# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# GENERIC GAME INGESTION — Agnostic RAG Doll
# All functions below are ADDITIVE. Nothing above is changed.
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════

def load_profile(profile_path):
    """Load a game profile JSON and return the dict."""
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_compiled_patterns(profile):
    """
    Compile the regex patterns stored in the profile.
    Returns dict with 'rule' and 'cross_ref' compiled pattern objects.
    """
    rule_pat = re.compile(profile["rule_pattern"], re.MULTILINE)

    # cross_ref may have multiple groups — compile with MULTILINE
    try:
        cross_pat = re.compile(profile["cross_ref_pattern"], re.MULTILINE)
    except re.error as e:
        print(f"  Warning: cross_ref_pattern compile failed: {e}. Using rule pattern.")
        cross_pat = rule_pat

    return {"rule": rule_pat, "cross_ref": cross_pat}


def extract_rule_numbers_with_pattern(text, rule_pattern):
    """Extract rule numbers using a custom compiled pattern."""
    return list(set(rule_pattern.findall(text)))


def extract_cross_refs_with_pattern(text, cross_pattern):
    """Extract cross-references using a custom compiled pattern."""
    refs = set()
    for match in cross_pattern.finditer(text):
        for group in match.groups():
            if group:
                refs.add(group)
    return list(refs)


def _build_chunk_generic(lines, source_file, doc_type, priority,
                         section_path, subsection, rule_number, page,
                         patterns, scenario=None):
    """
    Generic version of _build_chunk that uses profile-supplied patterns
    for rule number and cross-reference extraction.
    """
    text = "\n".join(lines)

    # Extract rule numbers using the profile pattern
    rule_numbers = extract_rule_numbers_with_pattern(text, patterns["rule"])

    # Primary rule
    primary_rule = rule_number
    if not primary_rule and rule_numbers:
        primary_rule = rule_numbers[0]

    # Cross-references
    cross_refs = extract_cross_refs_with_pattern(text, patterns["cross_ref"])
    if primary_rule and primary_rule in cross_refs:
        cross_refs.remove(primary_rule)

    # Content type (reuse existing detection)
    content_type = detect_content_type(text)

    # Scenario detection
    if not scenario:
        detected = detect_scenario(text)
        if detected:
            scenario = detected[0]

    # Build enriched text
    header = f"[Doc: {doc_type}] [Section: {section_path}]"
    if primary_rule:
        header += f" [Rule: {primary_rule}]"
    if scenario:
        header += f" [Scenario: {scenario}]"
    enriched_text = f"{header}\n{text}"

    # Metadata
    metadata = {
        "doc_type": doc_type,
        "source_file": source_file,
        "section_path": section_path or "",
        "content_type": content_type,
        "page": page,
        "priority": priority,
    }
    if primary_rule:
        metadata["rule_number"] = primary_rule
    if scenario:
        metadata["scenario"] = scenario
    if cross_refs:
        metadata["cross_refs"] = ",".join(cross_refs)
    if subsection:
        metadata["subsection"] = subsection

    return {
        "text": enriched_text,
        "metadata": metadata,
        "rule_number": primary_rule,
        "cross_refs": cross_refs,
        "all_rule_numbers": rule_numbers,
    }


def chunk_rules_generic(text, source_file, doc_type, priority, patterns):
    """
    Generic rule-boundary chunker driven by a compiled rule pattern.
    Splits on rule number headers detected by the profile regex.
    Works for numeric decimal (Up Front) and chapter-decimal (ASL) schemas.
    """
    chunks = []
    lines = text.split("\n")

    current_rule = None
    current_section = doc_type.replace("_", " ").title()
    accumulated_lines = []
    current_page = 1

    for line in lines:
        # Track page markers
        page_match = re.match(r"--- PAGE (\d+)", line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        # Detect rule number at start of line
        # Try matching the rule pattern at line start
        stripped = line.strip()
        rule_match = patterns["rule"].match(stripped)
        if not rule_match:
            # Try anchored version: does line start with a rule number?
            rule_match = re.match(r"^([A-Z]?\d{1,2}\.\d{1,4}(?:\.\d{1,2})?)\s", stripped)

        if rule_match:
            new_rule = rule_match.group(1)
            if accumulated_lines and len(" ".join(accumulated_lines)) >= 80:
                chunk = _build_chunk_generic(
                    accumulated_lines, source_file, doc_type, priority,
                    current_section, current_rule, current_rule,
                    current_page, patterns
                )
                chunks.append(chunk)
                accumulated_lines = []
            current_rule = new_rule
            current_section = new_rule

        if stripped:
            accumulated_lines.append(stripped)

        # Force split on large chunks
        if len(" ".join(accumulated_lines)) > 2500:
            split_pt = _find_split_point(accumulated_lines)
            if split_pt > 0:
                chunk = _build_chunk_generic(
                    accumulated_lines[:split_pt], source_file, doc_type, priority,
                    current_section, current_rule, current_rule, current_page, patterns
                )
                chunks.append(chunk)
                accumulated_lines = accumulated_lines[max(0, split_pt - 2):]
            else:
                chunk = _build_chunk_generic(
                    accumulated_lines, source_file, doc_type, priority,
                    current_section, current_rule, current_rule, current_page, patterns
                )
                chunks.append(chunk)
                accumulated_lines = accumulated_lines[-2:]

    # Final chunk
    if accumulated_lines and len(" ".join(accumulated_lines)) >= 50:
        chunk = _build_chunk_generic(
            accumulated_lines, source_file, doc_type, priority,
            current_section, current_rule, current_rule, current_page, patterns
        )
        chunks.append(chunk)

    return chunks


def chunk_scenarios_generic(text, source_file, doc_type, priority,
                             profile, patterns):
    """
    Generic scenario chunker — handles named/lettered/numeric scenario formats.
    Each scenario card (turn record + special rules + VCs) = one chunk.
    """
    chunks = []
    lines = text.split("\n")

    current_scenario = None
    current_section = "Scenarios"
    accumulated_lines = []
    current_page = 1

    # Detect scenario header pattern from profile
    scenario_fmt = profile.get("scenario_format", "named")
    if scenario_fmt == "letter":
        scenario_header_re = re.compile(r"^([A-K])[.:\s]+([A-Z])", re.MULTILINE)
    elif scenario_fmt == "numeric":
        scenario_header_re = re.compile(
            r"^(?:ASL\s+SCENARIO\s+|Scenario\s+)?(\w+\d+)\b|^(\d+)[.:\s]+[A-Z]",
            re.MULTILINE
        )
    else:  # named — split on ALL CAPS titles or "TURN RECORD CHART" (new scenario card start)
        scenario_header_re = re.compile(
            r"(?:^|\n)(TURN RECORD CHART|END\s+\d+|[A-Z][A-Z\s]{4,}(?:\n|$))",
            re.MULTILINE
        )

    for line in lines:
        page_match = re.match(r"--- PAGE (\d+)", line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        stripped = line.strip()

        # Check for scenario boundary
        sm = scenario_header_re.match(stripped)
        if sm and len(stripped) < 80:
            if accumulated_lines and current_scenario:
                chunk = _build_chunk_generic(
                    accumulated_lines, source_file, doc_type, priority,
                    f"Scenario {current_scenario}", current_scenario, None,
                    current_page, patterns, scenario=current_scenario
                )
                chunks.append(chunk)
                accumulated_lines = []

            current_scenario = sm.group(1) if sm.lastindex and sm.group(1) else stripped[:40]
            current_section = f"Scenario: {current_scenario}"

        if stripped:
            accumulated_lines.append(stripped)

        # Force split on large chunks
        if len(" ".join(accumulated_lines)) > 2000:
            chunk = _build_chunk_generic(
                accumulated_lines, source_file, doc_type, priority,
                current_section, current_scenario, None, current_page, patterns,
                scenario=current_scenario
            )
            chunks.append(chunk)
            accumulated_lines = accumulated_lines[-2:]

    # Final chunk
    if accumulated_lines and len(" ".join(accumulated_lines)) >= 50:
        chunk = _build_chunk_generic(
            accumulated_lines, source_file, doc_type, priority,
            current_section, current_scenario, None, current_page, patterns,
            scenario=current_scenario
        )
        chunks.append(chunk)

    return chunks


def chunk_generic_with_profile(text, source_file, doc_type, priority,
                                profile, patterns):
    """
    Fallback size-based chunker for documents without a clear rule structure
    (QA docs, version trackers, primers, etc).
    """
    chunks = []
    lines = text.split("\n")
    current_section = doc_type.replace("_", " ").title()
    accumulated_lines = []
    current_page = 1

    for line in lines:
        page_match = re.match(r"--- PAGE (\d+)", line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        stripped = line.strip()
        if stripped:
            accumulated_lines.append(stripped)

        if len(" ".join(accumulated_lines)) > 2000:
            chunk = _build_chunk_generic(
                accumulated_lines, source_file, doc_type, priority,
                current_section, None, None, current_page, patterns
            )
            chunks.append(chunk)
            accumulated_lines = accumulated_lines[-2:]

    if accumulated_lines and len(" ".join(accumulated_lines)) >= 50:
        chunk = _build_chunk_generic(
            accumulated_lines, source_file, doc_type, priority,
            current_section, None, None, current_page, patterns
        )
        chunks.append(chunk)

    return chunks


def parse_datasheet(lines):
    """
    Attempt to format complex unit datasheets into markdown tables.
    Returns the formatted text.
    """
    # Placeholder for PDFPlumber/Camelot integration if available,
    # otherwise we rely on PyMuPDF's raw text and format it slightly.
    # We will just return the lines for now, but tag it so the LLM knows it's a datasheet.
    return " ".join(lines)



def normalise_ocr_spaced_text(text):
    """
    Fix OCR-spaced letters in scanned rulebook PDFs.

    Scanned PDFs (pdfcoffee RL books) separate every letter with a space.
    Multi-word headers use double-spaces as word boundaries:
        'S U P P R E S S I O N'        -> 'SUPPRESSION'
        'H U L L  D O W N'             -> 'HULL DOWN'
        'P I L O T S  A N D  C R E W'  -> 'PILOTS AND CREW'
        'I N T E R C E P T O R'        -> 'INTERCEPTOR'

    Algorithm per line:
      1. Split on 2+ consecutive spaces - each part is a potential word group.
      2. Within each part, check if it is entirely single uppercase letters
         separated by single spaces (regex: ^([A-Z] )*[A-Z]$).
      3. If so, collapse by removing the spaces.
      4. If ALL parts in the line were collapsible, rejoin with ' '.
         Otherwise leave the line untouched (avoids mangling normal prose).
    """
    import re as _re
    spaced_group_re = _re.compile(r'^(?:[A-Z] )*[A-Z]$')

    def try_collapse(part):
        stripped = part.strip()
        if spaced_group_re.match(stripped):
            return stripped.replace(' ', '')
        return None  # not collapsible

    result_lines = []
    for line in text.split('\n'):
        # Split on 2-or-more spaces to find potential word groups
        parts = _re.split(r'  +', line)
        collapsed = [try_collapse(p) for p in parts]
        if all(c is not None for c in collapsed):
            # Every part was a spaced-letter group - join into normalised words
            result_lines.append(' '.join(c for c in collapsed if c))
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)


def chunk_keyword_header(text, source_file, doc_type, priority, profile, patterns):
    """
    Chunker for games without numbered rules (like Warhammer 40k).
    Uses visual formatting heuristics (ALL CAPS headers, title casing) to chunk.
    """
    chunks = []
    # --- OCR normalisation (RL scans have spaced letters in headers) ---
    text = normalise_ocr_spaced_text(text)

    lines = text.split("\n")

    current_header = None
    current_section = doc_type.replace("_", " ").title()
    accumulated_lines = []
    current_page = 1

    edition = profile.get("edition")

    # Chunk size: read from profile (RL uses 4000, others default to 2500)
    max_chunk_chars = profile.get("chunk_size", 2500)

    # Loose header pattern: All caps or Title Case, 4-60 chars, no periods at the end.
    header_pattern = re.compile(r"^([A-Z][A-Za-z0-9\s&\-]{3,60})$")

    for line in lines:
        page_match = re.match(r"--- PAGE (\d+)", line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        stripped = line.strip()
        
        # Check if line is a likely header
        is_header = False
        if header_pattern.match(stripped) and not stripped.endswith('.'):
            # Exclude common false positives
            if not any(stop in stripped.lower() for stop in ["page", "continue", "example"]):
                is_header = True

        if is_header:
            if accumulated_lines and len(" ".join(accumulated_lines)) >= 50:
                chunk_text = parse_datasheet(accumulated_lines) if "DATASHEET" in (current_header or "").upper() else " ".join(accumulated_lines)
                chunk = _build_chunk_generic(
                    [chunk_text], source_file, doc_type, priority,
                    current_section, current_header, current_header,
                    current_page, patterns
                )
                if edition:
                    chunk["metadata"]["edition"] = edition
                chunks.append(chunk)
                accumulated_lines = []
            
            current_header = stripped
            current_section = stripped

        if stripped:
            accumulated_lines.append(stripped)

        # Force split on large chunks (uses profile chunk_size; default 2500)
        if len(" ".join(accumulated_lines)) > max_chunk_chars:
            split_pt = _find_split_point(accumulated_lines)
            if split_pt > 0:
                chunk = _build_chunk_generic(
                    accumulated_lines[:split_pt], source_file, doc_type, priority,
                    current_section, current_header, current_header, current_page, patterns
                )
                if edition:
                    chunk["metadata"]["edition"] = edition
                chunks.append(chunk)
                accumulated_lines = accumulated_lines[max(0, split_pt - 2):]

    # Final chunk
    if accumulated_lines and len(" ".join(accumulated_lines)) >= 50:
        chunk = _build_chunk_generic(
            accumulated_lines, source_file, doc_type, priority,
            current_section, current_header, current_header, current_page, patterns
        )
        if edition:
            chunk["metadata"]["edition"] = edition
        chunks.append(chunk)

    return chunks

def route_chunk_generic(text, source_file, doc_info, profile, patterns):
    """
    Route a document to the appropriate generic chunker based on its doc_type.
    """
    doc_type = doc_info["doc_type"]
    priority = doc_info["priority"]

    # For keyword_header games (like RL), normalise OCR-spaced text on ALL doc types
    if profile.get("rule_schema") == "keyword_header":
        text = normalise_ocr_spaced_text(text)

    rule_chunk_types = {
        "core_rules", "core_rules_v1", "errata", "scenario_errata",
        "integrated_rules", "version_tracker", "codex"
    }
    scenario_types = {"scenarios", "scenario_balance", "tournament"}
    generic_types = {"qa", "journal", "variant", "primer", "supplement", "unknown"}

    rule_schema = profile.get("rule_schema", "unknown")

    if rule_schema == "keyword_header" and doc_type in rule_chunk_types:
        return chunk_keyword_header(text, source_file, doc_type, priority, profile, patterns)
    elif doc_type in rule_chunk_types:
        return chunk_rules_generic(text, source_file, doc_type, priority, patterns)
    elif doc_type in scenario_types:
        return chunk_scenarios_generic(text, source_file, doc_type, priority, profile, patterns)
    else:
        return chunk_generic_with_profile(text, source_file, doc_type, priority, profile, patterns)


def get_text_for_game_file(fname, profile):
    """
    Get text for a game file using profile-specified paths.
    Tries pre-processed text cache first, falls back to native PDF extraction.
    Respects max_pages setting from profile.
    """
    text_dir = profile.get("text_dir", "data/generic_text")
    data_dir = profile["data_dir"]
    doc_info = profile["documents"].get(fname, {})
    max_pages = doc_info.get("max_pages")

    # max_pages == 0 means skip this file
    if max_pages == 0:
        return None

    # Check text cache
    txt_path = os.path.join(text_dir, fname.replace(".pdf", ".txt"))
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    # Fall back to native PDF extraction
    pdf_path = os.path.join(data_dir, fname)
    if not os.path.exists(pdf_path):
        return None

    try:
        doc = fitz.open(pdf_path)
        pages = []
        limit = max_pages if max_pages else len(doc)

        for i in range(min(limit, len(doc))):
            try:
                text = doc[i].get_text().strip()
                if text:
                    pages.append(f"--- PAGE {i + 1} ---\n{text}")
            except Exception:
                continue

        doc.close()
        return "\n\n".join(pages) if pages else None
    except Exception as e:
        print(f"  Error reading {fname}: {e}")
        return None


def setup_vector_db_for_game(profile):
    """Initialize a ChromaDB collection for a specific game."""
    collection_name = profile["chroma_collection"]
    print(f"Initializing ChromaDB collection: {collection_name}")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    try:
        client.delete_collection(collection_name)
        print(f"  Cleared existing '{collection_name}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"  Created collection '{collection_name}'")
    return collection


def ingest_game(profile_path, dry_run=False):
    """
    Generic ingestion pipeline driven by a game profile JSON.

    Args:
        profile_path: Path to a game_profile.json generated by auto_discover.py.
        dry_run:      Parse and chunk without embedding or indexing.
    """
    profile = load_profile(profile_path)
    patterns = get_compiled_patterns(profile)

    game_name = profile["game_name"]
    rule_index_file = profile["rule_index_file"]

    print(f"\n{'=' * 70}")
    print(f"GENERIC INGESTION PIPELINE — {game_name}")
    print(f"Profile: {profile_path}")
    print(f"Schema: {profile.get('rule_schema', 'unknown')}")
    print(f"{'=' * 70}")

    if not dry_run:
        collection = setup_vector_db_for_game(profile)

    all_chunks = {}
    total_indexed = 0

    # Sort documents by priority for processing order
    docs_sorted = sorted(
        profile["documents"].items(),
        key=lambda x: x[1].get("priority", 9)
    )

    for fname, doc_info in docs_sorted:
        max_pages = doc_info.get("max_pages")
        if max_pages == 0:
            print(f"\n[SKIP] {fname} (max_pages=0 — too large/low priority)")
            continue

        priority = doc_info.get("priority", 9)
        doc_type = doc_info.get("doc_type", "unknown")

        cap_note = f" (first {max_pages}p)" if max_pages else ""
        print(f"\n[P{priority}] {fname}{cap_note}")
        print(f"     Type: {doc_type} — {doc_info.get('description', '')}")

        text = get_text_for_game_file(fname, profile)
        if text is None:
            print(f"     SKIPPED: No text available")
            continue

        if len(text.strip()) < 50:
            print(f"     SKIPPED: Insufficient text ({len(text.strip())} chars)")
            continue

        # Route to appropriate chunker
        chunks = route_chunk_generic(text, fname, doc_info, profile, patterns)
        print(f"     Chunks: {len(chunks)}")

        if dry_run:
            for i, chunk in enumerate(chunks[:3]):
                rule = chunk.get("rule_number", "N/A")
                refs = chunk.get("cross_refs", [])
                preview = chunk["text"][:100].replace("\n", " ")
                preview = preview.encode("ascii", errors="replace").decode("ascii")
                print(f"       [{i+1}] rule={rule} xrefs={refs[:3]}")
                print(f"           \"{preview}...\"")
            if len(chunks) > 3:
                print(f"       ... and {len(chunks) - 3} more")
            continue

        # Embed and index
        batch_ids, batch_embeddings, batch_metas, batch_docs = [], [], [], []

        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_id = (
                f"{doc_type}_{re.sub(r'[^a-z0-9]', '_', fname.lower()[:40])}"
                f"_chunk_{chunk_idx}"
            )

            try:
                response = ollama.embeddings(model="nomic-embed-text", prompt=chunk["text"])
                embedding = response["embedding"]
            except Exception as e:
                print(f"     ERROR embedding chunk {chunk_idx}: {e}")
                continue

            batch_ids.append(chunk_id)
            batch_embeddings.append(embedding)
            batch_metas.append(chunk["metadata"])
            batch_docs.append(chunk["text"])
            all_chunks[chunk_id] = chunk

            if len(batch_ids) >= 20:
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metas,
                    documents=batch_docs
                )
                batch_ids, batch_embeddings, batch_metas, batch_docs = [], [], [], []

        if batch_ids:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metas,
                documents=batch_docs
            )

        total_indexed += len(chunks)
        print(f"     Indexed: {len(chunks)} chunks")

    if not dry_run:
        # Build rule index
        print(f"\n{'=' * 70}")
        print("Building rule-number lookup index...")
        rule_index = build_rule_index(all_chunks)

        with open(rule_index_file, "w", encoding="utf-8") as f:
            json.dump(rule_index, f, indent=2)

        print(f"  Rule numbers indexed: {len(rule_index)}")
        print(f"  Saved: {rule_index_file}")

        print(f"\n{'=' * 70}")
        print(f"INGESTION COMPLETE — {game_name}")
        print(f"  Total chunks: {total_indexed}")
        print(f"  Unique rules: {len(rule_index)}")
        print(f"  Collection:   {profile['chroma_collection']}")
    else:
        print(f"\n{'=' * 70}")
        print(f"DRY RUN COMPLETE — {game_name}")
        print(f"  Chunks parsed: {total_indexed}")
