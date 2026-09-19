"""
Rules Lawyer Ingestion Pipeline.

Classifies, chunks, and indexes Up Front wargame rule documents into ChromaDB
with a parallel JSON rule-number lookup index.

Document types are prioritized for temporal supersession handling.
Chunking is rule-number-aware, preserving Q&A blocks and cross-references.
"""

import os
import sys
import re
import json
import fitz  # PyMuPDF
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from engine.models.cooccurrence_graph import CooccurrenceGraph, SectionTree
    from engine.providers import (
        BaseLLMProvider,
        BaseStorageProvider,
        CloudDynamoStorageProvider,
        get_llm_provider,
        get_storage_provider,
    )
except ImportError:
    from models.cooccurrence_graph import CooccurrenceGraph, SectionTree
    from providers import (
        BaseLLMProvider,
        BaseStorageProvider,
        CloudDynamoStorageProvider,
        get_llm_provider,
        get_storage_provider,
    )

DATA_DIR = "data/upfront"
TEXT_DIR = "data/upfront_text"
CHROMA_DB_DIR = "data/chroma"
CHROMA_COLLECTION = "upfront-rules-semantic"
RULE_INDEX_FILE = "data/upfront_rule_index.json"
COOCCURRENCE_GRAPH_FILE = "data/upfront_cooccurrence_graph.json"
SECTION_TREE_FILE = "data/upfront_section_tree.json"

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

def _extract_hierarchy_from_rule(rule_str, profile=None):
    """
    Extract (root_section, parent_id, hierarchy_level) for a rule identifier dynamically
    using the parsing grammar / hierarchy strategy defined in the domain profile.
    """
    if not rule_str:
        return "", "", 1
    clean_r = re.sub(r'[\(\)\[\]]', '', str(rule_str)).strip()
    if not clean_r:
        return "", "", 1

    # Extract configuration from profile if provided
    strategy = None
    delimiter = " > "
    custom_regex = None

    if profile:
        # Handle DomainProfile instance or dict
        if hasattr(profile, "get_hierarchy_strategy"):
            strategy = profile.get_hierarchy_strategy()
            delimiter = profile.get_hierarchy_delimiter()
            custom_regex = profile.get_hierarchy_regex()
        elif isinstance(profile, dict):
            grammar = profile.get("parsing_grammar") or {}
            strategy = grammar.get("hierarchy_strategy") or profile.get("hierarchy_strategy") or profile.get("rule_schema")
            delimiter = grammar.get("hierarchy_delimiter") or profile.get("hierarchy_delimiter") or " > "
            custom_regex = grammar.get("hierarchy_regex")

    # 1. User-defined custom regex strategy
    if custom_regex:
        m = re.match(custom_regex, clean_r)
        if m:
            gd = m.groupdict()
            root = gd.get("root", "")
            parent = gd.get("parent", "")
            try:
                level = int(gd.get("level", 2 if parent else 1))
            except (ValueError, TypeError):
                level = 2 if parent else 1
            return root, parent, level

    # 2. Strategy: Breadcrumb Path (e.g. for keyword_header: "COMBAT > Missile Combat > Penetration Resistance")
    if strategy in {"breadcrumb_path", "keyword_header"}:
        eff_delim = delimiter if (delimiter and delimiter in clean_r) else (" > " if " > " in clean_r else None)
        if eff_delim:
            crumbs = [c.strip() for c in clean_r.split(eff_delim) if c.strip()]
            if len(crumbs) == 1:
                return crumbs[0], "", 1
            elif len(crumbs) == 2:
                return crumbs[0], crumbs[0], 2
            else:
                return crumbs[0], crumbs[-2], len(crumbs)
        return clean_r, "", 1

    # 3. Strategy: Numbered Clause (e.g. "4. Armour Penetration" or "1. Initiative")
    if strategy == "numbered_clause":
        m_clause = re.match(r'^(\d{1,2})\.\s+([A-Za-z].*)', clean_r)
        if m_clause:
            clause_num = m_clause.group(1)
            clause_title = m_clause.group(2).strip()
            if not clause_title:
                return f"{clause_num}.0", "", 1
            return f"{clause_num}.0", f"{clause_num}.0", 2

    # 4. Strategy: Numeric Decimal (e.g. "20.73", "5.41", "3.2")
    if strategy == "numeric_decimal":
        m_dec = re.match(r'^(\d{1,2})\.(\d{1,4})$', clean_r)
        if m_dec:
            main_sec = m_dec.group(1)
            sub = m_dec.group(2)
            if sub in {"0", "00"}:
                return f"{main_sec}.0", "", 1
            elif len(sub) == 1:
                return f"{main_sec}.0", f"{main_sec}.0", 2
            else:
                return f"{main_sec}.0", f"{main_sec}.{sub[0]}", 3

    # 5. Strategy: Chapter Decimal (e.g. "A7.21")
    if strategy == "chapter_decimal":
        m_chap = re.match(r'^([A-Z])(\d{1,2})\.(\d{1,4})$', clean_r)
        if m_chap:
            chap = m_chap.group(1)
            main_sec = m_chap.group(2)
            sub = m_chap.group(3)
            return f"{chap}{main_sec}.0", f"{chap}{main_sec}.{sub[0]}", 3

    # 6. Strategy: Outline Parenthetical (e.g. "(D2.31)")
    if strategy == "outline_parenthetical":
        m_out = re.match(r'^(?:\()?([A-Z])(\d{1,2})\.(\d{1,4})(?:\))?$', clean_r)
        if m_out:
            chap = m_out.group(1)
            main_sec = m_out.group(2)
            sub = m_out.group(3)
            return f"({chap}{main_sec}.0)", f"({chap}{main_sec}.{sub[0]})", 3

    # 7. Generic fallback matching
    if " > " in clean_r:
        crumbs = [c.strip() for c in clean_r.split(" > ") if c.strip()]
        if len(crumbs) == 1:
            return crumbs[0], "", 1
        elif len(crumbs) == 2:
            return crumbs[0], crumbs[0], 2
        else:
            return crumbs[0], crumbs[-2], len(crumbs)

    m_clause = re.match(r'^(\d{1,2})\.\s+([A-Za-z].*)', clean_r)
    if m_clause:
        clause_num = m_clause.group(1)
        clause_title = m_clause.group(2).strip()
        if not clause_title:
            return f"{clause_num}.0", "", 1
        return f"{clause_num}.0", f"{clause_num}.0", 2

    m_chap = re.match(r'^([A-Z])(\d{1,2})\.(\d{1,4})$', clean_r)
    if m_chap:
        chap = m_chap.group(1)
        main_sec = m_chap.group(2)
        sub = m_chap.group(3)
        return f"{chap}{main_sec}.0", f"{chap}{main_sec}.{sub[0]}", 3

    m_dec = re.match(r'^(\d{1,2})\.(\d{1,4})$', clean_r)
    if m_dec:
        main_sec = m_dec.group(1)
        sub = m_dec.group(2)
        if sub in {"0", "00"}:
            return f"{main_sec}.0", "", 1
        elif len(sub) == 1:
            return f"{main_sec}.0", f"{main_sec}.0", 2
        else:
            return f"{main_sec}.0", f"{main_sec}.{sub[0]}", 3

    parts = [p.strip() for p in clean_r.split('.') if p.strip()]
    if len(parts) == 0:
        return "", "", 1
    elif len(parts) == 1:
        return f"{parts[0]}.0" if parts[0].isdigit() else parts[0], "", 1
    elif len(parts) == 2:
        main_sec, sub = parts[0], parts[1]
        if sub.isdigit():
            if sub in {"0", "00"}:
                return f"{main_sec}.0", "", 1
            elif len(sub) == 1:
                return f"{main_sec}.0", f"{main_sec}.0", 2
            else:
                return f"{main_sec}.0", f"{main_sec}.{sub[0]}", 3
        else:
            return main_sec, main_sec, 2
    else:
        return parts[0], parts[-2], len(parts)


def _build_chunk(lines, source_file, doc_type, priority, section_path,
                 subsection, rule_number, page, scenario=None, profile=None):
    """Build a chunk dictionary with full metadata including hierarchical section context."""
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

    # Hierarchical tagging using profile metadata
    root_section, parent_id, hierarchy_level = _extract_hierarchy_from_rule(primary_rule, profile=profile)
    
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
        "root_section": root_section,
        "parent_id": parent_id,
        "hierarchy_level": hierarchy_level,
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
        "all_rule_numbers": rule_numbers,
        "root_section": root_section,
        "parent_id": parent_id,
        "hierarchy_level": hierarchy_level
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
    import chromadb
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


def build_section_tree(all_chunks, game_id="generic"):
    """Build a SectionTree representing the document section hierarchy and rule mappings."""
    tree = SectionTree(game_id=game_id)
    for chunk_id, chunk in all_chunks.items():
        meta = chunk.get("metadata", {})
        rule_num = chunk.get("rule_number")
        root_sec = meta.get("root_section", "") or chunk.get("root_section", "")
        sec_path = meta.get("section_path", "")
        parent_id = meta.get("parent_id", "")
        doc_type = meta.get("doc_type", "core_rules")
        priority = meta.get("priority", 1)

        if not root_sec:
            if rule_num and "." in str(rule_num):
                root_sec = str(rule_num).split(".")[0] + ".0"
            else:
                root_sec = sec_path or "Rules Reference"

        if root_sec:
            tree.add_section(
                section_id=root_sec,
                title=sec_path or root_sec,
                parent_id=None,
                level=1,
                doc_type=doc_type,
                priority=priority
            )
        if parent_id and parent_id != root_sec:
            tree.add_section(
                section_id=parent_id,
                title=parent_id,
                parent_id=root_sec,
                level=2,
                doc_type=doc_type,
                priority=priority
            )
        if rule_num:
            eff_parent = parent_id or root_sec or "0.0"
            tree.register_rule(rule_id=rule_num, parent_section_id=eff_parent, chunk_id=chunk_id)
    return tree



def build_ingestion_cooccurrence_graph(all_chunks, rule_index, glossary=None, section_tree=None, game_id="generic"):
    """
    Build a rich, document-derived CooccurrenceGraph purely from document contents during ingestion:
    1. Cross-reference directed citations (W=1.0 forward, W=0.75 reciprocal).
    2. SectionTree structural siblings & parent-child links (W=0.75).
    3. Glossary & specialized entity co-occurrence across distinct chapters (PMI scoring W in [0.45, 0.85]).
    """
    graph = CooccurrenceGraph(game_id=game_id)

    # 1. Cross-reference edges
    for chunk_id, chunk in all_chunks.items():
        src_rule = chunk.get("rule_number")
        cross_refs = chunk.get("cross_refs", [])
        if src_rule and cross_refs:
            for ref in cross_refs:
                if ref != src_rule:
                    graph.add_edge(src_rule, ref, weight=1.0, relation_type="cross_reference")
                    graph.add_edge(ref, src_rule, weight=0.75, relation_type="cross_reference_reciprocal")

    # 2. Structural sibling edges via SectionTree (window of 4 adjacent siblings)
    if section_tree:
        for sec_id, sec_node in section_tree.sections.items():
            child_rules = sec_node.child_rules
            n = len(child_rules)
            if n > 1:
                for i, r1 in enumerate(child_rules):
                    for r2 in child_rules[i + 1 : min(n, i + 5)]:
                        graph.add_bidirectional_edge(r1, r2, weight=0.75, relation_type="structural_sibling")

    # 3. Glossary & Domain Entity Co-Occurrence (PMI via Inverted Index)
    domain_terms = set()
    if glossary and isinstance(glossary, dict):
        for term, expansion in glossary.items():
            if len(term) >= 2:
                domain_terms.add(term.lower())
            if expansion and len(expansion) > 3:
                for token in re.findall(r'\b[A-Za-z]{4,}\b', expansion.lower()):
                    if token not in {"there", "when", "although", "this", "that", "with", "from", "must", "have"}:
                        domain_terms.add(token)

    entity_pattern = re.compile(
        r'\b(?:Squad Leader|Close Combat|Flanking Fire|Moving Fire|Relative Range|Morale Check|Buttoned Up|Open Topped|Infiltrator|Sniper|Minefield|Radio|Smoke|Rally|Pinning|Armored Vehicle|AFV|LATW|Ordnance|Discard|Action Phase)\b',
        re.IGNORECASE
    )

    rule_terms = {}
    rule_chapters = {}
    for chunk_id, chunk in all_chunks.items():
        r = chunk.get("rule_number")
        if not r:
            continue
        text = chunk.get("text", "").lower()
        if r not in rule_terms:
            rule_terms[r] = set()

        for term in domain_terms:
            if term in text:
                rule_terms[r].add(term)

        for m in entity_pattern.findall(chunk.get("text", "")):
            rule_terms[r].add(m.lower())

        root_sec = chunk.get("metadata", {}).get("root_section", "")
        if root_sec:
            rule_chapters[r] = root_sec

    # Inverted index: term -> list of rule IDs
    term_to_rules = {}
    for r, terms in rule_terms.items():
        for term in terms:
            term_to_rules.setdefault(term, []).append(r)

    pair_shared_terms = {}
    for term, r_list in term_to_rules.items():
        if len(r_list) > 100:  # Skip ubiquitous stop-like terms
            continue
        for i, r1 in enumerate(r_list):
            for r2 in r_list[i + 1:]:
                pair = (min(r1, r2), max(r1, r2))
                pair_shared_terms.setdefault(pair, set()).add(term)

    for (r1, r2), shared in pair_shared_terms.items():
        if len(shared) < 2:
            continue
        chap1 = rule_chapters.get(r1, r1.split('.')[0] if '.' in r1 else r1)
        chap2 = rule_chapters.get(r2, r2.split('.')[0] if '.' in r2 else r2)
        if chap1 == chap2:
            continue
        terms1 = rule_terms.get(r1, set())
        terms2 = rule_terms.get(r2, set())
        union = terms1 | terms2
        jaccard = len(shared) / len(union) if union else 0.0
        if jaccard >= 0.12 or len(shared) >= 3:
            weight = min(0.85, 0.45 + (jaccard * 0.35) + (len(shared) * 0.05))
            graph.add_bidirectional_edge(
                r1, r2, weight=round(weight, 3),
                relation_type="glossary_pmi",
                shared_terms=sorted(list(shared))[:5]
            )

    return graph


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
        # Build and save rule number index, section tree, and cooccurrence graph
        print(f"\n{'='*70}")
        print("Building rule-number lookup index & hierarchy tree...")
        rule_index = build_rule_index(all_chunks)
        section_tree = build_section_tree(all_chunks, game_id="upfront")
        cooc_graph = build_ingestion_cooccurrence_graph(all_chunks, rule_index, section_tree=section_tree, game_id="upfront")

        # Embed section tree into rule index
        rule_index["__section_tree__"] = section_tree.model_dump()
        
        with open(RULE_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(rule_index, f, indent=2)
        
        cooc_graph.save_json(COOCCURRENCE_GRAPH_FILE)
        section_tree.save_json(SECTION_TREE_FILE)

        print(f"  Rule numbers indexed: {len(rule_index) - 1}")
        print(f"  Section nodes indexed: {len(section_tree.sections)}")
        print(f"  Co-occurrence graph edges: {sum(len(edges) for edges in cooc_graph.adjacency.values())}")
        print(f"  Saved to: {RULE_INDEX_FILE}")
        print(f"  Saved to: {COOCCURRENCE_GRAPH_FILE}")
        print(f"  Saved to: {SECTION_TREE_FILE}")
        
        # Print top-level stats
        print(f"\n{'='*70}")
        print("INGESTION COMPLETE")
        print(f"  Total chunks indexed: {total_indexed}")
        print(f"  Unique rule numbers: {len(rule_index) - 1}")
        print(f"  ChromaDB collection: {CHROMA_COLLECTION}")
        print(f"  Rule index: {RULE_INDEX_FILE}")
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
                         patterns, scenario=None, profile=None):
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

    # Hierarchical tagging using profile metadata
    root_section, parent_id, hierarchy_level = _extract_hierarchy_from_rule(primary_rule, profile=profile)

    # Metadata
    metadata = {
        "doc_type": doc_type,
        "source_file": source_file,
        "section_path": section_path or "",
        "content_type": content_type,
        "page": page,
        "priority": priority,
        "root_section": root_section,
        "parent_id": parent_id,
        "hierarchy_level": hierarchy_level,
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
        "root_section": root_section,
        "parent_id": parent_id,
        "hierarchy_level": hierarchy_level,
    }


def chunk_rules_generic(text, source_file, doc_type, priority, patterns, profile=None):
    """
    Generic rule-boundary chunker driven by a compiled rule pattern.
    Splits on rule number headers detected by the profile regex.
    Works for numeric decimal (Up Front) and chapter-decimal (ASL) schemas.
    """
    chunks = []
    lines = text.split("\n")

    # Detect initial chapter from filename if chapter_decimal (e.g. A15-A16vB.pdf -> A, B7-B8vB.pdf -> B, D5-D8vB.pdf -> D)
    current_chapter = None
    fname_match = re.match(r"^([A-Z])\d+", source_file)
    if fname_match:
        current_chapter = fname_match.group(1).upper()

    current_rule = None
    current_section = doc_type.replace("_", " ").title()
    accumulated_lines = []
    current_page = 1

    is_chapter_decimal = bool(profile and profile.get("rule_schema") == "chapter_decimal")

    for line in lines:
        # Track page markers
        page_match = re.match(r"--- PAGE (\d+)", line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        stripped = line.strip()

        # If chapter_decimal schema, detect chapter markers on page/headers:
        # e.g. "CHAPTER A", "CHAPTER B", "A. INFANTRY", "B. TERRAIN", "C. GUNS", "D. VEHICLES", "E. MISCELLANEOUS", "G. PTO"
        if is_chapter_decimal:
            chap_match = re.match(
                r"^(?:CHAPTER\s+([A-Z])|([A-Z])\.\s+[A-Z\s]{4,}|(?:CHAPTER\s+)?([A-Z])\s*[-–—]\s*[A-Z\s]+)",
                stripped,
                re.IGNORECASE
            )
            if chap_match:
                current_chapter = (chap_match.group(1) or chap_match.group(2) or chap_match.group(3)).upper()
            else:
                # Also check footer/header markers like "D24", "A10", "B5" alone on line
                footer_match = re.match(r"^([A-Z])\d{1,3}$", stripped)
                if footer_match:
                    current_chapter = footer_match.group(1).upper()

        # Detect rule number at start of line
        rule_match = patterns["rule"].match(stripped)
        if not rule_match:
            # Try anchored version: does line start with a rule number?
            rule_match = re.match(r"^([A-Z]?\d{1,2}\.\d{1,4}(?:\.\d{1,2})?)\s", stripped)

        if rule_match:
            raw_rule = rule_match.group(1)
            # Normalize OCR noise in decimal rule numbers (e.g. 3.O -> 3.0, [3.1] -> 3.1)
            if profile and (profile.get("ocr_normalize") or profile.get("rule_schema") == "numeric_decimal"):
                raw_rule = raw_rule.strip("[]")
                raw_rule = re.sub(r'(?<=\d\.)O\b', '0', raw_rule)
            # If chapter_decimal and rule starts with digits (e.g. "23.5") but we know current_chapter (e.g. "A"):
            if is_chapter_decimal and not raw_rule[0].isalpha() and current_chapter:
                new_rule = f"{current_chapter}{raw_rule}"
            else:
                new_rule = raw_rule
                if is_chapter_decimal and new_rule[0].isalpha():
                    current_chapter = new_rule[0].upper()

            if accumulated_lines and len(" ".join(accumulated_lines)) >= 80:
                chunk = _build_chunk_generic(
                    accumulated_lines, source_file, doc_type, priority,
                    current_section, current_rule, current_rule,
                    current_page, patterns,
                    profile=profile
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
                    current_section, current_rule, current_rule, current_page, patterns,
                    profile=profile
                )
                chunks.append(chunk)
                accumulated_lines = accumulated_lines[max(0, split_pt - 2):]
            else:
                chunk = _build_chunk_generic(
                    accumulated_lines, source_file, doc_type, priority,
                    current_section, current_rule, current_rule, current_page, patterns,
                    profile=profile
                )
                chunks.append(chunk)
                accumulated_lines = accumulated_lines[-2:]

    # Final chunk
    if accumulated_lines and len(" ".join(accumulated_lines)) >= 50:
        chunk = _build_chunk_generic(
            accumulated_lines, source_file, doc_type, priority,
            current_section, current_rule, current_rule, current_page, patterns,
            profile=profile
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
                    current_page, patterns, scenario=current_scenario,
                    profile=profile
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
                scenario=current_scenario,
                profile=profile
            )
            chunks.append(chunk)
            accumulated_lines = accumulated_lines[-2:]

    # Final chunk
    if accumulated_lines and len(" ".join(accumulated_lines)) >= 50:
        chunk = _build_chunk_generic(
            accumulated_lines, source_file, doc_type, priority,
            current_section, current_scenario, None, current_page, patterns,
            scenario=current_scenario,
            profile=profile
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
                current_section, None, None, current_page, patterns,
                profile=profile
            )
            chunks.append(chunk)
            accumulated_lines = accumulated_lines[-2:]

    if accumulated_lines and len(" ".join(accumulated_lines)) >= 50:
        chunk = _build_chunk_generic(
            accumulated_lines, source_file, doc_type, priority,
            current_section, None, None, current_page, patterns,
            profile=profile
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


def _find_split_point(lines):
    """Find a natural split point (e.g. empty line or sentence boundary) in accumulated lines."""
    total = len(lines)
    if total <= 2:
        return total
    for i in range(total - 1, max(1, total // 3), -1):
        line = lines[i].strip()
        if not line or line.endswith(('.', ':', ';', '!')):
            return i + 1
    return total // 2


def is_valid_keyword_header(stripped: str, patterns=None) -> bool:
    """Validate whether a stripped text line represents a genuine section header or rule boundary."""
    if not stripped or len(stripped) < 3 or len(stripped) > 65:
        return False
    if stripped.endswith(('.', ';', ',', ':', '!', '?', '-', '—')):
        return False

    words = stripped.split()
    if len(words) > 8:
        return False

    low = stripped.lower()
    if any(stop in low for stop in ["page ", "continue", "example", "table of", "copyright", "all rights reserved"]):
        return False

    # 1. Decimal or alphanumeric section header: e.g. "4. Armor Penetration", "3.2 Formations", "A. Action Chit Phase", "1. Initiative"
    if re.match(r'^(?:[0-9O]{1,2}\.[0-9O\.]*|[A-Z]\.)\s+[A-Z]', stripped):
        return True

    # 2. ALL CAPS header: e.g. "FORMATION SPEED", "TORPEDO FIRING ARCS", "DAMAGE LEVELS", "MISSILE COMBAT"
    if re.match(r'^[A-Z0-9\s&\-\'\/\(\)]{3,60}$', stripped) and any(c.isalpha() for c in stripped):
        return True

    # 3. Title Cased header: e.g. "Firing Resolution", "Brigade Morale", "Movement Orders", "Sequence of Missile Combat"
    lower_noise_words = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "by", "from"}
    if words[0][0].isupper():
        title_case_match = True
        for w in words[1:]:
            w_clean = re.sub(r'[^a-zA-Z]', '', w)
            if not w_clean:
                continue
            if w_clean.lower() in lower_noise_words:
                continue
            if not w_clean[0].isupper():
                title_case_match = False
                break
        if title_case_match:
            return True

    return False


def chunk_keyword_header(text, source_file, doc_type, priority, profile, patterns):
    """
    Chunker for domains with visual/keyword headers, numbered clauses,
    inline bold subheadings, and hierarchical macro-section chapters.
    """
    chunks = []
    # --- OCR normalisation (RL scans have spaced letters in headers) ---
    text = normalise_ocr_spaced_text(text)

    lines = text.split("\n")

    current_macro_section = None
    current_macro_prefix = None
    current_header = None
    current_rule = None
    current_section = doc_type.replace("_", " ").title()
    accumulated_lines = []
    current_page = 1

    edition = profile.get("edition")

    max_chunk_chars = profile.get("chunk_size", 2500)
    min_chunk_chars = profile.get("min_chunk_size", 60)

    inline_heading_pattern = re.compile(r'^([A-Z][A-Za-z0-9\s&\-\']{2,45})[.:]\s+([A-Z].*)')
    numbered_clause_pattern = re.compile(r'^\s*(\d{1,2}\.)\s+([A-Z][A-Za-z0-9\s&\-\'\(\)\/\,]{2,60})')

    for line in lines:
        page_match = re.match(r"--- PAGE (\d+)", line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        stripped = line.strip()
        if not stripped:
            continue

        # Check macro chapter / section (e.g. "COMBAT", "MISSILE COMBAT", "VEHICLE COMBAT", "EQUIPMENT")
        if re.match(r'^[A-Z\s]{4,40}$', stripped) and len(stripped.split()) <= 4 and any(c.isalpha() for c in stripped):
            current_macro_section = stripped.title()

        # Check inline heading (e.g. "Penetration Resistance. The armor...", "Controlled Bursts: Deliver a...")
        m_inline = inline_heading_pattern.match(stripped)
        m_num = numbered_clause_pattern.match(stripped)

        if m_inline and not is_valid_keyword_header(stripped, patterns):
            heading_title = m_inline.group(1).strip()
            if accumulated_lines and len(" ".join(accumulated_lines)) >= min_chunk_chars:
                chunk_text = parse_datasheet(accumulated_lines) if "DATASHEET" in (current_header or "").upper() else " ".join(accumulated_lines)
                chunk = _build_chunk_generic(
                    [chunk_text], source_file, doc_type, priority,
                    current_section, current_rule, current_rule, current_page, patterns,
                    profile=profile
                )
                if current_macro_section:
                    chunk["metadata"]["chapter"] = current_macro_section
                    chunk["metadata"]["root_section"] = current_macro_section
                    chunk["metadata"]["breadcrumbs"] = [current_macro_section, current_header] if current_header else [current_macro_section]
                if edition:
                    chunk["metadata"]["edition"] = edition
                chunks.append(chunk)
                accumulated_lines = []

            current_header = heading_title
            current_rule = f"{current_macro_section} > {heading_title}" if current_macro_section else heading_title
            current_section = current_rule
            accumulated_lines.append(stripped)
            continue

        if m_num or is_valid_keyword_header(stripped, patterns):
            heading_text = stripped
            m_sec = re.match(r'^(\d{1,2}\.\d{1,2})\s+(.*)', stripped)
            if m_sec:
                current_macro_prefix = m_sec.group(1)
                current_macro_section = stripped
                effective_rule = stripped
                effective_section = stripped
            elif current_macro_prefix:
                effective_rule = f"{current_macro_prefix} [{stripped}]"
                effective_section = f"{current_macro_section} > {stripped}" if current_macro_section else stripped
            elif current_macro_section and current_macro_section != stripped.title():
                effective_rule = f"{current_macro_section} > {stripped}"
                effective_section = f"{current_macro_section} > {stripped}"
            else:
                effective_rule = stripped
                effective_section = stripped

            if accumulated_lines and len(" ".join(accumulated_lines)) >= min_chunk_chars:
                chunk_text = parse_datasheet(accumulated_lines) if "DATASHEET" in (current_header or "").upper() else " ".join(accumulated_lines)
                chunk = _build_chunk_generic(
                    [chunk_text], source_file, doc_type, priority,
                    current_section, current_rule, current_rule,
                    current_page, patterns,
                    profile=profile
                )
                if current_macro_prefix or current_macro_section:
                    root_name = current_macro_section or f"Section {current_macro_prefix}"
                    chunk["metadata"]["chapter"] = root_name
                    chunk["metadata"]["root_section"] = root_name
                    chunk["metadata"]["breadcrumbs"] = [root_name, current_header] if current_header else [root_name]
                if edition:
                    chunk["metadata"]["edition"] = edition
                chunks.append(chunk)
                accumulated_lines = []

            current_header = heading_text
            current_rule = effective_rule
            current_section = effective_section

        accumulated_lines.append(stripped)

        # Force split on very large chunks
        if len(" ".join(accumulated_lines)) > max_chunk_chars:
            split_pt = _find_split_point(accumulated_lines)
            if split_pt > 0 and split_pt < len(accumulated_lines):
                chunk = _build_chunk_generic(
                    accumulated_lines[:split_pt], source_file, doc_type, priority,
                    current_section, current_rule, current_rule, current_page, patterns,
                    profile=profile
                )
                if current_macro_prefix or current_macro_section:
                    root_name = current_macro_section or f"Section {current_macro_prefix}"
                    chunk["metadata"]["chapter"] = root_name
                    chunk["metadata"]["root_section"] = root_name
                    chunk["metadata"]["breadcrumbs"] = [root_name, current_header] if current_header else [root_name]
                if edition:
                    chunk["metadata"]["edition"] = edition
                chunks.append(chunk)
                accumulated_lines = accumulated_lines[max(0, split_pt - 2):]
            else:
                chunk = _build_chunk_generic(
                    accumulated_lines, source_file, doc_type, priority,
                    current_section, current_rule, current_rule, current_page, patterns,
                    profile=profile
                )
                if current_macro_prefix or current_macro_section:
                    root_name = current_macro_section or f"Section {current_macro_prefix}"
                    chunk["metadata"]["chapter"] = root_name
                    chunk["metadata"]["root_section"] = root_name
                    chunk["metadata"]["breadcrumbs"] = [root_name, current_header] if current_header else [root_name]
                if edition:
                    chunk["metadata"]["edition"] = edition
                chunks.append(chunk)
                accumulated_lines = accumulated_lines[-2:]

    # Final chunk
    if accumulated_lines and len(" ".join(accumulated_lines)) >= min_chunk_chars:
        chunk = _build_chunk_generic(
            accumulated_lines, source_file, doc_type, priority,
            current_section, current_rule, current_rule, current_page, patterns,
            profile=profile
        )
        if current_macro_prefix or current_macro_section:
            root_name = current_macro_section or f"Section {current_macro_prefix}"
            chunk["metadata"]["chapter"] = root_name
            chunk["metadata"]["root_section"] = root_name
            chunk["metadata"]["breadcrumbs"] = [root_name, current_header] if current_header else [root_name]
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

    # For OCR-scanned docs, normalize OCR noise before chunking
    if profile.get("ocr_normalize") or doc_info.get("ocr_normalize"):
        # Normalize rule numbers like 3.O -> 3.0, 1.O -> 1.0
        text = re.sub(r'(?<=\d\.)O\b', '0', text)

    # Apply profile-defined regex substitutions
    ocr_regex_subs = profile.get("ocr_regex_substitutions") or doc_info.get("ocr_regex_substitutions")
    if ocr_regex_subs and isinstance(ocr_regex_subs, list):
        for entry in ocr_regex_subs:
            pat = entry.get("pattern")
            rep = entry.get("replacement", "")
            if pat:
                text = re.sub(pat, rep, text)

    # Apply profile-defined literal substitutions
    ocr_subs = profile.get("ocr_substitutions") or doc_info.get("ocr_substitutions")
    if ocr_subs and isinstance(ocr_subs, dict):
        for bad, good in ocr_subs.items():
            if len(bad) > 0:
                text = text.replace(bad, good)

    rule_chunk_types = {
        "core_rules", "core_rules_v1", "errata", "scenario_errata",
        "integrated_rules", "version_tracker", "codex"
    }
    scenario_types = {"scenarios", "scenario_balance", "tournament"}
    generic_types = {"qa", "journal", "variant", "primer", "supplement", "unknown"}

    rule_schema = profile.get("rule_schema", "unknown")

    if rule_schema == "keyword_header" and (doc_type in rule_chunk_types or doc_type in {"unknown", "supplement", "primer"} or len(profile.get("documents", {})) == 1):
        return chunk_keyword_header(text, source_file, doc_type, priority, profile, patterns)
    elif doc_type in rule_chunk_types:
        return chunk_rules_generic(text, source_file, doc_type, priority, patterns, profile=profile)
    elif doc_type in scenario_types:
        return chunk_scenarios_generic(text, source_file, doc_type, priority, profile, patterns)
    else:
        return chunk_generic_with_profile(text, source_file, doc_type, priority, profile, patterns)


def get_text_for_game_file(fname, profile):
    """
    Get text for a game file using profile-specified paths.
    Tries pre-processed text cache first, falls back to native PDF extraction.
    Respects max_pages setting from profile.

    When profile (or document) sets ocr_required=True, uses the OCR pipeline.
    When profile (or document) sets extract_tables=True, injects Markdown tables.
    """
    text_dir = profile.get("text_dir", "data/generic_text")
    data_dir = profile["data_dir"]
    doc_info = profile["documents"].get(fname, {})
    max_pages = doc_info.get("max_pages")

    # Resolve OCR and table flags — document-level overrides profile-level
    ocr_required = doc_info.get("ocr_required", False) or profile.get("ocr_required", False)
    extract_tables = doc_info.get("extract_tables", False) or profile.get("extract_tables", False)
    ocr_substitutions = profile.get("ocr_substitutions") or doc_info.get("ocr_substitutions")
    ocr_regex_substitutions = profile.get("ocr_regex_substitutions") or doc_info.get("ocr_regex_substitutions")

    # max_pages == 0 means skip this file
    if max_pages == 0:
        return None

    # Check text cache first (pre-OCR'd text files win over live extraction)
    txt_name = fname.replace(".pdf", ".txt")
    candidate_paths = [
        os.path.join(text_dir, txt_name),
        os.path.join("/tmp", text_dir, txt_name),
        os.path.join("/tmp", txt_name),
        os.path.join("/tmp/data", f"{profile.get('game_id', '')}_text", txt_name),
        os.path.join("/tmp/data", f"{profile.get('name', '').lower().replace(' ', '_')}_text", txt_name)
    ]
    for cp in candidate_paths:
        if os.path.exists(cp):
            with open(cp, "r", encoding="utf-8") as f:
                return f.read()

    # Fall back to native PDF extraction
    pdf_path = os.path.join(data_dir, fname)
    if not os.path.exists(pdf_path):
        return None

    # ── OCR pass for scanned / bitmap documents ──────────────────────────────
    if ocr_required:
        try:
            from engine.ingestion.ocr_processor import init_ocr, process_pdf_to_text
            ocr_engine = init_ocr()
            text = process_pdf_to_text(
                pdf_path, ocr_engine,
                max_pages=max_pages,
                ocr_substitutions=ocr_substitutions,
                ocr_regex_substitutions=ocr_regex_substitutions,
            )
            if not text:
                print(f"  [OCR Warning] OCR returned empty text for {fname}. Falling back to native.")
            else:
                if extract_tables:
                    try:
                        from engine.ingestion.table_extractor import inject_markdown_tables
                        text = inject_markdown_tables(pdf_path, text, max_pages=max_pages)
                    except ImportError:
                        pass  # table_extractor not yet built — silent no-op
                return text
        except Exception as e:
            print(f"  [OCR Warning] OCR pipeline failed for {fname}: {e}. Falling back to native.")

    # ── Native PyMuPDF extraction (digital PDFs) ─────────────────────────────
    try:
        doc = fitz.open(pdf_path)
        pages = []
        limit = max_pages if max_pages else len(doc)

        for i in range(min(limit, len(doc))):
            try:
                page_text = doc[i].get_text().strip()

                # Inject Markdown table representation alongside the page text
                if extract_tables:
                    try:
                        from engine.ingestion.table_extractor import extract_page_tables_as_markdown
                        tables_md = extract_page_tables_as_markdown(doc[i])
                        if tables_md:
                            page_text = page_text + "\n\n" + tables_md
                    except ImportError:
                        pass  # table_extractor not yet built — silent no-op

                if page_text:
                    pages.append(f"--- PAGE {i + 1} ---\n{page_text}")
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
    import chromadb
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


def save_cloud_knowledge_and_vector_index(
    title_id: str,
    all_chunks: Dict[str, Any],
    rule_index: Dict[str, Any],
    section_tree: Any,
    cooc_graph: Any,
    chunk_embeddings: Dict[str, List[float]],
    tenant_id: Optional[str] = None,
    table_name: Optional[str] = None,
    bucket_name: Optional[str] = None,
    region_name: str = "ap-southeast-2",
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist codified rules to DynamoDB (RagDoll-Knowledge-test) and build
    authoritative vector_index.json package uploaded to S3 bucket.
    """
    import boto3
    tenant_id = tenant_id or os.environ.get("TENANT_ID")
    if not tenant_id:
        raise ValueError("tenant_id must be provided to persist_rules_to_cloud")
    table_name = table_name or os.environ.get("KNOWLEDGE_TABLE_NAME", "RagDoll-Knowledge-test")
    bucket_name = bucket_name or os.environ.get("DOCUMENTS_BUCKET_NAME", "ragdoll-docs-test-1860")
    aws_profile = os.environ.get("AWS_PROFILE")

    if aws_profile:
        session = boto3.Session(profile_name=aws_profile, region_name=region_name)
        ddb = session.resource("dynamodb")
        s3 = session.client("s3")
    else:
        ddb = boto3.resource("dynamodb", region_name=region_name)
        s3 = boto3.client("s3", region_name=region_name)

    table = ddb.Table(table_name)
    print(f"\n[Cloud Ingest] Target DynamoDB Table: {table_name}")
    print(f"[Cloud Ingest] Target S3 Bucket: {bucket_name}")

    # 1. Collate chunks by rule number
    rules_dict: Dict[str, Dict[str, Any]] = {}
    rule_vectors: Dict[str, List[float]] = {}

    for r_num, entries in rule_index.items():
        if r_num.startswith("__"):
            continue
        c_ids = [e["chunk_id"] for e in entries if e.get("chunk_id") in all_chunks]
        if not c_ids:
            continue

        verbatim_parts = []
        source_files = set()
        title = f"Rule {r_num}"
        chapter = ""
        breadcrumbs = []
        cross_refs = []
        priority = 9

        for cid in c_ids:
            chk = all_chunks[cid]
            verbatim_parts.append(chk.get("text", ""))
            meta = chk.get("metadata", {})
            if meta.get("source_file"):
                source_files.add(meta["source_file"])
            if meta.get("title") and title == f"Rule {r_num}":
                title = meta["title"]
            if meta.get("chapter") or meta.get("root_section"):
                chapter = meta.get("chapter") or meta.get("root_section")
            if meta.get("priority") and int(meta["priority"]) < priority:
                priority = int(meta["priority"])
            for cr in chk.get("cross_refs", []):
                if cr not in cross_refs:
                    cross_refs.append(cr)
            if meta.get("breadcrumbs"):
                breadcrumbs = meta["breadcrumbs"]

        # Compute vector for rule
        vecs = [chunk_embeddings[cid] for cid in c_ids if cid in chunk_embeddings and chunk_embeddings[cid]]
        if vecs:
            dim = len(vecs[0])
            avg_vec = [0.0] * dim
            for v in vecs:
                for d in range(dim):
                    avg_vec[d] += v[d]
            norm = sum(x * x for x in avg_vec) ** 0.5
            if norm > 0:
                avg_vec = [round(x / norm, 6) for x in avg_vec]
            rule_vectors[r_num] = avg_vec

        # Get siblings from section tree if available
        siblings = []
        if section_tree:
            try:
                s_list = section_tree.get_sibling_rules(r_num)
                siblings = [{"number": s, "title": f"Rule {s}"} for s in s_list[:4]]
            except Exception:
                pass

        rule_item = {
            "PK": f"TITLE#{title_id}",
            "SK": f"RULE#{r_num}",
            "tenantId": tenant_id,
            "titleId": title_id,
            "ruleNumber": r_num,
            "title": title,
            "verbatimText": "\n\n---\n\n".join(verbatim_parts),
            "chapter": chapter or f"Section {r_num.split('.')[0]}.0",
            "priority": str(priority),
            "crossReferences": [{"number": cr, "title": f"Rule {cr}", "weight": "0.75"} for cr in cross_refs],
            "breadcrumbs": breadcrumbs or ["Rules Reference", f"Rule {r_num}"],
            "siblings": siblings,
            "sourceFiles": sorted(list(source_files)),
            "updatedAt": datetime.utcnow().isoformat() + "Z",
        }
        rules_dict[r_num] = rule_item

    # Fallback if rules_dict is empty (e.g. general narrative or keyword_header rules without explicit decimal IDs)
    if not rules_dict and all_chunks:
        print(f"[Cloud Ingest] Rule index empty — indexing {len(all_chunks)} chunks by section/chunk ID...")
        for cid, chk in all_chunks.items():
            meta = chk.get("metadata", {})
            r_num = chk.get("rule_number") or meta.get("section_path") or meta.get("subsection") or cid
            r_num_clean = re.sub(r'[\#\/]', '_', str(r_num).strip())[:80] or cid

            title = meta.get("title") or meta.get("section_path") or f"Section {r_num_clean}"
            chapter = meta.get("chapter") or meta.get("root_section") or "General Rules"
            verbatim = chk.get("text", "")

            vec = chunk_embeddings.get(cid)
            if vec:
                rule_vectors[r_num_clean] = vec

            rule_item = {
                "PK": f"TITLE#{title_id}",
                "SK": f"RULE#{r_num_clean}",
                "tenantId": tenant_id,
                "titleId": title_id,
                "ruleNumber": r_num_clean,
                "title": title,
                "verbatimText": verbatim,
                "chapter": chapter,
                "priority": str(meta.get("priority", 9)),
                "crossReferences": [],
                "breadcrumbs": meta.get("breadcrumbs") or ["Rules Reference", title],
                "siblings": [],
                "sourceFiles": [meta.get("source_file", "document.pdf")],
                "updatedAt": datetime.utcnow().isoformat() + "Z",
            }
            rules_dict[r_num_clean] = rule_item

    # 2. Batch write to DynamoDB
    print(f"[Cloud Ingest] Writing {len(rules_dict)} codified rules to DynamoDB...")
    with table.batch_writer() as batch:
        for r_num, item in rules_dict.items():
            batch.put_item(Item=item)
    print(f"[Cloud Ingest] DynamoDB write completed.")

    # 3. Build inverted keyword and title indexes
    title_index: Dict[str, List[str]] = {}
    keyword_index: Dict[str, List[str]] = {}

    stopwords = {
        "what", "is", "the", "and", "or", "of", "to", "in", "a", "for", "with", "on",
        "at", "by", "from", "under", "across", "vs", "versus", "compare", "does", "do",
        "how", "which", "who", "when", "where", "can", "could", "would", "should",
        "rule", "rules", "about", "tell", "me", "explain", "describe", "say", "says",
        "that", "this", "there", "are", "was", "were", "been", "have", "has", "had",
        "any", "all", "some", "an", "my", "its", "if", "then", "than", "into", "also",
        "between", "during", "after", "before", "section", "game", "play", "player"
    }

    for r_num, item in rules_dict.items():
        # Title words
        t_words = re.findall(r'\b[a-zA-Z]{3,}\b', item["title"].lower())
        for tw in t_words:
            if tw not in stopwords:
                title_index.setdefault(tw, []).append(r_num)

        # Body words
        b_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', item["verbatimText"][:800].lower()))
        for bw in b_words:
            if bw not in stopwords and bw not in t_words:
                keyword_index.setdefault(bw, []).append(r_num)

    # 4. Assemble vector_index.json package
    rules_meta = {
        r_num: {
            "title": item["title"],
            "chapter": item["chapter"],
            "crossReferences": [x["number"] for x in item["crossReferences"]],
        }
        for r_num, item in rules_dict.items()
    }

    vector_index_data = {
        "version": "2.0",
        "titleId": title_id,
        "modelId": "amazon.titan-embed-text-v2:0",
        "dimensions": len(next(iter(rule_vectors.values()))) if rule_vectors else 1024,
        "ruleCount": len(rules_dict),
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "vectors": rule_vectors,
        "titleIndex": title_index,
        "keywordIndex": keyword_index,
        "rulesMeta": rules_meta,
        "sectionTree": section_tree.model_dump() if hasattr(section_tree, "model_dump") else {},
        "cooccurrenceGraph": cooc_graph.model_dump() if hasattr(cooc_graph, "model_dump") else {},
    }

    # 5. Upload to S3
    s3_key = f"tenants/{tenant_id}/titles/{title_id}/vector_index.json"
    print(f"[Cloud Ingest] Uploading vector index package to S3: s3://{bucket_name}/{s3_key}...")
    try:
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json.dumps(vector_index_data),
            ContentType="application/json",
        )
        print(f"[Cloud Ingest] Successfully uploaded vector index to S3.")
    except Exception as e:
        print(f"[Cloud Ingest Warning] S3 upload error: {e}")

    # Upload dynamic metadata profile to S3
    prof_s3_key = f"tenants/{tenant_id}/titles/{title_id}/profile.json"
    if profile:
        try:
            prof_payload = profile.model_dump() if hasattr(profile, "model_dump") else profile
            s3.put_object(
                Bucket=bucket_name,
                Key=prof_s3_key,
                Body=json.dumps(prof_payload, indent=2, default=str),
                ContentType="application/json",
            )
            print(f"[Cloud Ingest] Successfully uploaded dynamic profile to S3: s3://{bucket_name}/{prof_s3_key}")
        except Exception as pe:
            print(f"[Cloud Ingest Warning] S3 profile upload error: {pe}")

    # Write SectionTree and Profile to DynamoDB for fast UI/agent lookup
    try:
        if hasattr(section_tree, "model_dump"):
            tree_data = section_tree.model_dump()
            table.put_item(Item={
                "PK": f"TITLE#{title_id}",
                "SK": "METADATA#SECTION_TREE",
                "tenantId": tenant_id,
                "titleId": title_id,
                "sections": list(tree_data.get("sections", {}).values()),
                "game_id": tree_data.get("game_id", title_id),
                "updatedAt": datetime.utcnow().isoformat() + "Z"
            })
    except Exception as ddb_sec_err:
        print(f"[Cloud Ingest Warning] DynamoDB SectionTree write error: {ddb_sec_err}")

    if profile:
        try:
            from decimal import Decimal
            prof_payload = profile.model_dump() if hasattr(profile, "model_dump") else profile
            raw_docs = prof_payload.get("documents", {})
            clean_docs = json.loads(json.dumps(raw_docs), parse_float=Decimal)
            table.put_item(Item={
                "PK": f"TITLE#{title_id}",
                "SK": "METADATA#PROFILE",
                "tenantId": tenant_id,
                "titleId": title_id,
                "rule_schema": prof_payload.get("rule_schema"),
                "scenario_format": prof_payload.get("scenario_format"),
                "documents": clean_docs,
                "updatedAt": datetime.utcnow().isoformat() + "Z"
            })
        except Exception as ddb_prof_err:
            print(f"[Cloud Ingest Warning] DynamoDB Profile write error: {ddb_prof_err}")


    # Also save locally into backend/assets if directory exists
    local_assets_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            f"../../../gaiia-rag-doll-cloud/backend/assets/titles/{title_id}"
        )
    )
    if os.path.exists(os.path.dirname(local_assets_dir)):
        try:
            os.makedirs(local_assets_dir, exist_ok=True)
            local_vidx_path = os.path.join(local_assets_dir, "vector_index.json")
            with open(local_vidx_path, "w", encoding="utf-8") as f:
                json.dump(vector_index_data, f)
            print(f"[Cloud Ingest] Saved local asset vector index: {local_vidx_path}")
        except Exception:
            pass

    return {
        "titleId": title_id,
        "ruleCount": len(rules_dict),
        "vectorCount": len(rule_vectors),
        "s3Key": s3_key,
        "profileKey": prof_s3_key,
    }



def ingest_game(
    profile_path: str,
    dry_run: bool = False,
    target: Optional[str] = None,
    llm_provider: Optional[BaseLLMProvider] = None,
    storage_provider: Optional[BaseStorageProvider] = None,
    title_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generic ingestion pipeline driven by a game profile JSON.
    Supports both Local (ChromaDB + JSON) and Cloud (DynamoDB + S3 vector index).
    """
    profile = load_profile(profile_path)
    patterns = get_compiled_patterns(profile)

    llm = llm_provider or get_llm_provider(target)
    storage = storage_provider or get_storage_provider(target)
    is_cloud = (
        (target == "cloud")
        or os.environ.get("TARGET") == "cloud"
        or isinstance(storage, CloudDynamoStorageProvider)
    )

    game_name = profile.get("game_name", "Game")
    game_id = profile.get("game_id", "generic")
    effective_title_id = title_id or (
        "up-front-core" if ("up" in game_id.lower() and "front" in game_id.lower()) else game_id
    )
    rule_index_file = profile.get("rule_index_file", f"data/{effective_title_id}_rule_index.json")

    print(f"\n{'=' * 70}")
    print(f"GENERIC INGESTION PIPELINE — {game_name} (Target: {'CLOUD' if is_cloud else 'LOCAL'})")
    print(f"Profile: {profile_path}")
    print(f"Schema: {profile.get('rule_schema', 'unknown')}")
    print(f"{'=' * 70}")

    if not dry_run and not is_cloud:
        collection = setup_vector_db_for_game(profile)
    else:
        collection = None

    all_chunks = {}
    chunk_embeddings = {}
    total_indexed = 0

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

        batch_ids, batch_embeddings, batch_metas, batch_docs = [], [], [], []

        if is_cloud:
            from concurrent.futures import ThreadPoolExecutor
            def _embed_chunk_worker(item):
                c_idx, chk = item
                cid = f"{doc_type}_{re.sub(r'[^a-z0-9]', '_', fname.lower()[:40])}_chunk_{c_idx}"
                try:
                    emb = llm.embed(chk["text"])
                    return cid, chk, emb
                except Exception as ex:
                    print(f"     ERROR embedding chunk {c_idx}: {ex}")
                    return cid, chk, None

            with ThreadPoolExecutor(max_workers=16) as executor:
                embed_results = list(executor.map(_embed_chunk_worker, list(enumerate(chunks, 1))))

            for cid, chk, emb in embed_results:
                all_chunks[cid] = chk
                if emb:
                    chunk_embeddings[cid] = emb
        else:
            for chunk_idx, chunk in enumerate(chunks, 1):
                chunk_id = (
                    f"{doc_type}_{re.sub(r'[^a-z0-9]', '_', fname.lower()[:40])}"
                    f"_chunk_{chunk_idx}"
                )

                try:
                    embedding = llm.embed(chunk["text"])
                    chunk_embeddings[chunk_id] = embedding
                except Exception as e:
                    print(f"     ERROR embedding chunk {chunk_idx}: {e}")
                    continue

                all_chunks[chunk_id] = chunk

                if collection:
                    batch_ids.append(chunk_id)
                    batch_embeddings.append(embedding)
                    batch_metas.append(chunk["metadata"])
                    batch_docs.append(chunk["text"])
                    if len(batch_ids) >= 20:
                        collection.upsert(
                            ids=batch_ids,
                            embeddings=batch_embeddings,
                            metadatas=batch_metas,
                            documents=batch_docs
                        )
                        batch_ids, batch_embeddings, batch_metas, batch_docs = [], [], [], []

            if collection and batch_ids:
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metas,
                    documents=batch_docs
                )

        total_indexed += len(chunks)
        print(f"     Indexed: {len(chunks)} chunks")

    if not dry_run:
        print(f"\n{'=' * 70}")
        print("Building rule-number lookup index & hierarchy tree...")
        rule_index = build_rule_index(all_chunks)
        
        glossary = profile.get("glossary", {})
        section_tree = build_section_tree(all_chunks, game_id=title_id)
        cooc_graph = build_ingestion_cooccurrence_graph(all_chunks, rule_index, glossary=glossary, section_tree=section_tree, game_id=title_id)

        if is_cloud:
            result = save_cloud_knowledge_and_vector_index(
                title_id=effective_title_id,
                all_chunks=all_chunks,
                rule_index=rule_index,
                section_tree=section_tree,
                cooc_graph=cooc_graph,
                chunk_embeddings=chunk_embeddings,
                tenant_id=tenant_id,
                profile=profile,
            )
            print(f"\n{'=' * 70}")
            print(f"CLOUD INGESTION COMPLETE — {game_name}")
            print(f"  Total chunks: {total_indexed}")
            print(f"  Codified rules: {result.get('ruleCount', 0)}")
            print(f"  S3 Vector Index: {result.get('s3Key', '')}")
            return result
        else:
            rule_index["__section_tree__"] = section_tree.model_dump()
            with open(rule_index_file, "w", encoding="utf-8") as f:
                json.dump(rule_index, f, indent=2)

            cooc_path = profile.get("cooccurrence_graph_file")
            if not cooc_path:
                cooc_path = rule_index_file.replace("_rule_index.json", "_cooccurrence_graph.json")
                if cooc_path == rule_index_file:
                    cooc_path = f"data/{game_id}_cooccurrence_graph.json"

            sec_tree_path = rule_index_file.replace("_rule_index.json", "_section_tree.json")
            if sec_tree_path == rule_index_file:
                sec_tree_path = f"data/{game_id}_section_tree.json"

            cooc_graph.save_json(cooc_path)
            section_tree.save_json(sec_tree_path)

            print(f"  Rule numbers indexed: {len(rule_index) - 1}")
            print(f"  Section nodes indexed: {len(section_tree.sections)}")
            print(f"  Co-occurrence graph edges: {sum(len(edges) for edges in cooc_graph.adjacency.values())}")
            print(f"  Saved rule index: {rule_index_file}")
            print(f"  Saved co-occurrence graph: {cooc_path}")
            print(f"  Saved section tree: {sec_tree_path}")

            print(f"\n{'=' * 70}")
            print(f"INGESTION COMPLETE — {game_name}")
            print(f"  Total chunks: {total_indexed}")
            print(f"  Unique rules: {len(rule_index) - 1}")
            print(f"  Collection:   {profile['chroma_collection']}")
            return {
                "titleId": title_id,
                "ruleCount": len(rule_index) - 1,
                "totalChunks": total_indexed,
            }
    else:
        print(f"\n{'=' * 70}")
        print(f"DRY RUN COMPLETE — {game_name}")
        print(f"  Chunks parsed: {total_indexed}")
        return {"totalChunks": total_indexed}


def process_rules_pdf(
    file_path,
    target=None,
    game_id=None,
    profile_path=None,
    llm_provider=None,
    storage_provider=None,
    tenant_id=None,
    ocr_required=False,
    pre_extracted_text=None,
    pre_extracted_text_path=None,
):
    """
    Process and ingest a rulebook PDF using dynamic profiling.
    1. If profile_path is not provided, dynamically generate one using DynamicProfileGenerator.
    2. Support pre-extracted OCR text to bypass Stage 1 re-OCR in decoupled pipelines.
    3. Ingest the PDF using run_game_ingestion().
    """
    from engine.ingestion.dynamic_profiler import DynamicProfileGenerator

    fname = os.path.basename(file_path)
    effective_id = game_id or re.sub(
        r"[^a-z0-9]", "_", os.path.splitext(fname)[0].lower()
    )

    # If pre-extracted text is provided, persist it to cache BEFORE dynamic profiling
    if pre_extracted_text or pre_extracted_text_path:
        cache_dirs = [
            f"/tmp/data/{effective_id}_text",
            f"data/{effective_id}_text",
            "/tmp"
        ]
        for cdir in cache_dirs:
            try:
                os.makedirs(cdir, exist_ok=True)
                txt_out = os.path.join(cdir, fname.replace(".pdf", ".txt"))
                if pre_extracted_text:
                    with open(txt_out, "w", encoding="utf-8") as tf:
                        tf.write(pre_extracted_text)
                elif pre_extracted_text_path and os.path.exists(pre_extracted_text_path):
                    with open(pre_extracted_text_path, "r", encoding="utf-8") as src_tf, open(txt_out, "w", encoding="utf-8") as dst_tf:
                        dst_tf.write(src_tf.read())
            except Exception:
                pass
        print(f"[Stage 2 Fast-Path] Loaded pre-extracted text into cache for: {fname}")

    if not profile_path or not os.path.exists(profile_path):
        gen = DynamicProfileGenerator(use_llm=False)
        if (
            target == "cloud"
            or os.environ.get("TARGET") == "cloud"
            or file_path.startswith("/tmp")
        ):
            temp_profile_path = f"/tmp/{effective_id}_profile.json"
        else:
            temp_profile_path = f"data/{effective_id}_profile.json"

        prof = gen.generate_profile(
            target_path=file_path,
            game_id=effective_id,
            output_path=temp_profile_path,
            save=False,
        )
        if hasattr(prof, "model_dump"):
            prof_dict = prof.model_dump()
        else:
            prof_dict = prof

        if target == "cloud" or file_path.startswith("/tmp"):
            prof_dict["text_dir"] = f"/tmp/data/{effective_id}_text"

        if ocr_required and not pre_extracted_text and not pre_extracted_text_path:
            prof_dict["ocr_required"] = True
            for d in prof_dict.get("documents", {}).values():
                d["ocr_required"] = True

        os.makedirs(os.path.dirname(temp_profile_path), exist_ok=True)
        with open(temp_profile_path, "w", encoding="utf-8") as f:
            json.dump(prof_dict, f, indent=2, default=str)

        profile_path = temp_profile_path

    return ingest_game(
        profile_path=profile_path,
        target=target,
        llm_provider=llm_provider,
        storage_provider=storage_provider,
        title_id=game_id or effective_id,
        tenant_id=tenant_id,
    )



# Alias for backward compatibility
run_game_ingestion = ingest_game

