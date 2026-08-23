"""
Domain Profile Meta-Contract Specification — Gaiia RAG Doll.

Provides strict, validated Pydantic models for declarative domain configurations.
Compatible with legacy game profiles (Up Front, ASL, Renegade Legion, SFB, 40K, Home Insurance)
and new generic multi-modal domains (Visual Media, Technical Schematics, Legal Policies).
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator


class DocumentMetadata(BaseModel):
    """Metadata describing a single file in the corpus."""
    model_config = ConfigDict(extra="allow")

    doc_type: str = Field(default="unknown", description="Categorical type (e.g. core_rules, errata, pictorial, pds)")
    priority: int = Field(default=9, description="Temporal / authority priority (1=highest, 9=lowest)")
    description: Optional[str] = Field(default=None, description="Human-readable description of document role")
    size_mb: Optional[float] = Field(default=None, description="File size in megabytes")
    total_pages: Optional[int] = Field(default=None, description="Total number of pages in document")
    max_pages: Optional[int] = Field(default=None, description="Max pages to ingest (0=skip, null=all)")
    edition: Optional[str] = Field(default=None, description="Edition label e.g. 1st, 2nd, 2006")
    supersedes: Optional[List[str]] = Field(default_factory=list, description="Filenames superseded by this document")
    carrier: Optional[str] = Field(default=None, description="Brand, carrier, or publisher name")


class ParsingGrammar(BaseModel):
    """Text parsing and chunking grammar rules."""
    model_config = ConfigDict(extra="allow")

    rule_schema: Optional[str] = Field(default=None, description="Schema identifier (e.g. chapter_decimal, keyword_header)")
    rule_pattern: Optional[str] = Field(default=None, description="Regex pattern matching primary section headers")
    cross_ref_pattern: Optional[str] = Field(default=None, description="Regex pattern matching cross-references")
    section_delimiter_regex: Optional[str] = Field(default=None, description="Optional regex splitting major chapters")
    chunk_size_chars: int = Field(default=2500, description="Max characters before splitting text chunks")


class StructuredExtractionConfig(BaseModel):
    """VLM / LLM structured extraction guidelines (e.g. for entity sheets, pictorials)."""
    model_config = ConfigDict(extra="allow")

    target_schema: Dict[str, Any] = Field(default_factory=dict, description="Target JSON schema structure")
    target_schema_file: Optional[str] = Field(default=None, description="Path to external schema JSON file")
    vlm_extraction_hints: List[str] = Field(default_factory=list, description="Specific field extraction warnings/instructions")
    spatial_regions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional bounding box guidelines")


class OntologyConfig(BaseModel):
    """Domain terms, taxonomies, and acronym dictionaries."""
    model_config = ConfigDict(extra="allow")

    glossary: Dict[str, Any] = Field(default_factory=dict, description="Acronym to expanded term mapping")
    entity_types: List[str] = Field(default_factory=list, description="Recognized entity classification types")
    synonyms: Dict[str, List[str]] = Field(default_factory=dict, description="Synonym mappings")


class AgentPersonaConfig(BaseModel):
    """Agent retrieval persona and citation configuration."""
    model_config = ConfigDict(extra="allow")

    role: str = Field(default="Reference Assistant", description="Persona role in system prompts")
    citation_format: str = Field(default="[{document}, Rule {section}]", description="Citation guideline template")
    conflict_resolution_rule: Optional[str] = Field(
        default="Higher priority and newer editions supersede older documents.",
        description="Rule for temporal arbitration"
    )
    query_intents: List[str] = Field(
        default_factory=lambda: ["direct_rule", "concept", "situation", "comparison", "variant"],
        description="Supported query classification categories"
    )
    custom_instructions: Optional[str] = Field(default=None, description="Extra domain-specific reasoning directives")


class DomainProfile(BaseModel):
    """
    Unified Meta-Contract representing a complete domain configuration for Gaiia RAG Doll.
    """
    domain_name: Optional[str] = Field(default=None, description="Full human-readable domain name")
    game_name: Optional[str] = Field(default=None, description="Legacy alias for domain_name")

    domain_id: Optional[str] = Field(default=None, description="Unique slug identifier (e.g. 'renegade_legion')")
    game_id: Optional[str] = Field(default=None, description="Legacy alias for domain_id")

    pipeline_mode: str = Field(default="RULEBOOK_TECHNICAL", description="Default pipeline mode (RULEBOOK_TECHNICAL, VISUAL_MEDIA, POLICY_HIERARCHICAL, GENERAL_TEXT)")
    
    data_dir: str = Field(default="data/generic", description="Path to raw source files")
    text_dir: Optional[str] = Field(default=None, description="Path to cached extracted text files")
    chroma_collection: str = Field(default="rag-doll-generic", description="ChromaDB collection name")
    rule_index_file: Optional[str] = Field(default=None, description="Path to exact-match JSON index")

    # Document map
    documents: Dict[str, DocumentMetadata] = Field(default_factory=dict, description="File-to-metadata registry")

    # Legacy flat fields supported directly
    rule_schema: Optional[str] = None
    rule_pattern: Optional[str] = None
    cross_ref_pattern: Optional[str] = None
    scenario_format: Optional[str] = None
    scenario_pattern: Optional[str] = None
    schema_detection_scores: Optional[Dict[str, int]] = None
    glossary: Optional[Dict[str, Any]] = None

    # Modular sub-contracts
    parsing_grammar: Optional[ParsingGrammar] = None
    structured_extraction: Optional[StructuredExtractionConfig] = None
    ontology: Optional[OntologyConfig] = None
    agent_persona: Optional[AgentPersonaConfig] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_profile_data(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        
        # Ensure either domain_name or game_name is set
        d_name = data.get("domain_name") or data.get("game_name")
        if not d_name:
            raise ValueError("DomainProfile requires either 'domain_name' or 'game_name'")
        data["domain_name"] = d_name
        data["game_name"] = d_name

        # Ensure either domain_id or game_id is set
        d_id = data.get("domain_id") or data.get("game_id")
        if not d_id:
            # Generate slug from name
            d_id = d_name.lower().replace(" ", "_").replace("-", "_")
        data["domain_id"] = d_id
        data["game_id"] = d_id

        # Normalize documents map values to DocumentMetadata-compatible dicts
        docs = data.get("documents", {})
        if isinstance(docs, dict):
            for k, v in docs.items():
                if isinstance(v, dict):
                    if "priority" not in v:
                        v["priority"] = 9
                    if "doc_type" not in v:
                        v["doc_type"] = "unknown"

        return data

    @property
    def name(self) -> str:
        return self.domain_name or self.game_name or "Unknown Domain"

    @property
    def id(self) -> str:
        return self.domain_id or self.game_id or "unknown_domain"

    def get_glossary(self) -> Dict[str, str]:
        """Return combined glossary from ontology sub-contract or top-level field, flattened to strings."""
        raw: Dict[str, Any] = {}
        if self.glossary:
            raw.update(self.glossary)
        if self.ontology and self.ontology.glossary:
            raw.update(self.ontology.glossary)

        flattened: Dict[str, str] = {}
        for k, v in raw.items():
            if isinstance(v, str):
                flattened[k] = v
            elif isinstance(v, list):
                flattened[k] = ", ".join(str(x) for x in v)
            elif isinstance(v, dict):
                # Join non-empty keys or values
                keys = [str(x) for x in v.keys() if x]
                flattened[k] = ", ".join(keys) if keys else str(v)
            else:
                flattened[k] = str(v)
        return flattened

    def get_rule_pattern(self) -> Optional[str]:
        if self.parsing_grammar and self.parsing_grammar.rule_pattern:
            return self.parsing_grammar.rule_pattern
        return self.rule_pattern

    def get_cross_ref_pattern(self) -> Optional[str]:
        if self.parsing_grammar and self.parsing_grammar.cross_ref_pattern:
            return self.parsing_grammar.cross_ref_pattern
        return self.cross_ref_pattern


def load_domain_profile(profile_path: str) -> DomainProfile:
    """Load and validate a domain profile from a JSON file."""
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Domain profile not found at {profile_path}")
    
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return DomainProfile.model_validate(data)
