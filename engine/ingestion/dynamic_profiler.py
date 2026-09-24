"""
Dynamic Profile Generator — Gaiia RAG Doll.

Autonomous profiling engine that analyzes rulebooks, technical specifications,
or legal documents (directories or single PDFs) and synthesizes a validated
DomainProfile meta-contract.
"""

import os
import sys
import json
import re
from typing import Optional, Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import fitz  # PyMuPDF

from engine.ingestion.auto_discover import (
    discover,
    detect_rule_schema,
    detect_scenario_format,
    classify_document,
    sample_pdf_text,
    KNOWN_SCHEMAS,
    SCENARIO_FORMATS,
)
from engine.models.domain_profile import DomainProfile, ParsingGrammar, AgentPersonaConfig


class DynamicProfileGenerator:
    """
    Autonomous domain profiler. Inspects documents and generates a complete
    DomainProfile meta-contract for ingestion and retrieval.
    """

    def __init__(self, use_llm: bool = False, model: str = "llama3.1:8b"):
        self.use_llm = use_llm
        self.model = model

    def inspect_outline(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract PDF bookmarks / outline entries if available.
        Returns list of dicts with level, title, page.
        """
        if not os.path.isfile(pdf_path):
            return []
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            doc.close()
            return [{"level": item[0], "title": item[1], "page": item[2]} for item in toc]
        except Exception:
            return []

    def generate_profile(
        self,
        target_path: str,
        game_name: Optional[str] = None,
        game_id: Optional[str] = None,
        output_path: Optional[str] = None,
        save: bool = True,
    ) -> DomainProfile:
        """
        Dynamically generate a DomainProfile for a directory of PDFs or a single PDF.

        Args:
            target_path: Directory path or path to a single PDF file.
            game_name: Optional human-readable name. Auto-inferred if omitted.
            game_id: Optional slug identifier. Auto-derived if omitted.
            output_path: Destination JSON path. If save=False, not written.
            save: If True, writes JSON to output_path or default path.

        Returns:
            Validated DomainProfile Pydantic instance.
        """
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target path does not exist: {target_path}")

        effective_output_path = output_path if save else False

        # Run discovery
        profile_dict = discover(
            data_dir=target_path,
            game_name=game_name,
            game_id=game_id,
            use_llm=self.use_llm,
            output_path=effective_output_path,
        )

        if not profile_dict:
            raise RuntimeError(f"Failed to generate profile for {target_path}")

        # Check outline/bookmarks on primary document
        primary_pdf = None
        if os.path.isfile(target_path):
            primary_pdf = target_path
        else:
            for fname, doc_info in profile_dict.get("documents", {}).items():
                if doc_info.get("doc_type") in ("core_rules", "supplement") and doc_info.get("priority") == 1:
                    primary_pdf = os.path.join(profile_dict.get("data_dir", target_path), fname)
                    break

        if primary_pdf:
            outline = self.inspect_outline(primary_pdf)
            if outline:
                profile_dict["outline"] = outline

        # Ensure parsing_grammar is explicitly populated
        if not profile_dict.get("parsing_grammar"):
            profile_dict["parsing_grammar"] = {
                "rule_schema": profile_dict.get("rule_schema"),
                "hierarchy_strategy": profile_dict.get("hierarchy_strategy") or ("breadcrumb_path" if profile_dict.get("rule_schema") == "keyword_header" else profile_dict.get("rule_schema", "breadcrumb_path")),
                "hierarchy_delimiter": profile_dict.get("hierarchy_delimiter", " > "),
                "rule_pattern": profile_dict.get("rule_pattern"),
                "cross_ref_pattern": profile_dict.get("cross_ref_pattern"),
            }
        else:
            if not profile_dict["parsing_grammar"].get("hierarchy_strategy"):
                profile_dict["parsing_grammar"]["hierarchy_strategy"] = profile_dict.get("hierarchy_strategy") or ("breadcrumb_path" if profile_dict.get("rule_schema") == "keyword_header" else profile_dict.get("rule_schema", "breadcrumb_path"))
            if not profile_dict["parsing_grammar"].get("hierarchy_delimiter"):
                profile_dict["parsing_grammar"]["hierarchy_delimiter"] = profile_dict.get("hierarchy_delimiter", " > ")

        # Ensure agent persona has default citation format
        if not profile_dict.get("agent_persona"):
            clean_name = profile_dict.get("domain_name") or profile_dict.get("game_name") or "Rules"
            profile_dict["agent_persona"] = {
                "role": f"{clean_name} Reference Assistant",
                "citation_format": f"[{clean_name}, Rule {{section}}]",
            }

        # Re-save updated dictionary if save was requested and output_path was provided
        if save and effective_output_path:
            with open(effective_output_path, "w", encoding="utf-8") as f:
                json.dump(profile_dict, f, indent=2)

        return DomainProfile.model_validate(profile_dict)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a dynamic DomainProfile for a rulebook corpus.")
    parser.add_argument("target", help="Path to PDF directory or single PDF file")
    parser.add_argument("--name", default=None, help="Human-readable domain/game name")
    parser.add_argument("--id", default=None, help="Short slug identifier")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM classification/glossary fallback")
    args = parser.parse_args()

    generator = DynamicProfileGenerator(use_llm=not args.no_llm)
    profile = generator.generate_profile(
        target_path=args.target,
        game_name=args.name,
        game_id=args.id,
        output_path=args.output,
        save=True,
    )

    print("\n" + "=" * 60)
    print("DYNAMIC PROFILE GENERATION SUCCESSFUL")
    print(f"Domain Name:     {profile.domain_name}")
    print(f"Domain ID:       {profile.domain_id}")
    print(f"Rule Schema:     {profile.rule_schema}")
    print(f"Rule Pattern:    {profile.rule_pattern}")
    print(f"Documents:       {len(profile.documents)}")
    for fname, meta in profile.documents.items():
        print(f"  - {fname}: {meta.doc_type} (Priority {meta.priority})")
    print("=" * 60)


if __name__ == "__main__":
    main()
