"""
Universal RAG Agent — Gaiia RAG Doll.

Unified, domain-agnostic retrieval and reasoning kernel driven entirely by
external Meta-Contracts (DomainProfile). Routes automatically to technical rule
arbitration, visual catalog search, or comparative policy analysis.
"""

import os
import json
from typing import Dict, Any, Optional, Tuple, Union

from engine.models.domain_profile import DomainProfile, load_domain_profile


class UniversalRagAgent:
    """
    Domain-agnostic agent runtime driven 100% by the loaded Meta-Contract.
    """

    def __init__(self, profile_input: Union[str, DomainProfile, Dict[str, Any]]):
        if isinstance(profile_input, DomainProfile):
            self.profile = profile_input
            self.profile_path = None
        elif isinstance(profile_input, dict):
            self.profile = DomainProfile.model_validate(profile_input)
            self.profile_path = None
        elif isinstance(profile_input, str):
            self.profile = load_domain_profile(profile_input)
            self.profile_path = profile_input
        else:
            raise TypeError(f"Invalid profile input type: {type(profile_input)}")

        self.pipeline_mode = self.profile.pipeline_mode
        self.collection_name = self.profile.chroma_collection

    def build_prompts(self) -> Tuple[str, str]:
        """Synthesize domain classification and generation prompts."""
        glossary = self.profile.get_glossary()
        glossary_text = ""
        if glossary:
            glossary_text = (
                f"Use this specific glossary to expand abbreviations in your sub-queries:\n"
                f"{json.dumps(glossary, indent=2)}\n\n"
            )

        role = self.profile.agent_persona.role if self.profile.agent_persona else "authoritative reference assistant"
        citation_fmt = self.profile.agent_persona.citation_format if self.profile.agent_persona else "[Source, Section]"
        conflict_rule = self.profile.agent_persona.conflict_resolution_rule if self.profile.agent_persona else "Superseding documents override base text."

        classify_prompt = (
            f'You are a query classifier for a reference system for the document collection "{self.profile.name}".\n'
            'Classify the user\'s query into exactly ONE category:\n\n'
            '- "direct_rule": User asks about a specific rule, clause, or section number\n'
            '- "concept": User asks about a general concept or mechanic\n'
            '- "situation": User describes a situation and wants a ruling or clarification\n'
            '- "scenario": User asks about a specific scenario, special case, or addendum\n'
            '- "comparison": User asks about errata, amendments, or changes between versions\n'
            '- "variant": User asks about unofficial, variant, or modified rules\n\n'
            'If the query is complex and requires multiple searches, provide 2-3 sub-queries.\n'
            'CRITICAL INSTRUCTION FOR ABBREVIATIONS: Expand domain-specific abbreviations in your sub-queries to include BOTH the abbreviation and the full term.\n'
            f'{glossary_text}'
            'Respond with ONLY a JSON object: '
            '{"query_type": "<type>", "rule_numbers": [<any rule/clause numbers>], '
            '"scenario": "<scenario id if any>", "sub_queries": ["<sub1>", "<sub2>"]}\n\n'
            'User query: {query}'
        )

        generation_prompt = (
            f'You are an {role} for the document collection "{self.profile.name}".\n\n'
            'TASK: Answer the user\'s question using ONLY the provided text.\n\n'
            '<thinking>\n'
            'Before answering, work through:\n'
            '1. Which sections directly address this question?\n'
            '2. Do any errata, amendments, or Q&A entries supersede the base text?\n'
            '3. Are there cross-references that affect the answer?\n'
            '4. What is the authoritative final answer?\n'
            '</thinking>\n\n'
            'DOCUMENTS:\n{context}\n\n'
            'QUESTION: {query}\n\n'
            'Rules for your response:\n'
            f'- EVERY factual statement must cite its source document or section number using format: {citation_fmt}\n'
            f'- Conflict arbitration guideline: {conflict_rule}\n'
            '- If an amendment/errata supersedes a base rule, say so explicitly\n'
            '- If you cannot find the answer in the provided text, say so clearly\n'
            '- Be precise and concise\n\n'
            'ANSWER:'
        )

        return classify_prompt, generation_prompt

    def query(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """
        Execute query against the domain corpus and return structured answer.
        """
        if self.pipeline_mode == "VISUAL_MEDIA":
            from engine.retrieval.media_agent import MediaAgent
            agent = MediaAgent(self.profile_path)
            results = agent.search(query_text, top_k=kwargs.get("top_k", 5))
            return {
                "domain": self.profile.name,
                "pipeline_mode": self.pipeline_mode,
                "query": query_text,
                "answer": results,
                "context": results,
                "debug_info": {"mode": "visual_media"}
            }

        elif self.pipeline_mode == "POLICY_HIERARCHICAL":
            from engine.retrieval.policy_agent import compare_policies
            carriers = kwargs.get("carriers")
            answer, context = compare_policies(query_text, carriers=carriers, profile_path=self.profile_path)
            return {
                "domain": self.profile.name,
                "pipeline_mode": self.pipeline_mode,
                "query": query_text,
                "answer": answer,
                "context": context,
                "debug_info": {"mode": "policy_comparison"}
            }

        else:
            # Default / RULEBOOK_TECHNICAL mode
            from engine.retrieval.rules_lawyer import ask_rules_lawyer_game
            answer, context, debug_info = ask_rules_lawyer_game(query_text, profile_path=self.profile_path)
            return {
                "domain": self.profile.name,
                "pipeline_mode": self.pipeline_mode,
                "query": query_text,
                "answer": answer,
                "context": context,
                "debug_info": debug_info
            }
