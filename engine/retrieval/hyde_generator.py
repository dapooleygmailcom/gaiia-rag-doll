"""
Dedicated HyDE (Hypothetical Document Embeddings) Generator — Gaiia RAG Doll.

Generates authoritative, domain-style pseudo-rulebook clauses from distilled queries
and extracted rule entities to bridge the semantic gap between user questions and
technical rulebook texts.
"""

import json
import re
from typing import List, Optional, Dict, Any
import ollama


class HydeGenerator:
    """
    Dedicated Stage-2 generator for synthesizing hypothetical rulebook passages.
    Operates at temperature T=0.4 for controlled lexical diversity and domain vocabulary.
    """

    def __init__(self, default_model: str = "llama3.1:8b", temperature: float = 0.4):
        self.default_model = default_model
        self.temperature = temperature

    def generate_pseudo_clause(
        self,
        distilled_query: str,
        rule_numbers: Optional[List[str]] = None,
        game_name: str = "Technical Rulebook",
        glossary: Optional[Dict[str, str]] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a 2-3 sentence hypothetical rulebook clause that answers/defines the topic.
        """
        if not distilled_query or not distilled_query.strip():
            return ""

        active_model = model or self.default_model
        rules_hint = ""
        if rule_numbers:
            rules_hint = f"Applicable Rule / Section Reference(s): {', '.join(str(r) for r in rule_numbers)}\n"

        glossary_hint = ""
        if glossary:
            relevant_terms = {
                k: v for k, v in glossary.items()
                if k.lower() in distilled_query.lower() or any(term in distilled_query.lower() for term in v.lower().split())
            }
            if relevant_terms:
                glossary_hint = f"Domain Terminology to Incorporate: {json.dumps(relevant_terms)}\n"

        prompt = (
            f'You are the principal author and rules editor for "{game_name}".\n\n'
            f'TASK: Write a concise, 2-to-3 sentence authoritative rulebook excerpt or clause that formally defines, '
            f'regulates, or answers the situation below.\n\n'
            f'GUIDELINES:\n'
            f'1. Use formal technical rulebook style, active voice, and precision condition phrasing ("A unit may...", "Unless...", "Provided that...").\n'
            f'2. Incorporate exact game mechanics, statuses, and procedural steps.\n'
            f'3. Do NOT include conversational filler, meta-talk, or introductions.\n'
            f'4. Output ONLY the raw pseudo-rulebook clause text.\n\n'
            f'{rules_hint}'
            f'{glossary_hint}'
            f'Topic / Situation: {distilled_query.strip()}\n\n'
            f'RULEBOOK CLAUSE:'
        )

        try:
            response = ollama.generate(
                model=active_model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "num_predict": 180
                }
            )
            raw = response.get("response", "").strip()

            cleaned = re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.DOTALL).strip()
            cleaned = re.sub(r'^(?:Here is (?:the|a) rulebook clause:?|RULEBOOK CLAUSE:?|Clause:?)\s*', '', cleaned, flags=re.IGNORECASE).strip()
            cleaned = cleaned.strip('"\'')

            if cleaned:
                return cleaned
            return distilled_query

        except Exception as e:
            print(f"  [HyDE Warning] Generation fallback ({e}). Using distilled query.")
            return distilled_query

    def generate_multi_perspective_clauses(
        self,
        distilled_query: str,
        rule_numbers: Optional[List[str]] = None,
        sub_queries: Optional[List[str]] = None,
        game_name: str = "Technical Rulebook",
        glossary: Optional[Dict[str, str]] = None,
        model: Optional[str] = None
    ) -> List[str]:
        """
        Generate two complementary hypothetical clauses (Primary Mechanic & Secondary Interaction/Exception)
        to ensure multi-faceted queries retrieve both core and peripheral consequence rules.
        """
        if not distilled_query or not distilled_query.strip():
            return []

        # 1. Primary Clause (Direct definition / mechanic)
        primary_clause = self.generate_pseudo_clause(
            distilled_query=distilled_query,
            rule_numbers=rule_numbers,
            game_name=game_name,
            glossary=glossary,
            model=model
        )

        clauses = [primary_clause] if primary_clause else []

        # If compound / multi-faceted query, generate secondary consequence/exception perspective
        is_compound = bool(sub_queries) or any(w in distilled_query.lower() for w in [" and ", " or ", "if ", "when ", "also", "then", "consequence", "effect", "after", "must", "check"])
        
        if is_compound:
            active_model = model or self.default_model
            secondary_prompt = (
                f'You are the principal author and rules editor for "{game_name}".\n\n'
                f'TASK: Write a 2-to-3 sentence authoritative rulebook excerpt describing the SECONDARY INTERACTIONS, '
                f'EXCEPTIONS, OR CONSEQUENCES (such as movement restrictions, bogging, weapon malfunction/discard, or status retention) '
                f'related to the situation below.\n\n'
                f'Situation: {distilled_query.strip()}\n\n'
                f'SECONDARY / EXCEPTION CLAUSE:'
            )
            try:
                response = ollama.generate(
                    model=active_model,
                    prompt=secondary_prompt,
                    options={
                        "temperature": self.temperature,
                        "top_p": 0.9,
                        "num_predict": 180
                    }
                )
                raw = response.get("response", "").strip()
                cleaned = re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.DOTALL).strip()
                cleaned = re.sub(r'^(?:Here is (?:the|a) rulebook clause:?|SECONDARY.*?CLAUSE:?|Clause:?)\s*', '', cleaned, flags=re.IGNORECASE).strip()
                cleaned = cleaned.strip('"\'')
                if cleaned and cleaned != primary_clause:
                    clauses.append(cleaned)
            except Exception:
                pass

        return clauses
