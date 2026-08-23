"""
Vision-Assisted Entity & Diagram Record Sheet Parser — Gaiia RAG Doll.

Extracts structured JSON from complex diagrammatic record sheets (e.g. wargame vehicle SSDs,
engineering schematics) using Vision LLMs guided dynamically by DomainProfile Meta-Contracts.
"""

import os
import re
import json
import base64
from typing import Optional, List, Dict, Any

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def build_ssd_extraction_prompt(ontology: Dict[str, Any], hints: Optional[List[str]] = None, domain_name: str = "wargame") -> str:
    """
    Construct a dynamic extraction prompt tailored to the target ontology and domain extraction hints.
    """
    hints_text = ""
    if hints:
        hints_text = "\n" + "\n".join(f"- {h}" for h in hints) + "\n"

    prompt = f"""You are a structured data extraction agent for "{domain_name}".
I am providing you with a PDF / image of an entity record sheet or schematic.

Expected JSON Ontology Schema:
{json.dumps(ontology, indent=2)}

Field-specific extraction guidelines and validation warnings:{hints_text}
Please extract all data from this document and populate a JSON object that strictly adheres to this ontology schema.
Output ONLY the raw JSON object matching the ontology:"""
    return prompt


def parse_ssd_pdf(pdf_path: str, ontology_path: Optional[str] = None, profile_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Parse a record sheet PDF using a Vision LLM driven by the domain ontology and profile hints.
    """
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    if genai is None:
        raise ImportError("google-genai library is not installed.")

    ontology: Dict[str, Any] = {}
    hints: List[str] = []
    domain_name: str = "entity specification"

    if profile_path and os.path.exists(profile_path):
        from engine.models.domain_profile import load_domain_profile
        profile = load_domain_profile(profile_path)
        domain_name = profile.name
        if profile.structured_extraction:
            hints = profile.structured_extraction.vlm_extraction_hints
            if profile.structured_extraction.target_schema:
                ontology = profile.structured_extraction.target_schema
            elif profile.structured_extraction.target_schema_file:
                schema_file = profile.structured_extraction.target_schema_file
                if os.path.exists(schema_file):
                    with open(schema_file, "r", encoding="utf-8") as f:
                        ontology = json.load(f)

    if not ontology and ontology_path and os.path.exists(ontology_path):
        with open(ontology_path, "r", encoding="utf-8") as f:
            ontology = json.load(f)

    if not ontology:
        raise ValueError("No ontology schema provided for SSD parsing.")

    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    print(f"Parsing {pdf_path} using Vision LLM for {domain_name}...")
    prompt = build_ssd_extraction_prompt(ontology, hints=hints, domain_name=domain_name)

    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=[
            types.Part.from_bytes(
                data=pdf_data,
                mime_type='application/pdf',
            ),
            prompt,
        ]
    )

    output = response.text.strip()
    json_match = re.search(r'\{.*\}', output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            basename = os.path.basename(pdf_path).replace(".pdf", "")
            out_file = f"data/entities/{basename}.json"
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            print(f"Success! Entity saved to {out_file}")
            return parsed
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from Vision LLM: {e}")
    else:
        print("Failed to extract JSON from response.")
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python vision_ssd_parser.py <pdf_path> <ontology_path_or_profile_path>")
        sys.exit(1)
    
    arg = sys.argv[2]
    if arg.endswith("_profile.json"):
        parse_ssd_pdf(sys.argv[1], profile_path=arg)
    else:
        parse_ssd_pdf(sys.argv[1], ontology_path=arg)
