import os
import json
import base64
from google import genai
from google.genai import types

def parse_ssd_pdf(pdf_path, ontology_path):
    # Ensure API key is set
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    
    # Initialize the client
    client = genai.Client()

    # Load the ontology
    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    # Read the PDF file
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    print(f"Parsing {pdf_path} using Vision LLM...")

    prompt = f"""You are a data extraction agent. I am providing you with a PDF of a wargame entity sheet (like a vehicle record sheet).
I am also providing you with the expected JSON Ontology for this game:
{json.dumps(ontology, indent=2)}

Please extract all data from this PDF and populate a JSON object that strictly adheres to this ontology.
For grids (like Armor), try to infer the SF (Size Factor) and the dimensions (Width, Depth) by counting the boxes.
For weapons, extract the tables completely.

Output ONLY the raw JSON object."""

    # We use gemini-2.5-pro for vision tasks
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
    
    import re
    json_match = re.search(r'\{.*\}', output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            # Save the entity
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
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python vision_ssd_parser.py <pdf_path> <ontology_path>")
        sys.exit(1)
    parse_ssd_pdf(sys.argv[1], sys.argv[2])
