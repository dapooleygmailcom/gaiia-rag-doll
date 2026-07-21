import chromadb
import ollama
import json
import os
import re

def generate_ontology(profile_path="data/renegade_legion_profile.json"):
    print("Loading profile...")
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    
    client = chromadb.PersistentClient(path="data/chroma")
    collection_name = profile["chroma_collection"]
    collection = client.get_collection(name=collection_name)
    
    # Query for vehicle construction and damage rules
    query = "vehicle construction damage components armor internal systems"
    print(f"Querying Chroma collection '{collection_name}' for: {query}")
    
    response = ollama.embeddings(model="nomic-embed-text", prompt=query)
    
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=20,
        include=["documents"]
    )
    
    docs = results.get("documents", [[]])[0]
    context = "\n".join(docs)
    
    print(f"Retrieved {len(docs)} chunks. Synthesizing ontology...")
    
    prompt = f"""You are a data architect for the wargame '{profile["game_name"]}'.
Read the following retrieved rule texts about vehicle construction, damage templates, and internal components.

Based on these rules, deduce the ontological structure (a JSON Schema) of a generic vehicle entity in this game.
The schema must capture all standard attributes (like maximum thrust), all damage grids/templates (like armor facings), and component lists (like weapon mounts).

IMPORTANT: Provide ONLY the raw JSON object representing the ontology schema. Do not include markdown blocks, explanations, or any other text.
The JSON should follow a structure like:
{{
  "entity_type": "vehicle",
  "attributes": [ "list", "of", "expected", "attribute", "names" ],
  "grids": [ "list", "of", "expected", "damage", "grid", "names" ],
  "collections": [ "list", "of", "expected", "collections" ]
}}

Retrieved Rules Context:
{context}

JSON Schema Output ONLY:"""

    resp = ollama.generate(model="qwen2.5:14b", prompt=prompt)
    output = resp["response"].strip()
    
    # Try to extract just the JSON
    json_match = re.search(r'\{.*\}', output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            out_file = f"data/{profile['game_id']}_ontology.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            print(f"Success! Ontology saved to {out_file}")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from LLM: {e}")
            print(f"Raw Output:\n{output}")
    else:
        print("Failed to find JSON block in output.")
        print(f"Raw Output:\n{output}")

if __name__ == "__main__":
    generate_ontology()
