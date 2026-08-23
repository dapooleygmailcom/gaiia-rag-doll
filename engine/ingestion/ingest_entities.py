import os
import glob
import json
import chromadb
import ollama

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data/chroma")
CHROMA_COLLECTION = "renegade_legion-entities"

def json_to_semantic_text(entity):
    name = entity.get("name", "Unknown Unit")
    etype = entity.get("entity_type", "vehicle")
    lines = [f"{name} is a {etype} unit."]
    
    attrs = entity.get("attributes", {})
    if attrs:
        lines.append("Attributes:")
        for k, v in attrs.items():
            lines.append(f"- {k}: {v}")
            
    grids = entity.get("grids", {})
    if grids:
        lines.append("Armor and Grids:")
        for facing, data in grids.items():
            if "SF" in data:
                lines.append(f"- {facing} has SF (Size Factor) {data['SF']}.")
            else:
                lines.append(f"- {facing} is {data.get('Width', '?')} columns wide.")
                
    weapons = entity.get("collections", {}).get("Weapons", [])
    if weapons:
        lines.append("Weapons:")
        for w in weapons:
            lines.append(f"- {w.get('Name', 'Weapon')} mounted on {w.get('Location', 'Unknown')} (Damage: {w.get('Damage', 'N/A')}, Range: {w.get('Range', 'N/A')})")
            
    missiles = entity.get("collections", {}).get("Missiles", [])
    if missiles:
        lines.append("Missiles:")
        for m in missiles:
            lines.append(f"- {m.get('Count', 1)}x {m.get('Type', 'Missile')}")
            
    return "\n".join(lines)

def ingest_entities(entities_dir):
    print(f"Connecting to ChromaDB at {CHROMA_DB_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Create or get the collection specifically for entity profiles
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    
    json_files = glob.glob(os.path.join(entities_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {entities_dir}")
    
    for jpath in json_files:
        basename = os.path.basename(jpath)
        with open(jpath, "r", encoding="utf-8") as f:
            try:
                entity = json.load(f)
            except Exception as e:
                print(f"Skipping {basename} due to JSON error: {e}")
                continue
                
        if "name" not in entity:
            print(f"Skipping {basename} - no 'name' key")
            continue
            
        doc_text = json_to_semantic_text(entity)
        chunk_id = f"entity_{basename}"
        
        try:
            response = ollama.embeddings(model="nomic-embed-text", prompt=doc_text)
            embedding = response["embedding"]
            
            collection.upsert(
                ids=[chunk_id],
                documents=[doc_text],
                embeddings=[embedding],
                metadatas=[{
                    "source_file": basename,
                    "entity_name": entity["name"],
                    "doc_type": "entity_profile",
                    "priority": 1 # High priority for direct factual lookup
                }]
            )
            print(f"Ingested {entity['name']} ({basename})")
        except Exception as e:
            print(f"Error ingesting {basename}: {e}")

if __name__ == "__main__":
    entities_dir = os.path.join(PROJECT_ROOT, "data/entities")
    ingest_entities(entities_dir)
