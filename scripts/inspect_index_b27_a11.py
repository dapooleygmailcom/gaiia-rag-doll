import json
import chromadb

client = chromadb.PersistentClient("data/chroma")
col = client.get_collection("asl-rules-semantic")

with open("data/asl_rule_index.json", "r", encoding="utf-8") as f:
    rule_index = json.load(f)

print("=== CHECKING B27.13 (Foxholes) ===")
print("In rule_index keys containing '27.13' or 'b27.13':")
for k in rule_index:
    if "27.13" in k.lower():
        print(f"  Key: '{k}' -> {len(rule_index[k])} entries")
        for e in rule_index[k]:
            print(f"    Chunk ID: {e.get('chunk_id')}")
            try:
                res = col.get(ids=[e.get('chunk_id')])
                if res['documents']:
                    print(f"      Doc: {res['metadatas'][0].get('source_doc')}")
                    print(f"      Snippet: {res['documents'][0][:250]}")
            except Exception as ex:
                print(f"      Error: {ex}")

print("\n=== CHECKING A11.22 (Close Combat Infiltration / Snakes) ===")
for k in rule_index:
    if "11.22" in k.lower():
        print(f"  Key: '{k}' -> {len(rule_index[k])} entries")
        for e in rule_index[k]:
            print(f"    Chunk ID: {e.get('chunk_id')}")
            try:
                res = col.get(ids=[e.get('chunk_id')])
                if res['documents']:
                    print(f"      Doc: {res['metadatas'][0].get('source_doc')}")
                    print(f"      Snippet: {res['documents'][0][:250]}")
            except Exception as ex:
                print(f"      Error: {ex}")

print("\n=== CHECKING SECTION TREE FOR B27 AND A11 ===")
with open("data/asl_section_tree.json", "r", encoding="utf-8") as f:
    st = json.load(f)
print("Section tree B27:", st.get("B27") or st.get("27"))
print("Section tree A11:", st.get("A11") or st.get("11"))
