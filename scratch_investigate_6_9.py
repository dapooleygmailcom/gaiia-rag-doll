import chromadb
import json

client = chromadb.PersistentClient("data/chroma")
col = client.get_collection("asl-rules-semantic")

print("=" * 70)
print("=== INVESTIGATING TC 6 (Landing Craft) ===")
print("=" * 70)

# Check documents in ASL profile
profile = json.load(open("data/asl_profile.json", encoding="utf-8"))
docs = profile.get("documents", {})
print("Ingested docs count in profile:", len(docs))
chapter_g_docs = [k for k in docs if "g" in k.lower() or "pto" in k.lower() or "landing" in k.lower()]
print("Chapter G / Landing craft docs in profile:", chapter_g_docs)

from chromadb.utils import embedding_functions
import ollama

def embed_text(text):
    res = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return res["embedding"]

emb = embed_text("Landing Craft vehicle unloading costs LCM MP allotment")
q_lc = col.query(query_embeddings=[emb], n_results=10)
print("\nTop 10 semantic matches for Landing Craft in Chroma:")
for i in range(len(q_lc["ids"][0])):
    meta = q_lc["metadatas"][0][i]
    doc = q_lc["documents"][0][i]
    print(f"[{i+1}] Rule: {meta.get('rule_number')} | Doc: {meta.get('source_doc')}")
    print(f"    Snippet: {doc[:180].strip()}")

print("\n" + "=" * 70)
print("=== INVESTIGATING TC 9 (No Quarter & Low Crawl) ===")
print("=" * 70)

# Check exact rules for TC 9: A20.21, A20.3, A10.52
for r in ["A20.21", "20.21", "A20.3", "20.3", "A10.52", "10.52", "A10.5", "10.5"]:
    res = col.get(where={"rule_number": r})
    print(f"Chroma lookup for rule_number '{r}': {len(res['ids'])} chunks")
    for d, m in zip(res["documents"], res["metadatas"]):
        print(f"  -> Doc: {m.get('source_doc')} | Snippet: {d[:200].strip()}")

# Also check what was retrieved in the checkpoint for TC 9
chk = json.load(open("data/eval/asl_bgg_eval_checkpoint.json", encoding="utf-8"))
for item in chk["results"]:
    if item["id"] == "bgg_asl_3728472":
        print("\nTC 9 retrieved rules from checkpoint:")
        print("Expected:", item.get("expected_rules"))
        print("Retrieved:", item.get("retrieved_rules"))
        print("Hits:", item.get("hits"))
        print("\nGenerated answer:")
        print(item.get("generated_answer"))
