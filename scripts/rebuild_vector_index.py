import os
import sys
import json
import re
import time
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STOPWORDS = {
    "what", "is", "the", "and", "or", "of", "to", "in", "a", "for", "with", "on",
    "at", "by", "from", "under", "across", "vs", "versus", "compare", "does", "do",
    "how", "which", "who", "when", "where", "can", "could", "would", "should",
    "rule", "rules", "about", "tell", "me", "explain", "describe", "say", "says",
    "that", "this", "there", "are", "was", "were", "been", "have", "has", "had",
    "any", "all", "some", "an", "my", "its", "if", "then", "than", "into", "also",
    "between", "during", "after", "before", "section", "game", "play", "player"
}

GAME_ACRONYMS = {
    "sl", "cc", "dc", "fp", "th", "rr", "rc", "mg", "ba", "ig", "fl", "dr",
    "afv", "ap", "he", "los", "drm", "rnc", "rpc", "ot", "atg", "ft", "smg", "lmg", "mmg", "hmg"
}

def extract_tokens(text: str):
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    words = cleaned.split()
    tokens = set()
    for w in words:
        if w in GAME_ACRONYMS:
            tokens.add(w)
        elif w not in STOPWORDS and len(w) > 2:
            tokens.add(w)
    return tokens

def rebuild_index():
    print("=" * 70)
    print("🚀 REBUILDING UP FRONT VECTOR & HIGH-PRECISION KEYWORD INDEX")
    print("=" * 70)
    
    session = boto3.Session(profile_name="aiia", region_name="ap-southeast-2")
    ddb = session.resource("dynamodb")
    table = ddb.Table("RagDoll-Knowledge-test")
    bedrock = session.client("bedrock-runtime")

    # 1. Fetch all rules
    res = table.query(
        KeyConditionExpression="PK = :pk",
        ExpressionAttributeValues={":pk": "TITLE#up-front-core"}
    )
    items = res.get("Items", [])
    while "LastEvaluatedKey" in res:
        res = table.query(
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={":pk": "TITLE#up-front-core"},
            ExclusiveStartKey=res["LastEvaluatedKey"]
        )
        items.extend(res.get("Items", []))

    print(f"Retrieved {len(items)} rules from DynamoDB")

    # Reuse existing embeddings if available to save time, or embed
    existing_file = "c:/programming/aiia/gaiia-rag-doll-cloud/backend/assets/titles/up-front-core/vector_index.json"
    existing_vectors = {}
    if os.path.exists(existing_file):
        try:
            with open(existing_file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                existing_vectors = old_data.get("vectors", {})
                print(f"Reusing {len(existing_vectors)} pre-computed 1024-dim Titan vectors")
        except Exception:
            pass

    title_index = {}    # token -> list of rule numbers where token is in TITLE
    body_index = {}     # token -> list of rule numbers where token is in BODY
    rules_meta = {}

    for item in items:
        rn = str(item.get("ruleNumber", "")).strip()
        title = str(item.get("title", ""))
        verbatim = str(item.get("verbatimText", ""))
        chapter = str(item.get("chapter", ""))

        title_tokens = extract_tokens(f"{title} {verbatim[:80]}")
        body_tokens = extract_tokens(f"{chapter} {verbatim}")

        for t in title_tokens:
            if t not in title_index:
                title_index[t] = []
            if rn not in title_index[t]:
                title_index[t].append(rn)

        for t in body_tokens:
            if t not in body_index:
                body_index[t] = []
            if rn not in body_index[t]:
                body_index[t].append(rn)

        rules_meta[rn] = {
            "title": title,
            "chapter": chapter
        }

    print(f"Title index tokens: {len(title_index)}, Body index tokens: {len(body_index)}")
    print("Tokens for 'radio': in title ->", title_index.get("radio"), "| in body ->", len(body_index.get("radio", [])))
    print("Tokens for 'sl': in title ->", title_index.get("sl"), "| in body ->", len(body_index.get("sl", [])))
    print("Tokens for 'smoke': in title ->", title_index.get("smoke"), "| in body ->", len(body_index.get("smoke", [])))

    payload = {
        "version": "2.1.0",
        "titleId": "up-front-core",
        "modelId": "amazon.titan-embed-text-v2:0",
        "dimensions": 1024,
        "ruleCount": len(items),
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vectors": existing_vectors,
        "titleIndex": title_index,
        "keywordIndex": body_index,
        "rulesMeta": rules_meta
    }

    with open(existing_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    file_size_mb = os.path.getsize(existing_file) / (1024 * 1024)
    print(f"✅ Enhanced vector index saved to {existing_file} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    rebuild_index()
