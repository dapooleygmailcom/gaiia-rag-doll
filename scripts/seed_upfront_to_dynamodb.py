"""
Seed Up Front Ingested Corpus into AWS DynamoDB RagDoll-Knowledge-test.

1. Removes the dummy test record (RULE#12.4).
2. Reads local ingestion artifacts:
   - data/up_front_rule_index.json (609 rules)
   - data/up_front_section_tree.json (hierarchical sections & breadcrumbs)
   - data/up_front_cooccurrence_graph.json (cross-references)
   - data/chroma (verbatim text chunks from up_front-rules-semantic collection)
3. Formats canonical Knowledge Table records.
4. Batch-writes all 609 rules into RagDoll-Knowledge-test using AWS profile 'aiia'.
5. Updates title metadata in RagDoll-Tenants-test.
"""

import os
import sys
import json
import re
from datetime import datetime
from decimal import Decimal

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import boto3
from botocore.exceptions import ClientError

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

AWS_PROFILE = "aiia"
AWS_REGION = "ap-southeast-2"
KNOWLEDGE_TABLE_NAME = "RagDoll-Knowledge-test"
TENANTS_TABLE_NAME = "RagDoll-Tenants-test"
TITLE_PK = "TITLE#up-front-core"
TENANT_PK = "TENANT#internal-test-tenant"
TITLE_SK = "TITLE#up-front-core"


def clean_rule_text(text: str) -> str:
    """Clean chunk text, stripping internal system doc/section header prefixes."""
    # Remove header patterns like [Doc: core_rules] [Section: ...]
    cleaned = re.sub(r'\[Doc:\s*[^\]]+\]\s*', '', text)
    cleaned = re.sub(r'\[Section:\s*[^\]]+\]\s*', '', cleaned)
    # Collapse multiple blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def extract_title_from_text(rule_num: str, text: str, default_title: str = "") -> str:
    """Extract human-readable clause title from the beginning of a rule chunk."""
    # Match patterns like: "12.1 IMMOBILIZATION" or "12.1. Immobilization" or "12.1 - Immobilization"
    pattern = rf'(?:^|\n)\s*{re.escape(rule_num)}\.?\s*[-–—:]?\s*([A-Z0-9\s,\-\'\(\)\/]{{3,60}})(?:\r?\n|$|\.)'
    match = re.search(pattern, text)
    if match:
        candidate = match.group(1).strip()
        # Filter out numbers only or boilerplate
        if len(candidate) > 2 and not candidate.isdigit():
            return candidate.title()
    return default_title or f"Rule {rule_num}"


def load_local_corpus():
    """Load local rule index, section tree, co-occurrence graph, and Chroma chunks."""
    print("📖 Loading local Up Front ingestion corpus...")

    rule_index_path = os.path.join(PROJECT_ROOT, "data", "up_front_rule_index.json")
    section_tree_path = os.path.join(PROJECT_ROOT, "data", "up_front_section_tree.json")
    cooc_graph_path = os.path.join(PROJECT_ROOT, "data", "up_front_cooccurrence_graph.json")
    chroma_path = os.path.join(PROJECT_ROOT, "data", "chroma")

    if not os.path.exists(rule_index_path):
        raise FileNotFoundError(f"Missing {rule_index_path}")

    with open(rule_index_path, "r", encoding="utf-8") as f:
        rule_index = json.load(f)

    # Filter out internal section tree key if embedded
    rule_keys = [k for k in rule_index.keys() if not k.startswith("__")]
    print(f"  • Found {len(rule_keys)} rules in rule index.")

    section_tree = {}
    if os.path.exists(section_tree_path):
        with open(section_tree_path, "r", encoding="utf-8") as f:
            section_tree = json.load(f)
        print(f"  • Loaded section tree ({len(section_tree.get('sections', {}))} sections).")

    cooc_graph = {}
    if os.path.exists(cooc_graph_path):
        with open(cooc_graph_path, "r", encoding="utf-8") as f:
            cooc_graph = json.load(f)
        print(f"  • Loaded co-occurrence graph ({len(cooc_graph.get('adjacency', {}))} nodes).")

    # Load chunk texts from ChromaDB
    chunk_text_map = {}
    if os.path.exists(chroma_path):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=chroma_path)
            col_name = "up_front-rules-semantic"
            if col_name in [c.name for c in client.list_collections()]:
                col = client.get_collection(col_name)
                total_chunks = col.count()
                print(f"  • Connecting to Chroma collection '{col_name}' ({total_chunks} chunks)...")
                # Retrieve all chunks in batches
                limit = 500
                offset = 0
                while offset < total_chunks:
                    data = col.get(limit=limit, offset=offset, include=["documents", "metadatas"])
                    for cid, doc in zip(data["ids"], data["documents"]):
                        chunk_text_map[cid] = doc
                    offset += limit
                print(f"  • Loaded {len(chunk_text_map)} chunk texts from Chroma.")
        except Exception as e:
            print(f"  ⚠️ Warning loading Chroma collection: {e}")

    return rule_index, rule_keys, section_tree, cooc_graph, chunk_text_map


def remove_test_records(knowledge_table):
    """Delete existing test/dummy records for TITLE#up-front-core."""
    print("\n🧹 Step 1: Checking and removing existing test records...")
    
    # Query all items under TITLE#up-front-core
    response = knowledge_table.query(
        KeyConditionExpression="PK = :pk",
        ExpressionAttributeValues={":pk": TITLE_PK}
    )
    items = response.get("Items", [])
    print(f"  • Found {len(items)} existing records in {KNOWLEDGE_TABLE_NAME} for {TITLE_PK}.")

    for item in items:
        sk = item["SK"]
        print(f"    - Deleting existing record: PK={TITLE_PK}, SK={sk}")
        knowledge_table.delete_item(
            Key={"PK": TITLE_PK, "SK": sk}
        )
    print("  ✅ Pre-existing test records successfully removed.")


def build_knowledge_records(rule_index, rule_keys, section_tree, cooc_graph, chunk_text_map):
    """Assemble DynamoDB item representations for all rules."""
    print("\n🔨 Step 2: Constructing canonical knowledge records for 609 rules...")

    sections = section_tree.get("sections", {})
    rule_to_sec = section_tree.get("rule_to_section_map", {})
    adjacency = cooc_graph.get("adjacency", {})

    records = []

    for rule_num in sorted(rule_keys):
        chunk_entries = rule_index.get(rule_num, [])
        if not isinstance(chunk_entries, list):
            continue

        # Gather texts from chunk IDs
        texts = []
        source_files = set()
        highest_priority = 9

        for entry in chunk_entries:
            cid = entry.get("chunk_id")
            sfile = entry.get("source_file")
            prio = entry.get("priority", 9)
            if sfile:
                source_files.add(sfile)
            if prio < highest_priority:
                highest_priority = prio

            if cid and cid in chunk_text_map:
                txt = clean_rule_text(chunk_text_map[cid])
                if txt and txt not in texts:
                    texts.append(txt)

        # Build verbatim text
        if texts:
            combined_verbatim = "\n\n---\n\n".join(texts)
        else:
            combined_verbatim = f"Up Front codified clause {rule_num}. Please refer to official rulebook for full text."

        # Section and breadcrumb resolution
        sec_id = rule_to_sec.get(rule_num)
        sec_info = sections.get(sec_id, {}) if sec_id else {}
        sec_title = sec_info.get("title", "")

        # Extract clause title
        clause_title = extract_title_from_text(rule_num, combined_verbatim, default_title=sec_title)

        # Chapter determination
        major_section = rule_num.split(".")[0]
        chapter = f"Section {major_section}.0"
        if sec_title:
            chapter = f"Section {major_section}.0: {sec_title}"

        # Breadcrumbs
        breadcrumbs = ["Up Front Core Rules", f"Section {major_section}.0"]
        if sec_id and sec_id != major_section:
            breadcrumbs.append(f"Section {sec_id}")
        breadcrumbs.append(f"Rule {rule_num} {clause_title}".strip())

        # Siblings from section tree
        siblings = []
        if sec_info and "child_rules" in sec_info:
            for sib in sec_info["child_rules"]:
                if sib != rule_num:
                    siblings.append({"number": sib, "title": f"Rule {sib}"})
                    if len(siblings) >= 5:
                        break

        # Cross-references from co-occurrence graph
        cross_refs = []
        if rule_num in adjacency:
            for edge in adjacency[rule_num]:
                target = edge.get("target")
                if target and target != rule_num:
                    cross_refs.append({
                        "number": target,
                        "title": f"Rule {target}",
                        "weight": Decimal(str(round(float(edge.get("weight", 0.5)), 3)))
                    })
                    if len(cross_refs) >= 5:
                        break

        record = {
            "PK": TITLE_PK,
            "SK": f"RULE#{rule_num}",
            "ruleNumber": str(rule_num),
            "title": clause_title[:120],
            "verbatimText": combined_verbatim,
            "chapter": chapter[:120],
            "priority": int(highest_priority if highest_priority != 9 else 1),
            "breadcrumbs": breadcrumbs,
            "siblings": siblings,
            "crossReferences": cross_refs,
            "sourceFiles": list(source_files),
            "updatedAt": datetime.utcnow().isoformat() + "Z"
        }
        records.append(record)

    print(f"  • Formatted {len(records)} knowledge records ready for upload.")
    return records


def batch_write_to_dynamodb(knowledge_table, records):
    """Batch write all records into DynamoDB table."""
    print(f"\n🚀 Step 3: Batch writing {len(records)} records into {KNOWLEDGE_TABLE_NAME}...")

    total = len(records)
    written = 0

    with knowledge_table.batch_writer() as batch:
        for i, item in enumerate(records, 1):
            batch.put_item(Item=item)
            written += 1
            if i % 50 == 0 or i == total:
                pct = (i / total) * 100
                print(f"  • Progress: {i}/{total} ({pct:.1f}%) uploaded...")

    print(f"  ✅ Successfully uploaded all {written} rule records to {KNOWLEDGE_TABLE_NAME}!")


def update_title_metadata(tenants_table, rule_count):
    """Update title record in RagDoll-Tenants-test to reflect full 609 rules."""
    print(f"\n📝 Step 4: Updating title metadata in {TENANTS_TABLE_NAME}...")

    now = datetime.utcnow().isoformat() + "Z"
    try:
        tenants_table.update_item(
            Key={"PK": TENANT_PK, "SK": TITLE_SK},
            UpdateExpression="SET ruleCount = :rc, #st = :ready, updatedAt = :now",
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={
                ":rc": rule_count,
                ":ready": "READY",
                ":now": now
            }
        )
        print(f"  ✅ Title record updated: ruleCount={rule_count}, status=READY.")
    except Exception as e:
        print(f"  ⚠️ Warning updating title metadata: {e}")


def verify_cloud_knowledge_table(knowledge_table):
    """Verify item count and test a few sample queries."""
    print("\n🔍 Step 5: Verifying DynamoDB cloud records...")
    
    # Check item count
    res = knowledge_table.query(
        KeyConditionExpression="PK = :pk",
        ExpressionAttributeValues={":pk": TITLE_PK},
        Select="COUNT"
    )
    count = res.get("Count", 0)
    print(f"  • Verified items under {TITLE_PK}: {count} records in DynamoDB.")

    # Sample test lookups
    sample_rules = ["1.0", "8.52", "12.1", "34.3"]
    for r in sample_rules:
        item_res = knowledge_table.get_item(
            Key={"PK": TITLE_PK, "SK": f"RULE#{r}"}
        )
        item = item_res.get("Item")
        if item:
            print(f"  ✅ Verified Rule {r}: '{item.get('title')}' ({len(item.get('verbatimText', ''))} chars)")
        else:
            print(f"  ❌ Rule {r} not found!")


def main():
    print("=" * 70)
    print("📦 SEEDING UP FRONT INGESTED CORPUS TO AWS DYNAMODB PRODUCTION")
    print(f"AWS Profile: {AWS_PROFILE} | Region: {AWS_REGION}")
    print(f"Target Knowledge Table: {KNOWLEDGE_TABLE_NAME}")
    print("=" * 70)

    # Initialize boto3 session
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    dynamodb = session.resource("dynamodb")
    knowledge_table = dynamodb.Table(KNOWLEDGE_TABLE_NAME)
    tenants_table = dynamodb.Table(TENANTS_TABLE_NAME)

    # Step 1: Remove existing test record
    remove_test_records(knowledge_table)

    # Step 2: Load local corpus
    rule_index, rule_keys, section_tree, cooc_graph, chunk_text_map = load_local_corpus()

    # Step 3: Build records
    records = build_knowledge_records(rule_index, rule_keys, section_tree, cooc_graph, chunk_text_map)

    # Step 4: Batch write
    batch_write_to_dynamodb(knowledge_table, records)

    # Step 5: Update title metadata
    update_title_metadata(tenants_table, len(records))

    # Step 6: Verify
    verify_cloud_knowledge_table(knowledge_table)

    print("\n" + "=" * 70)
    print("🎉 UP FRONT CORPUS SEEDING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
