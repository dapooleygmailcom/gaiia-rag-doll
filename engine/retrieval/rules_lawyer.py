"""
Rules Lawyer Agent — Query pipeline for Up Front wargame rules.

Handles 6 query types with routed retrieval, priority reranking,
cross-reference chasing, temporal conflict resolution, and cited generation.

Models:
  - llama3.1:8b  → query classification
  - qwen2.5:14b  → answer generation
  - nomic-embed-text → embeddings
"""

import os
import re
import json
import chromadb
import ollama

CHROMA_DB_DIR = "data/chroma"
CHROMA_COLLECTION = "upfront-rules-semantic"
RULE_INDEX_FILE = "data/upfront_rule_index.json"

# ═══════════════════════════════════════════════════════════════════
# Query Classification
# ═══════════════════════════════════════════════════════════════════

CLASSIFY_PROMPT = """You are a query classifier for a wargame rules reference system.
Classify the user's query into exactly ONE of these categories:

- "direct_rule": User asks about a specific rule number (e.g., "What does rule 5.41 say?")
- "concept": User asks about a game concept or mechanic (e.g., "How does relative range work?")
- "situation": User describes a game situation and wants a ruling (e.g., "Can I fire at a pinned man?")
- "scenario": User asks about a specific scenario (A through K) or its special rules
- "comparison": User asks what changed between versions, or about errata for a specific rule
- "variant": User asks about house rules, variant rules, or unofficial modifications

If the user's query is complex and requires multiple independent searches (e.g. asking how a scenario rule modifies a base rule), provide a list of 2-3 simpler sub-queries.
CRITICAL INSTRUCTION FOR ABBREVIATIONS: Expand game-specific abbreviations in your sub-queries to include BOTH the abbreviation and the full term (e.g., if the user asks about "MPh", your sub-query should include "Movement Phase MPh". Expand AFV, NTC, DRM, LOS, OVR, etc. appropriately).


Respond with ONLY a JSON object: {{"query_type": "<type>", "rule_numbers": [<any rule numbers mentioned>], "scenario": "<scenario letter if any>", "sub_queries": ["<sub query 1>", "<sub query 2>"]}}

User query: {query}"""


def classify_query(query):
    """
    Classify a user query using llama3.1:8b.
    Returns dict with query_type, rule_numbers, scenario.
    """
    prompt = CLASSIFY_PROMPT.format(query=query)
    
    try:
        response = ollama.generate(model="llama3.1:8b", prompt=prompt)
        raw = response["response"].strip()
        
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Validate
            valid_types = {"direct_rule", "concept", "situation", "scenario", "comparison", "variant"}
            if result.get("query_type") not in valid_types:
                result["query_type"] = "concept"  # safe fallback
            return result
        else:
            print(f"  Warning: Could not parse classifier response: {raw[:100]}")
            return {"query_type": "concept", "rule_numbers": [], "scenario": None}
    except Exception as e:
        print(f"  Error in query classification: {e}")
        return {"query_type": "concept", "rule_numbers": [], "scenario": None}


# ═══════════════════════════════════════════════════════════════════
# Keyword Extraction (hybrid boosting)
# ═══════════════════════════════════════════════════════════════════

def extract_keywords(query):
    """Extract core keywords for BM25-style keyword boosting."""
    cleaned = re.sub(r'[^\w\s]', '', query.lower())
    words = cleaned.split()
    stopwords = {
        "what", "is", "the", "and", "or", "of", "to", "in", "a", "for", "with", "on",
        "at", "by", "from", "under", "across", "vs", "versus", "compare", "does", "do",
        "how", "which", "who", "when", "where", "can", "could", "would", "should",
        "rule", "rules", "about", "tell", "me", "explain", "describe", "say", "says",
        "that", "this", "there", "are", "was", "were", "been", "have", "has", "had",
        "any", "all", "some", "an", "my", "its", "if", "then", "than", "into", "also",
        "between", "during", "after", "before"
    }
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords


# ═══════════════════════════════════════════════════════════════════
# Retrieval Strategies
# ═══════════════════════════════════════════════════════════════════

def load_rule_index():
    """Load the JSON rule-number lookup index."""
    if not os.path.exists(RULE_INDEX_FILE):
        print(f"  Warning: Rule index not found at {RULE_INDEX_FILE}")
        return {}
    with open(RULE_INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_collection():
    """Get the ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_collection(name=CHROMA_COLLECTION)


def retrieve_by_rule_number(collection, rule_index, rule_numbers):
    """
    Direct rule-number lookup via JSON index → ChromaDB fetch.
    Returns chunks sorted by priority.
    """
    chunk_ids = []
    for rule_num in rule_numbers:
        entries = rule_index.get(rule_num, [])
        for entry in entries:
            if entry["chunk_id"] not in chunk_ids:
                chunk_ids.append(entry["chunk_id"])
    
    if not chunk_ids:
        return [], []
    
    try:
        results = collection.get(
            ids=chunk_ids[:20],  # Limit to 20 chunks
            include=["documents", "metadatas"]
        )
        return results.get("documents", []), results.get("metadatas", [])
    except Exception as e:
        print(f"  Error in rule lookup: {e}")
        return [], []


def retrieve_semantic(collection, query, n_results=8, where_filter=None):
    """
    Semantic search with hybrid keyword boosting.
    Returns (documents, metadatas, distances).
    """
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        query_vector = response["embedding"]
    except Exception as e:
        print(f"  Error generating embedding: {e}")
        return [], [], []
    
    # Oversample for reranking
    oversample_n = int(n_results * 2.5)
    
    search_args = {
        "query_embeddings": [query_vector],
        "n_results": oversample_n,
        "include": ["metadatas", "documents", "distances"]
    }
    if where_filter:
        search_args["where"] = where_filter
    
    try:
        results = collection.query(**search_args)
    except Exception as e:
        print(f"  Error querying ChromaDB: {e}")
        return [], [], []
    
    if not results or not results.get("ids") or len(results["ids"][0]) == 0:
        return [], [], []
    
    # Hybrid keyword boosting
    keywords = extract_keywords(query)
    
    scored = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    
    for i in range(len(ids)):
        text = docs[i].lower()
        match_count = sum(1 for kw in keywords if kw in text)
        boost = match_count * 0.12
        final_score = dists[i] - boost
        
        scored.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "distance": dists[i],
            "boost": boost,
            "final_score": final_score
        })
    
    # Sort by final score and take top n
    scored.sort(key=lambda x: x["final_score"])
    top = scored[:n_results]
    
    return (
        [s["text"] for s in top],
        [s["metadata"] for s in top],
        [s["final_score"] for s in top]
    )


# ═══════════════════════════════════════════════════════════════════
# Priority Reranking
# ═══════════════════════════════════════════════════════════════════

def priority_rerank(documents, metadatas, distances=None):
    """
    Rerank retrieved documents by document authority priority.
    Integrated rules (P1) > errata (P3) > core rules (P2) > scenarios (P5) > etc.
    """
    if not documents:
        return documents, metadatas
    
    items = []
    for i in range(len(documents)):
        priority = metadatas[i].get("priority", 9)
        dist = distances[i] if distances and i < len(distances) else 0.5
        
        # Combine semantic score with priority weight
        # Lower priority number = higher authority = lower combined score
        priority_weight = priority * 0.05
        combined = dist + priority_weight
        
        items.append({
            "doc": documents[i],
            "meta": metadatas[i],
            "combined_score": combined
        })
    
    items.sort(key=lambda x: x["combined_score"])
    
    return (
        [it["doc"] for it in items],
        [it["meta"] for it in items]
    )


# ═══════════════════════════════════════════════════════════════════
# Cross-Reference Chasing
# ═══════════════════════════════════════════════════════════════════

def chase_cross_references(collection, rule_index, documents, metadatas, max_refs=5):
    """
    Chase cross-references found in retrieved chunks.
    One-hop depth limit. Returns additional context documents.
    """
    # Extract all cross-references from retrieved documents
    all_refs = set()
    existing_rules = set()
    
    for doc in documents:
        refs = set()
        for match in re.finditer(
            r'\[(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\]|'
            r'(?:see|See)\s+(?:\[)?(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)(?:\])?|'
            r'EXC:\s*\[(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\]',
            doc
        ):
            for group in match.groups():
                if group:
                    refs.add(group)
        
        # Track which rules we already have
        rule_match = re.search(r'\[Rule:\s*([\d.]+)\]', doc)
        if rule_match:
            existing_rules.add(rule_match.group(1))
        
        all_refs.update(refs)
    
    # Remove rules we already have in the context
    new_refs = all_refs - existing_rules
    
    if not new_refs:
        return [], []
    
    # Fetch referenced chunks via rule index
    ref_chunk_ids = []
    for ref in list(new_refs)[:max_refs]:
        entries = rule_index.get(ref, [])
        if entries:
            # Take the highest-priority entry for this rule
            ref_chunk_ids.append(entries[0]["chunk_id"])
    
    if not ref_chunk_ids:
        return [], []
    
    try:
        results = collection.get(
            ids=ref_chunk_ids,
            include=["documents", "metadatas"]
        )
        ref_docs = results.get("documents", [])
        ref_metas = results.get("metadatas", [])
        return ref_docs, ref_metas
    except Exception as e:
        print(f"  Error chasing cross-references: {e}")
        return [], []


# ═══════════════════════════════════════════════════════════════════
# Temporal Conflict Resolution
# ═══════════════════════════════════════════════════════════════════

def resolve_conflicts(documents, metadatas):
    """
    Detect and annotate temporal conflicts between document versions.
    Returns conflict annotations to include in the generation context.
    """
    # Group documents by rule number
    by_rule = {}
    for i, meta in enumerate(metadatas):
        rule = meta.get("rule_number", "")
        if rule:
            if rule not in by_rule:
                by_rule[rule] = []
            by_rule[rule].append({
                "doc_type": meta.get("doc_type", ""),
                "priority": meta.get("priority", 9),
                "index": i
            })
    
    conflicts = []
    for rule_num, entries in by_rule.items():
        if len(entries) < 2:
            continue
        
        doc_types = set(e["doc_type"] for e in entries)
        
        # Check for actual conflict (different doc types covering same rule)
        if len(doc_types) > 1:
            # Find the highest-authority version
            entries.sort(key=lambda x: x["priority"])
            authoritative = entries[0]
            
            conflict_note = (
                f"CONFLICT ON RULE {rule_num}: Multiple sources found. "
                f"The authoritative version is from '{authoritative['doc_type']}' "
                f"(priority {authoritative['priority']}). "
                f"Other versions from: {', '.join(e['doc_type'] for e in entries[1:])}."
            )
            conflicts.append(conflict_note)
    
    return conflicts


# ═══════════════════════════════════════════════════════════════════
# Retrieval Router
# ═══════════════════════════════════════════════════════════════════

def route_and_retrieve(query, classification):
    """
    Route the classified query to the appropriate retrieval strategy.
    Returns (context_string, metadata_for_debug).
    """
    collection = get_collection()
    rule_index = load_rule_index()
    query_type = classification["query_type"]
    rule_numbers = classification.get("rule_numbers", [])
    scenario = classification.get("scenario")
    sub_queries = classification.get("sub_queries", [])
    
    print(f"  Query type: {query_type}")
    if rule_numbers:
        print(f"  Rule numbers: {rule_numbers}")
    if scenario:
        print(f"  Scenario: {scenario}")
    if sub_queries:
        print(f"  Sub-queries: {sub_queries}")
    
    documents = []
    metadatas = []
    distances = []
    
    # ─── Multi-query handling ───
    if sub_queries:
        for sq in sub_queries:
            sem_docs, sem_metas, sem_dists = retrieve_semantic(collection, sq, n_results=6)
            documents.extend(sem_docs)
            metadatas.extend(sem_metas)
            distances.extend(sem_dists)
            
    # ─── Route by query type ───
    if query_type == "direct_rule" and rule_numbers:
        # JSON index lookup
        docs, metas = retrieve_by_rule_number(collection, rule_index, rule_numbers)
        documents.extend(docs)
        metadatas.extend(metas)
        
        # Also do a semantic search to catch related content
        sem_docs, sem_metas, sem_dists = retrieve_semantic(collection, query, n_results=4)
        documents.extend(sem_docs)
        metadatas.extend(sem_metas)
        distances.extend(sem_dists)
    
    elif query_type == "scenario" and scenario:
        # Metadata-filtered retrieval
        where_filter = {"scenario": scenario}
        sem_docs, sem_metas, sem_dists = retrieve_semantic(
            collection, query, n_results=10, where_filter=where_filter
        )
        documents.extend(sem_docs)
        metadatas.extend(sem_metas)
        distances.extend(sem_dists)
        
        # Also get unfiltered results in case scenario rules reference base rules
        gen_docs, gen_metas, gen_dists = retrieve_semantic(collection, query, n_results=4)
        documents.extend(gen_docs)
        metadatas.extend(gen_metas)
        distances.extend(gen_dists)
    
    elif query_type == "comparison" and rule_numbers:
        # Multi-query: get both core rules and errata versions
        for rule_num in rule_numbers:
            entries = rule_index.get(rule_num, [])
            chunk_ids = [e["chunk_id"] for e in entries]
            if chunk_ids:
                try:
                    results = collection.get(
                        ids=chunk_ids[:10],
                        include=["documents", "metadatas"]
                    )
                    documents.extend(results.get("documents", []))
                    metadatas.extend(results.get("metadatas", []))
                except Exception:
                    pass
        
        # Supplement with semantic search
        sem_docs, sem_metas, sem_dists = retrieve_semantic(collection, query, n_results=4)
        documents.extend(sem_docs)
        metadatas.extend(sem_metas)
        distances.extend(sem_dists)
    
    elif query_type == "variant":
        # Filter by variant doc type
        where_filter = {"doc_type": "variant"}
        sem_docs, sem_metas, sem_dists = retrieve_semantic(
            collection, query, n_results=6, where_filter=where_filter
        )
        documents.extend(sem_docs)
        metadatas.extend(sem_metas)
        distances.extend(sem_dists)
        
        # Also get base rules for context
        gen_docs, gen_metas, gen_dists = retrieve_semantic(collection, query, n_results=4)
        documents.extend(gen_docs)
        metadatas.extend(gen_metas)
        distances.extend(gen_dists)
    
    elif query_type == "situation":
        # Broad semantic search + cross-ref chasing
        sem_docs, sem_metas, sem_dists = retrieve_semantic(collection, query, n_results=10)
        documents.extend(sem_docs)
        metadatas.extend(sem_metas)
        distances.extend(sem_dists)
    
    else:  # "concept" or fallback
        sem_docs, sem_metas, sem_dists = retrieve_semantic(collection, query, n_results=8)
        documents.extend(sem_docs)
        metadatas.extend(sem_metas)
        distances.extend(sem_dists)
    
    if not documents:
        return "No matching rules found in the database.", {}
    
    # ─── Deduplicate ───
    seen_texts = set()
    unique_docs = []
    unique_metas = []
    unique_dists = []
    for i, doc in enumerate(documents):
        # Use first 200 chars as dedup key
        key = doc[:200]
        if key not in seen_texts:
            seen_texts.add(key)
            unique_docs.append(doc)
            unique_metas.append(metadatas[i])
            if i < len(distances):
                unique_dists.append(distances[i])
            else:
                unique_dists.append(0.5)
    
    documents = unique_docs
    metadatas = unique_metas
    distances = unique_dists
    
    # ─── Priority rerank ───
    documents, metadatas = priority_rerank(documents, metadatas, distances)
    
    # ─── Cross-reference chasing ───
    ref_docs, ref_metas = chase_cross_references(
        collection, rule_index, documents, metadatas
    )
    
    # ─── Temporal conflict resolution ───
    all_docs = documents + ref_docs
    all_metas = metadatas + ref_metas
    conflicts = resolve_conflicts(all_docs, all_metas)
    
    # ─── Format context ───
    context_pieces = []
    
    # Main retrieved documents
    context_pieces.append("=== RETRIEVED RULES ===")
    for i, doc in enumerate(documents):
        meta = metadatas[i]
        source = meta.get("source_file", "unknown")
        doc_type = meta.get("doc_type", "unknown")
        rule = meta.get("rule_number", "")
        page = meta.get("page", "?")
        
        header = f"[Source: {source} | Type: {doc_type} | Rule: {rule} | Page: {page}]"
        context_pieces.append(f"{header}\n{doc}\n---")
    
    # Cross-referenced documents
    if ref_docs:
        context_pieces.append("\n=== CROSS-REFERENCED RULES ===")
        for i, doc in enumerate(ref_docs):
            meta = ref_metas[i]
            source = meta.get("source_file", "unknown")
            rule = meta.get("rule_number", "")
            context_pieces.append(f"[Cross-ref: {source} | Rule: {rule}]\n{doc}\n---")
    
    # Conflict annotations
    if conflicts:
        context_pieces.append("\n=== CONFLICT NOTES ===")
        for note in conflicts:
            context_pieces.append(note)
    
    context = "\n".join(context_pieces)
    
    debug_info = {
        "query_type": query_type,
        "num_retrieved": len(documents),
        "num_cross_refs": len(ref_docs),
        "num_conflicts": len(conflicts),
        "rule_numbers": rule_numbers,
        "scenario": scenario
    }
    
    return context, debug_info


# ═══════════════════════════════════════════════════════════════════
# Answer Generation
# ═══════════════════════════════════════════════════════════════════

GENERATION_PROMPT = """You are a Rules Lawyer for the WWII card wargame "Up Front" by Avalon Hill (1983).
You answer rules questions with precision, always citing the specific rule number and source.

CRITICAL INSTRUCTIONS:
1. When rules conflict, the ERRATA/UPDATED version ALWAYS supersedes the original rules.
2. Cite every answer with [Rule X.Y, Source Document] format.
3. If a scenario has special rules that modify the base rule, mention both.
4. If the answer references other rules (cross-references), include those rules in your answer.
5. If tournament rules modify the base rule, note the tournament-specific interpretation.
6. If the information is not in the provided context, say "This is not covered in the available rules documents" — do NOT guess.
7. If variant/house rules exist for this topic, mention them separately and clearly label them as unofficial.

You MUST think step-by-step before answering.
First, use <thinking> tags to identify the specific rules, cross-references, and source documents from the context that apply to the question.
Second, output your final answer formatted with citations.

Retrieved Rules Context:
{context}

User Question: {query}

Detailed Answer (with citations):"""


def generate_answer(query, context):
    """Generate a cited answer using qwen2.5:14b."""
    prompt = GENERATION_PROMPT.format(context=context, query=query)
    
    try:
        response = ollama.generate(model="qwen2.5:14b", prompt=prompt)
        answer = response["response"].strip()
        return answer
    except Exception as e:
        print(f"  Error generating answer: {e}")
        return f"Error generating answer: {e}"


# ═══════════════════════════════════════════════════════════════════
# Main Query Pipeline
# ═══════════════════════════════════════════════════════════════════

def ask_rules_lawyer(query):
    """
    Full query pipeline: classify → route → retrieve → rerank → chase → resolve → generate.
    Returns (answer, context, debug_info).
    """
    print(f"\n{'='*60}")
    print(f"RULES LAWYER QUERY: {query}")
    print(f"{'='*60}")
    
    # Step 1: Classify
    print("\n[1] Classifying query...")
    classification = classify_query(query)
    print(f"    Classification: {classification}")
    
    # Step 2: Route and Retrieve
    print("\n[2] Retrieving relevant rules...")
    context, debug_info = route_and_retrieve(query, classification)
    
    if context == "No matching rules found in the database.":
        return "I could not find any matching rules in the database.", context, debug_info
    
    print(f"    Retrieved {debug_info.get('num_retrieved', 0)} primary chunks")
    print(f"    Chased {debug_info.get('num_cross_refs', 0)} cross-references")
    print(f"    Detected {debug_info.get('num_conflicts', 0)} temporal conflicts")
    
    # Step 3: Generate answer
    print("\n[3] Generating cited answer...")
    answer = generate_answer(query, context)
    
    return answer, context, debug_info


# ═══════════════════════════════════════════════════════════════════
# Interactive CLI
# ═══════════════════════════════════════════════════════════════════

def interactive_mode():
    """Run the Rules Lawyer in interactive CLI mode."""
    print("=" * 60)
    print("  UP FRONT RULES LAWYER")
    print("  Ask any rules question. Type 'quit' to exit.")
    print("=" * 60)
    
    while True:
        print()
        query = input("Your question: ").strip()
        if not query or query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        
        answer, context, debug = ask_rules_lawyer(query)
        
        print(f"\n{'─'*60}")
        print("ANSWER:")
        print(f"{'─'*60}")
        # Safe encode for terminal
        safe_answer = answer.encode('ascii', errors='replace').decode('ascii')
        print(safe_answer)
        print(f"\n[Debug: type={debug.get('query_type')}, "
              f"retrieved={debug.get('num_retrieved', 0)}, "
              f"xrefs={debug.get('num_cross_refs', 0)}, "
              f"conflicts={debug.get('num_conflicts', 0)}]")


if __name__ == "__main__":
    import sys
    profile_arg = next((a for a in sys.argv[1:] if a.endswith("_profile.json")), None)
    if profile_arg:
        load_game_profile(profile_arg)
        interactive_mode_game()
    else:
        interactive_mode()


# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# GENERIC GAME PROFILE SUPPORT — Agnostic RAG Doll
# All functions below are ADDITIVE. Nothing above is changed.
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════

# Active game config — defaults to Up Front settings
_game_config = {
    "game_name": "Up Front",
    "chroma_collection": CHROMA_COLLECTION,
    "rule_index_file": RULE_INDEX_FILE,
    "cross_ref_pattern": None,   # None = use built-in Up Front pattern
    "rule_pattern": None,        # None = use built-in Up Front pattern
    "scenario_format": "letter", # Up Front uses A-K lettered scenarios
}


def load_game_profile(profile_path):
    """
    Load a game profile JSON and update the active game config.
    Call this before ask_rules_lawyer_game() to configure the agent.
    """
    global _game_config
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    _game_config = {
        "game_name": profile["game_name"],
        "chroma_collection": profile["chroma_collection"],
        "rule_index_file": profile["rule_index_file"],
        "cross_ref_pattern": profile.get("cross_ref_pattern"),
        "rule_pattern": profile.get("rule_pattern"),
        "scenario_format": profile.get("scenario_format", "named"),
        "profile": profile,
    }
    print(f"[Game Profile Loaded] {profile['game_name']} — "
          f"collection={profile['chroma_collection']}")
    return _game_config


def _get_active_collection():
    """Return a ChromaDB collection for the active game."""
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection_name = _game_config["chroma_collection"]
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        raise RuntimeError(
            f"ChromaDB collection '{collection_name}' not found. "
            f"Run ingest_game() first."
        )


def _load_active_rule_index():
    """Load the rule index JSON for the active game."""
    index_file = _game_config["rule_index_file"]
    if not os.path.exists(index_file):
        return {}
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _chase_cross_refs_game(text, rule_index, collection):
    """
    Chase cross-references using the active game's cross-ref pattern.
    Falls back to the Up Front pattern if no game pattern is set.
    """
    import re as _re

    cross_ref_pattern_str = _game_config.get("cross_ref_pattern")
    if cross_ref_pattern_str:
        try:
            cross_pat = _re.compile(cross_ref_pattern_str, _re.MULTILINE)
        except _re.error:
            cross_pat = None
    else:
        cross_pat = None

    refs = set()
    if cross_pat:
        for match in cross_pat.finditer(text):
            for group in match.groups():
                if group:
                    refs.add(group)
    else:
        # Fall back to Up Front pattern (bracket and see references)
        for m in _re.finditer(
            r'\[(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\]|'
            r'(?:see|See)\s+(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)',
            text, _re.MULTILINE
        ):
            for g in m.groups():
                if g:
                    refs.add(g)

    # Fetch from rule index and collection
    chased_chunks = []
    for ref in list(refs)[:5]:
        if ref in rule_index:
            chunk_ids = [e["chunk_id"] for e in rule_index[ref][:2]]
            try:
                results = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
                for doc, meta in zip(results["documents"], results["metadatas"]):
                    chased_chunks.append({"text": doc, "metadata": meta, "chased_ref": ref})
            except Exception:
                pass

    return chased_chunks


def build_game_prompt(game_name):
    """Build classification and generation prompts for a specific game."""
    glossary_text = ""
    glossary = _game_config.get("profile", {}).get("glossary")
    if glossary:
        import json
        glossary_text = (
            f"Use this specific glossary to expand abbreviations in your sub-queries:\n"
            f"{json.dumps(glossary, indent=2)}\n\n"
        )
    
    classify = (
        f'You are a query classifier for a reference system for the document collection "{game_name}".\n'
        'Classify the user\'s query into exactly ONE category:\n\n'
        '- "direct_rule": User asks about a specific rule, clause, or section number\n'
        '- "concept": User asks about a general concept or mechanic\n'
        '- "situation": User describes a situation and wants a ruling or clarification\n'
        '- "scenario": User asks about a specific scenario, special case, or addendum\n'
        '- "comparison": User asks about errata, amendments, or changes between versions\n'
        '- "variant": User asks about unofficial, variant, or modified rules\n\n'
        'If the query is complex and requires multiple searches, provide 2-3 sub-queries.\n'
        'CRITICAL INSTRUCTION FOR ABBREVIATIONS: Expand domain-specific abbreviations in your sub-queries to include BOTH the abbreviation and the full term.\n'
        'CRITICAL INSTRUCTION FOR NUMBERS: If a rule or clause has a letter prefix (e.g. A7.2, D5.6), you MUST include both the prefixed AND unprefixed versions in rule_numbers (e.g. ["A7.2", "7.2"]).\n'
        f'{glossary_text}'
        'Respond with ONLY a JSON object: '
        '{"query_type": "<type>", "rule_numbers": [<any rule/clause numbers>], '
        '"scenario": "<scenario id if any>", "sub_queries": ["<sub1>", "<sub2>"]}\n\n'
        'User query: {query}'
    )

    generation = (
        f'You are an authoritative reference assistant for the document collection "{game_name}".\n\n'
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
        '- EVERY factual statement must cite its source document or section number in [brackets]\n'
        '- If an amendment/errata supersedes a base rule, say so explicitly\n'
        '- If the user cites a specific section number but the provided text shows the concept is actually covered by a DIFFERENT section, correct the user and answer using the correct section\n'
        '- If the retrieved context contains text from multiple different editions or versions, formulate the definitive answer based on the LATEST edition available in the context. Then, explicitly conclude your answer by stating that it has changed across versions and ask the user a re-entrant question: "This text has changed across versions. Would you like to know how it worked in a specific version, or how it has evolved over time?"\n'
        '- If you cannot find the answer in the provided text, say so clearly\n'
        '- Be precise and concise\n\n'
        'ANSWER:'
    )

    return classify, generation


def ask_rules_lawyer_game(query, profile_path=None):
    """
    Ask the Rules Lawyer a question for the active game (set by load_game_profile).

    Args:
        query:        The rules question.
        profile_path: Optional path to load a profile before querying.

    Returns:
        (answer, context_chunks, debug_info)
    """
    if profile_path:
        load_game_profile(profile_path)

    game_name = _game_config.get("game_name", "Unknown Game")
    classify_prompt, generation_prompt = build_game_prompt(game_name)

    # 1. Classify query
    try:
        clf_response = ollama.generate(
            model="llama3.1:8b",
            prompt=classify_prompt.format(query=query)
        )
        raw = clf_response["response"].strip()
        json_match = __import__("re").search(r'\{.*?\}', raw, __import__("re").DOTALL)
        classification = json.loads(json_match.group()) if json_match else {}
    except Exception:
        classification = {"query_type": "concept", "rule_numbers": [], "sub_queries": []}

    query_type = classification.get("query_type", "concept")
    rule_numbers = classification.get("rule_numbers", [])
    sub_queries = classification.get("sub_queries", [])

    # 2. Retrieve from ChromaDB
    collection = _get_active_collection()
    rule_index = _load_active_rule_index()

    queries_to_run = sub_queries if sub_queries else [query]
    all_docs, all_metas = [], []

    for q in queries_to_run:
        try:
            emb_resp = ollama.embeddings(model="nomic-embed-text", prompt=q)
            results = collection.query(
                query_embeddings=[emb_resp["embedding"]],
                n_results=8,
                include=["documents", "metadatas"]
            )
            all_docs.extend(results["documents"][0])
            all_metas.extend(results["metadatas"][0])
        except Exception:
            pass

    # Direct rule lookup via index
    for rn in (rule_numbers or []):
        if rn and rn in rule_index:
            for entry in rule_index[rn][:3]:
                try:
                    res = collection.get(
                        ids=[entry["chunk_id"]],
                        include=["documents", "metadatas"]
                    )
                    if res["documents"]:
                        all_docs.extend(res["documents"])
                        all_metas.extend(res["metadatas"])
                except Exception:
                    pass

    # Deduplicate by content
    seen = set()
    unique_docs, unique_metas = [], []
    for doc, meta in zip(all_docs, all_metas):
        key = doc[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
            unique_metas.append(meta)

    # Sort by priority
    paired = sorted(
        zip(unique_docs, unique_metas),
        key=lambda x: x[1].get("priority", 9)
    )

    # Chase cross-references
    all_text_for_chase = "\n".join(d for d, _ in paired[:10])
    chased = _chase_cross_refs_game(all_text_for_chase, rule_index, collection)

    # Build context
    context_parts = []
    for doc, meta in paired[:12]:
        src = meta.get("source_file", "unknown")
        p = meta.get("priority", 9)
        rn = meta.get("rule_number", "")
        header = f"[Source: {src} | Priority: P{p}{' | Rule: ' + rn if rn else ''}]"
        context_parts.append(f"{header}\n{doc}")

    for ch in chased[:4]:
        meta = ch["metadata"]
        context_parts.append(
            f"[Cross-ref: {ch['chased_ref']} | Source: {meta.get('source_file', '')}]\n"
            f"{ch['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # 3. Generate answer
    try:
        gen_response = ollama.generate(
            model="qwen2.5:14b",
            prompt=generation_prompt.format(context=context, query=query)
        )
        answer = gen_response["response"].strip()
        # Strip <thinking> block
        answer = __import__("re").sub(
            r'<thinking>.*?</thinking>', '', answer,
            flags=__import__("re").DOTALL
        ).strip()
    except Exception as e:
        answer = f"Generation error: {e}"

    debug_info = {
        "game_name": game_name,
        "query_type": query_type,
        "rule_numbers": rule_numbers,
        "num_retrieved": len(unique_docs),
        "num_cross_refs": len(chased),
    }

    return answer, paired, debug_info


def interactive_mode_game():
    """Interactive CLI for the active game (set via load_game_profile)."""
    game_name = _game_config.get("game_name", "Unknown Game")

    print("=" * 60)
    print(f"  {game_name.upper()} RULES LAWYER")
    print("  Ask any rules question. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        print()
        query = input("Your question: ").strip()
        if not query or query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        answer, context, debug = ask_rules_lawyer_game(query)

        print(f"\n{'─'*60}")
        print("ANSWER:")
        print(f"{'─'*60}")
        safe = answer.encode("ascii", errors="replace").decode("ascii")
        print(safe)
        print(f"\n[Debug: game={debug.get('game_name')}, "
              f"type={debug.get('query_type')}, "
              f"retrieved={debug.get('num_retrieved', 0)}, "
              f"xrefs={debug.get('num_cross_refs', 0)}]")
