import os
import re
import chromadb
import ollama

CHROMA_DB_DIR = "data/chroma"
CHROMA_COLLECTION_NAME = "policy-comparison"

def extract_keywords(query):
    """
    Clean the query and extract core nouns/keywords for BM25-style keyword boosting.
    """
    cleaned = re.sub(r'[^\w\s]', '', query.lower())
    words = cleaned.split()
    stopwords = {
        "what", "is", "the", "and", "or", "of", "to", "in", "a", "for", "with", "on", 
        "at", "by", "from", "under", "across", "vs", "versus", "compare", "coverage", 
        "exclusions", "policy", "policies", "insurance", "pds", "carrier", "carriers", 
        "limits", "maximum", "limit", "amount", "any", "which", "who", "offer", "offers",
        "detailed", "sidebyside", "comparison", "between", "how", "does", "do", "they",
        "terms", "conditions", "cover", "covered", "exclusions", "excluded", "stance"
    }
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords

def get_sliding_window_context(collection, chunk_id, carrier, original_doc):
    """
    Fetches the preceding and succeeding chunks to expand the context window around the match.
    """
    match = re.match(r'^(.*)_chunk_(\d+)$', chunk_id)
    if not match:
        return original_doc
        
    prefix = match.group(1)
    num = int(match.group(2))
    
    neighbor_ids = []
    if num > 1:
        neighbor_ids.append(f"{prefix}_chunk_{num - 1}")
    neighbor_ids.append(f"{prefix}_chunk_{num + 1}")
    
    try:
        results = collection.get(ids=neighbor_ids, include=["documents"])
        if results and results.get("documents"):
            # Sort the neighbors in reading order (num-1 first, then original_doc, then num+1)
            docs = results["documents"]
            ids = results["ids"]
            
            prev_doc = ""
            next_doc = ""
            
            for idx, n_id in enumerate(ids):
                if n_id == f"{prefix}_chunk_{num - 1}":
                    prev_doc = docs[idx]
                elif n_id == f"{prefix}_chunk_{num + 1}":
                    next_doc = docs[idx]
                    
            # Combine them, keeping header info clean
            combined = []
            if prev_doc:
                # Strip metadata header prefix to avoid repeating it
                clean_prev = re.sub(r'^\[Carrier:.*?\]\s*', '', prev_doc)
                combined.append(clean_prev)
                
            combined.append(original_doc)
            
            if next_doc:
                clean_next = re.sub(r'^\[Carrier:.*?\]\s*', '', next_doc)
                combined.append(clean_next)
                
            return "\n".join(combined)
    except Exception as e:
        print(f"    Warning: Sliding window expansion failed for {chunk_id}: {e}")
        
    return original_doc

def resolve_cross_references(collection, text, carrier, f_name, existing_pages):
    """
    Scans chunk text for page references (e.g. "page 33", "pages 27-41"),
    queries ChromaDB for those pages from the same carrier, and returns them.
    """
    page_refs = []
    # Match patterns like "page 33", "pages 27-41", "page 40"
    matches = re.findall(r'(?:page|pages)\s+(\d+)', text, re.IGNORECASE)
    for m in matches:
        p_num = int(m)
        if p_num not in existing_pages:
            page_refs.append(p_num)
            
    if not page_refs:
        return "", []
        
    print(f"  Detected cross-references to pages: {page_refs} for {carrier}")
    ref_contexts = []
    resolved_pages = []
    
    for page in page_refs:
        try:
            # Query ChromaDB for chunks belonging to this carrier and page
            res = collection.get(
                where={"$and": [{"carrier": carrier}, {"page": page}]},
                include=["documents"]
            )
            if res and res.get("documents"):
                ref_text = "\n".join(res["documents"])
                ref_contexts.append(f"=== CROSS-REFERENCE: {carrier} Page {page} (from {f_name}) ===\n{ref_text}\n")
                resolved_pages.append(page)
        except Exception as e:
            print(f"    Failed to retrieve cross-reference {carrier} Page {page}: {e}")
            
    return "\n".join(ref_contexts), resolved_pages

def search_policies(query, carriers=None, n_results=5):
    """
    Search the policy collection in ChromaDB using a hybrid semantic-keyword ranking approach,
    expanding adjacent context windows and resolving cross-references recursively.
    """
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        query_vector = response["embedding"]
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return ""
        
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        
        where_filter = None
        if carriers:
            if len(carriers) == 1:
                where_filter = {"carrier": carriers[0]}
            else:
                where_filter = {"carrier": {"$in": list(carriers)}}
                
        # Retrieve candidate pool (2.5x the requested results)
        oversample_n = int(n_results * 2.5)
        search_args = {
            "query_embeddings": [query_vector],
            "n_results": oversample_n,
            "include": ["metadatas", "documents", "distances"]
        }
        if where_filter:
            search_args["where"] = where_filter
            
        results = collection.query(**search_args)
        
        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            return "No matching policy documents found."
            
        # Hybrid Keyword Boosting
        keywords = extract_keywords(query)
        print(f"  Extracted keywords for hybrid boosting: {keywords}")
        
        scored_chunks = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for i in range(len(ids)):
            text = docs[i].lower()
            match_count = sum(1 for kw in keywords if kw in text)
            # Boost score (cosine distance subtracts weight)
            boost = match_count * 0.12
            final_score = distances[i] - boost
            
            scored_chunks.append({
                "id": ids[i],
                "text": docs[i],
                "metadata": metadatas[i],
                "distance": distances[i],
                "boost": boost,
                "final_score": final_score
            })
            
        # Re-sort and keep top n_results
        scored_chunks = sorted(scored_chunks, key=lambda x: x["final_score"])
        top_chunks = scored_chunks[:n_results]
        
        # Track pages we have already included to avoid duplicate reference fetching
        included_pages_per_carrier = {}
        
        # Group results and perform sliding window expansion
        grouped_context = {}
        for chunk in top_chunks:
            chunk_id = chunk["id"]
            md = chunk["metadata"]
            doc_text = chunk["text"]
            
            carrier = md.get("carrier")
            page = md.get("page")
            source = md.get("source")
            section = md.get("section")
            subsection = md.get("subsection")
            
            if carrier not in included_pages_per_carrier:
                included_pages_per_carrier[carrier] = set()
            included_pages_per_carrier[carrier].add(page)
            
            # 1. SLIDING WINDOW CONTEXT EXPANSION (Parent-Child & Sibling binding)
            expanded_text = get_sliding_window_context(collection, chunk_id, carrier, doc_text)
            
            if carrier not in grouped_context:
                grouped_context[carrier] = []
                
            grouped_context[carrier].append({
                "page": page,
                "source": source,
                "section": section,
                "subsection": subsection,
                "text": expanded_text
            })
            
        # 2. RECURSIVE CROSS-REFERENCE RESOLUTION
        cross_ref_contexts = []
        total_context_chars = sum(len(c["text"]) for k in grouped_context for c in grouped_context[k])
        
        for carrier, items in list(grouped_context.items()):
            for item in items:
                # Guardrail: stop adding references if total context size exceeds 18,000 chars
                if total_context_chars > 18000:
                    break
                    
                existing_pages = included_pages_per_carrier.get(carrier, set())
                ref_text, resolved_pages = resolve_cross_references(
                    collection, item["text"], carrier, item["source"], list(existing_pages)
                )
                if ref_text:
                    cross_ref_contexts.append(ref_text)
                    total_context_chars += len(ref_text)
                    # Mark these pages as resolved to prevent duplicate queries
                    for rp in resolved_pages:
                        included_pages_per_carrier[carrier].add(rp)
                        
        # 3. Stringify grouped context
        formatted_context_pieces = []
        for carrier, items in grouped_context.items():
            formatted_context_pieces.append(f"=== Carrier: {carrier} ===")
            for item in items:
                formatted_context_pieces.append(
                    f"Source: {item['source']} | Page: {item['page']} | Section: {item['section']} | Subsection: {item['subsection']}"
                )
                formatted_context_pieces.append(f"Content:\n{item['text']}\n---\n")
                
        # Append cross-references if any were found
        if cross_ref_contexts:
            formatted_context_pieces.append("=== SUPPORTING CROSS-REFERENCE DOCUMENTS ===")
            formatted_context_pieces.extend(cross_ref_contexts)
            
        return "\n".join(formatted_context_pieces) if formatted_context_pieces else "No matching policy documents found."
        
    except Exception as e:
        print(f"Failed to query ChromaDB for policies: {e}")
        return "Error querying policy database."

def compare_policies(query, carriers=None):
    """
    Query the policies in ChromaDB and use Llama3 to generate a comparative analysis.
    """
    print(f"Analyzing comparative query: '{query}'")
    if carriers:
        print(f"Filtering by carriers: {carriers}")
        # Dynamically scale results
        n_results = max(4, min(6, 2 * len(carriers)))
    else:
        n_results = 5
        
    # Retrieve relevant policy extracts (with sliding window and cross-references)
    context = search_policies(query, carriers=carriers, n_results=n_results)
    
    if context == "No matching policy documents found.":
        return "I could not find any matching policy details to compare.", context
        
    prompt = f"""
    You are an expert insurance analyst. Analyze and compare the provided Home & Contents policy extracts below to answer the user's comparison question.
    
    CRITICAL INSTRUCTIONS:
    1. Organize your response clearly. Where applicable, use a Markdown table to compare features, limits, and exclusions across the carriers.
    2. Cite the source document, page numbers, and specific section names for EVERY detail you include (e.g., "AAMI Page 12, Section: Exclusions").
    3. If the information is present in a SUPPORTING CROSS-REFERENCE section, note that you traced the reference (e.g., "Allianz Page 40 referencing Page 33").
    4. If the information is not present in the extracts for a carrier, explicitly state "Not mentioned in provided extracts".
    
    Retrieved Policy Extracts:
    {context}
    
    User Comparison Question: {query}
    
    Detailed Side-by-Side Comparison:
    """
    
    try:
        response = ollama.generate(model="qwen2.5:14b", prompt=prompt)
        answer = response["response"].strip()
        return answer, context
    except Exception as e:
        print(f"Error generating comparison response: {e}")
        return f"Error generating comparison response: {e}", context

if __name__ == "__main__":
    # Test query
    q = "Compare the flood cover and policy exclusions for flood across AAMI and CBA."
    ans, ctx = compare_policies(q, carriers=["AAMI", "CBA"])
    safe_ans = ans.encode('ascii', errors='replace').decode('ascii')
    print("\n--- Response ---")
    print(safe_ans)
