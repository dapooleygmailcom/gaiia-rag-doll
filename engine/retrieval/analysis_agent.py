import json
import ollama
import chromadb

CHROMA_DB_DIR = "data/chroma"
CHROMA_COLLECTION_NAME = "airbnb-analysis"

def extract_filters(query):
    # Ask Llama3 to extract price and rating constraints
    prompt = f"""
    Extract the maximum price and minimum rating from the user's query if they exist.
    Return ONLY a JSON object with keys 'price_max' and 'rating_min'. If not mentioned, set the value to null.
    User Query: "{query}"
    """
    response = ollama.generate(model="qwen2.5:7b", prompt=prompt, format="json")
    response_text = response['response'].strip()
    
    try:
        return json.loads(response_text)
    except Exception as e:
        print(f"Warning: Could not parse LLM JSON: {response_text}")
        return {"price_max": None, "rating_min": None}

def build_chroma_filter(extracted):
    filters = []
    if extracted.get("price_max") is not None:
        filters.append({"price": {"$lte": float(extracted["price_max"])}})
    if extracted.get("rating_min") is not None:
        filters.append({"rating": {"$gte": float(extracted["rating_min"])}})
        
    if len(filters) == 0:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        return {"$and": filters}

def search_airbnb(query):
    print(f"Analyzing Query: '{query}'")
    
    # 1. Extract metadata filters via Llama3
    print("Extracting structured filters with Llama3...")
    extracted_filters = extract_filters(query)
    chroma_filter = build_chroma_filter(extracted_filters)
    print(f"Applied Metadata Filters: {chroma_filter}")
    
    # 2. Embed semantic portion
    print("Generating semantic embedding...")
    emb_res = ollama.embeddings(model="nomic-embed-text", prompt=query)
    vector = emb_res["embedding"]
    
    # 3. Query ChromaDB
    print("Querying ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        
        search_args = {
            "query_embeddings": [vector],
            "n_results": 3,
            "include": ["metadatas", "documents", "distances"]
        }
        if chroma_filter:
            search_args["where"] = chroma_filter
            
        results = collection.query(**search_args)
        
        context_pieces = []
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                md = results["metadatas"][0][i]
                review = results["documents"][0][i]
                context_pieces.append(f"- Listing {doc_id} in {md.get('neighborhood')}: {md.get('room_type')} at ${md.get('price')} (Rating: {md.get('rating')}). Review: {review}")
        
        return "\n".join(context_pieces) if context_pieces else "No matching listings found."
    except Exception as e:
        print(f"Failed to query ChromaDB: {e}")
        return ""

def generate_rag_response(query):
    context = search_airbnb(query)
    
    prompt = f"""
    You are a helpful travel assistant. Using ONLY the following Airbnb listings, answer the user's question.
    If there are no listings or the answer cannot be found in the context, say "I could not find any matching listings."
    
    Listings Context:
    {context}
    
    User Question: {query}
    
    Answer:
    """
    
    response = ollama.generate(model="qwen2.5:7b", prompt=prompt)
    answer = response['response'].strip()
    return answer, context

if __name__ == "__main__":
    test_queries = [
        "Find me a quiet neighborhood under $150."
    ]
    for q in test_queries:
        print(f"\n--- Testing Query: {q} ---")
        answer, context = generate_rag_response(q)
        print(f"\nRetrieved Context:\n{context}")
        print(f"\nFinal Llama3 Answer:\n{answer}")
        print("-" * 50)
