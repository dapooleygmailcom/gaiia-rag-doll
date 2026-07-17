import mysql.connector
import chromadb
import ollama

DB_HOST = "127.0.0.1"
DB_USER = "root"
DB_PASS = "transportme"
DB_NAME = "gaiia_rag"

def run_quantitative():
    print("\n" + "="*50)
    print("1. QUANTITATIVE & STATISTICAL ANALYSIS (MySQL)")
    print("Question: What is the average price and rating of 'Entire home/apt' listings by neighborhood, ranked by the highest guest satisfaction?")
    print("="*50)
    
    conn = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT neighborhood, ROUND(AVG(price), 2) as avg_price, ROUND(AVG(rating), 2) as avg_rating, COUNT(id) as total_listings
        FROM listings 
        WHERE room_type = 'Entire home/apt'
        GROUP BY neighborhood
        ORDER BY avg_rating DESC
        LIMIT 5;
    """)
    rows = cursor.fetchall()
    for row in rows:
        print(f"Neighborhood: {row['neighborhood'].ljust(15)} | Avg Price: ${row['avg_price']} | Avg Rating: {row['avg_rating']} | Total Listings: {row['total_listings']}")
    conn.close()

def run_qualitative():
    print("\n" + "="*50)
    print("2. QUALITATIVE ANALYSIS (ChromaDB + LLM)")
    print("Question: Based on the reviews, what are the primary positive themes and complaints for listings in 'Newtown'?")
    print("="*50)
    
    client = chromadb.PersistentClient(path="data/chroma")
    collection = client.get_collection(name="airbnb-analysis")
    
    # We embed a neutral/mixed sentiment query to fetch a variety of reviews
    query = "guest experience, complaints, dirty, loud, amazing, beautiful, perfect"
    emb = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    results = collection.query(
        query_embeddings=[emb],
        where={"neighborhood": "Newtown"},
        n_results=8,
        include=["documents"]
    )
    
    reviews = "\n".join(results["documents"][0])
    prompt = f"Analyze the following Airbnb reviews for Newtown and summarize the primary positive themes and any negative complaints.\n\nReviews:\n{reviews}\n\nSummary Analysis:"
    res = ollama.generate(model="qwen2.5:14b", prompt=prompt)
    print(res["response"].strip())

def run_decision_making():
    print("\n" + "="*50)
    print("3. EXECUTIVE DECISION MAKING (Hybrid Data)")
    print("Question: I am an investor with a budget constraint. Which neighborhood offers the best balance of affordability (Under $150/night) and high semantic 'romantic getaway' appeal?")
    print("="*50)
    
    client = chromadb.PersistentClient(path="data/chroma")
    collection = client.get_collection(name="airbnb-analysis")
    
    # 1. Strict Database Filtering + Semantic Embedding
    query = "perfect for a romantic getaway, couples, beautiful views, cozy, intimate"
    emb = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    
    # Fetch top 10 under $150
    results = collection.query(
        query_embeddings=[emb],
        where={"price": {"$lte": 150}},
        n_results=10,
        include=["documents", "metadatas"]
    )
    
    context = ""
    for i in range(len(results["ids"][0])):
        md = results["metadatas"][0][i]
        context += f"- Neighborhood: {md['neighborhood']} (Price: ${md['price']}, Rating: {md['rating']}). Review: {results['documents'][0][i]}\n"
        
    prompt = f"You are an investment travel advisor. Based ONLY on the following affordable listings (under $150), which specific neighborhood would you recommend prioritizing for a romantic getaway, and why?\n\nListings:\n{context}\n\nExecutive Recommendation:"
    res = ollama.generate(model="qwen2.5:14b", prompt=prompt)
    print(res["response"].strip())

if __name__ == "__main__":
    run_quantitative()
    run_qualitative()
    run_decision_making()
