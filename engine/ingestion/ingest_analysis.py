import os
import csv
import mysql.connector
import chromadb
import ollama

DB_HOST = "127.0.0.1"
DB_USER = "root"
DB_PASS = "transportme"
DB_NAME = "gaiia_rag"
CSV_FILE = "data/analysis/airbnb_listings.csv"

def setup_mysql():
    print("Setting up MySQL Database...")
    conn = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASS)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    cursor.close()
    conn.close()
    
    conn = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS listings (
            id INT PRIMARY KEY,
            neighborhood VARCHAR(255),
            room_type VARCHAR(255),
            price DECIMAL(10, 2),
            rating DECIMAL(3, 1),
            review_text TEXT
        )
    """)
    conn.commit()
    return conn, cursor

CHROMA_DB_DIR = "data/chroma"
CHROMA_COLLECTION_NAME = "airbnb-analysis"

def setup_vector_db():
    print("Setting up ChromaDB (Persistent Local)...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def ingest_data():
    conn, cursor = setup_mysql()
    collection = setup_vector_db()
    
    print("Reading CSV and generating embeddings...")
    batch_ids = []
    batch_embeddings = []
    batch_metadatas = []
    batch_documents = []
    
    with open(CSV_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            count += 1
            # Limit to 500 for prototyping speed so you don't wait 10 minutes
            if count > 500: 
                break
                
            cursor.execute("""
                INSERT IGNORE INTO listings (id, neighborhood, room_type, price, rating, review_text)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (row["id"], row["neighborhood"], row["room_type"], row["price"], row["rating"], row["review_text"]))
            
            response = ollama.embeddings(model="nomic-embed-text", prompt=row["review_text"])
            embedding = response["embedding"]
            
            metadata = {
                "neighborhood": row["neighborhood"],
                "room_type": row["room_type"],
                "price": float(row["price"]),
                "rating": float(row["rating"])
            }
            batch_ids.append(f"listing_{row['id']}")
            batch_embeddings.append(embedding)
            batch_metadatas.append(metadata)
            batch_documents.append(row["review_text"])
            
            if len(batch_ids) >= 100:
                print(f"Upserting batch of 100 to ChromaDB... (Total processed: {count})")
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                batch_ids, batch_embeddings, batch_metadatas, batch_documents = [], [], [], []
                conn.commit()
                
        if batch_ids:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )
            conn.commit()
            
    cursor.close()
    conn.close()
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
