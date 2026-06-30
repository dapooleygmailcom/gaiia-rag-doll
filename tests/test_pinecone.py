import os
import time
from pinecone import Pinecone

def test_pinecone_local():
    print("Connecting to Pinecone Local Emulator on port 5081...")
    try:
        # Initialize the Pinecone client pointing to the local emulator
        pc = Pinecone(
            api_key="pclocal", 
            host="http://localhost:5081"
        )
        
        index_name = "test-index"
        
        # Create the index if it doesn't exist (using the local emulator)
        if not pc.has_index(index_name):
            from pinecone import ServerlessSpec
            print(f"Creating index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=4,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
        index = pc.Index(index_name)
        
        print(f"Client initialized for index: {index_name}")
        
        print("Upserting a test vector (dimension 4)...")
        index.upsert(vectors=[
            {"id": "test_doc_1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"carrier": "TestCarrier"}}
        ])
        print("Upsert successful!")
        
        # Small delay to ensure eventual consistency (though local is usually instant)
        time.sleep(1)
        
        print("Fetching the test vector...")
        response = index.fetch(ids=["test_doc_1"])
        
        if "test_doc_1" in response.get("vectors", {}):
            print("Fetch successful. Vector retrieved!")
            print("Metadata:", response["vectors"]["test_doc_1"].get("metadata"))
        else:
            print("Vector not found in response.")
            print(response)
            
    except Exception as e:
        print(f"Pinecone Local connection failed: {e}")

if __name__ == "__main__":
    test_pinecone_local()
