import os
import pytest
import chromadb
import ollama

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data/chroma")
CHROMA_COLLECTION = "renegade_legion-entities"

@pytest.fixture
def collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        col = client.get_collection(name=CHROMA_COLLECTION)
        return col
    except Exception as e:
        pytest.skip(f"Entity collection not found, skipping RAG test: {e}")

def test_rag_query_entity_stats(collection):
    """Test that querying for specific vehicle stats retrieves the right entity."""
    query = "What is the Maximum Thrust of a Liberator?"
    response = ollama.embeddings(model="nomic-embed-text", prompt=query)
    
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=3,
        include=["documents", "metadatas"]
    )
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    # We should have found some documents
    assert len(docs) > 0
    
    # At least one result should be the Liberator entity
    liberator_found = any("Liberator" in meta.get("entity_name", "") for meta in metas)
    assert liberator_found, f"Liberator not found in top 3 results for query: {query}"
    
    # The text should explicitly mention the thrust
    doc_text = " ".join(docs)
    assert "Maximum Thrust" in doc_text

def test_rag_query_missile_loadout(collection):
    """Test retrieving missile loadouts."""
    query = "How many TVLG rounds does a Horatius have?"
    response = ollama.embeddings(model="nomic-embed-text", prompt=query)
    
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=3,
        include=["documents", "metadatas"]
    )
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    horatius_found = any("Horatius" in meta.get("entity_name", "") for meta in metas)
    assert horatius_found, "Horatius not found for missile query."
    
    # Horatius has TVLG rounds
    doc_text = "\n".join(docs)
    assert "TVLG" in doc_text
