import pytest
from src.docvector.core.vectors.chroma_db import ChromaDB

@pytest.fixture
def chroma_db():
    return ChromaDB(collection_name="test_collection", embedding_dim=128, persist_dir="./test_chroma_db")

def test_create_collection(chroma_db):
    assert chroma_db.client.get_collection("test_collection") is not None

def test_save_vectors(chroma_db):
    vectors = [
        {"id": "1", "text": "document 1", "embedding": [0.1] * 128},
        {"id": "2", "text": "document 2", "embedding": [0.2] * 128},
    ]
    chroma_db.save_vectors(vectors)
    collection = chroma_db.client.get_collection("test_collection")
    assert collection.count() == 2

def test_search_vectors(chroma_db):
    vectors = [
        {"id": "1", "text": "document 1", "embedding": [0.1] * 128},
        {"id": "2", "text": "document 2", "embedding": [0.2] * 128},
    ]
    chroma_db.save_vectors(vectors)
    results = chroma_db.search_vectors([0.1] * 128, top_k=1)
    assert results is not None
    assert len(results["metadatas"]) == 1

def test_remove_collection(chroma_db):
    vectors = [
        {"id": "1", "text": "document 1", "embedding": [0.1] * 128},
        {"id": "2", "text": "document 2", "embedding": [0.2] * 128},
    ]
    chroma_db.save_vectors(vectors)
    chroma_db.remove_collection()
    collection = chroma_db.client.get_collection("test_collection")
    assert collection.count() == 0
