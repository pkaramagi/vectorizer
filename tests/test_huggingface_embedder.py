import pytest
from src.docvector.core.embedders.huggingface_embedder import HuggingFaceEmbedder

@pytest.fixture
def embedder():
    return HuggingFaceEmbedder()

def test_embed(embedder):
    text = "This is a test sentence."
    embedding = embedder.embed(text)
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)

def test_embed_batch(embedder):
    texts = ["This is the first test sentence.", "This is the second test sentence."]
    embeddings = embedder.embed_batch(texts)
    assert isinstance(embeddings, list)
    assert all(isinstance(embedding, list) for embedding in embeddings)
    assert all(isinstance(x, float) for embedding in embeddings for x in embedding)
