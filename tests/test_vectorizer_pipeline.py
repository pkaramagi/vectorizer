import pytest
from unittest.mock import MagicMock

from src.docvector.core.vectorizer_pipeline import VectorizationPipeline
from src.docvector.core.text_splitters.base import TextSplitter
from src.docvector.core.embedders.base import Embedder
from src.docvector.core.vectors.base import VectorDB

@pytest.fixture
def mock_splitter():
    splitter = MagicMock(spec=TextSplitter)
    splitter.split.return_value = ["chunk1", "chunk2"]
    return splitter

@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=Embedder)
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    return embedder

@pytest.fixture
def mock_vector_db():
    vector_db = MagicMock(spec=VectorDB)
    return vector_db

def test_process_document(mock_splitter, mock_embedder, mock_vector_db):
    pipeline = VectorizationPipeline(splitter=mock_splitter, embedder=mock_embedder, vector_db=mock_vector_db)
    pipeline.process_document("test.txt")
    mock_splitter.split.assert_called_once()
    mock_embedder.embed.assert_called()
    mock_vector_db.save_vectors.assert_called_once()
