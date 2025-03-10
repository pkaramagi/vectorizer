import pytest
from unittest.mock import patch, MagicMock
from src.docvector.core.embedders.openai_embedder import OpenAIEmbedder
from openai import OpenAIError

@pytest.fixture
def embedder():
    return OpenAIEmbedder(model_name="text-embedding-3-small")

def test_embed_success(embedder):
    mock_response = MagicMock()
    mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
    
    with patch('core.embedders.openai_embedder.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.embeddings.create.return_value = mock_response
        result = embedder.embed("test text")
        assert result == [0.1, 0.2, 0.3]

def test_embed_failure(embedder):
    with patch('core.embedders.openai_embedder.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.embeddings.create.side_effect = OpenAIError("API Error")
        with pytest.raises(OpenAIError, match="OpenAI API Error: API Error"):
            embedder.embed("test text")
