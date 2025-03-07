from typing import List
from openai import OpenAI, OpenAIError
from src.docvector.core.embedders.base import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name:str="text-embedding-3-small"):
        self.model_name = model_name

    def embed(self, text:str) -> List[float]:
        try:
            openai_client = OpenAI()
            response = openai_client.embeddings.create(
                model= self.model_name,
                input= text
                )
            return response.data[0].embedding
        except OpenAIError as error:
            raise(OpenAIError(f"OpenAI API Error: {error}"))
        
        return []
        
        
