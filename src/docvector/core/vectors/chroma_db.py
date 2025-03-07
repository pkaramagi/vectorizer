from typing import Any, Dict, List
import chromadb
from src.docvector.core.vectors.base import VectorDB


class ChromaDB(VectorDB):
    def __init__(self, collection_name: str, embedding_dim, persist_dir:str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

    @property
    def _doc(self) -> Dict[str, Any]:
        return {
            "id": None,
            "text": "",
            "embedding": []
        }    
    
    def _create_collection(self) -> None:
        collection = self.client.get_or_create_collection(self.collection_name)
        if collection is not None:
            print(f"Collection '{self.collection_name} is ready for use.")

    def save_vectors(self, vectors: List[Dict[str, Any]]):
        collection = self.client.get_collection(self.collection_name)
        ids = []
        texts = []
        embeddings = []

        for vector in vectors:
            document_id = vector.get("id")
            if not document_id:
                raise ValueError("Vector Must have an 'id' field" )
            
            ids.append(document_id)
            texts.append(vector.get("text",""))
            embeddings.append(vector.get("embedding",[]))
        
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings
        )
        print(f"Saved {len(vectors)} vectors to collection '{self.collection_name}'")
        