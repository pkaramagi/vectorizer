from typing import Any, Dict, List
from elasticsearch import Elasticsearch
from core.vectors.base import VectorDB


class ElasticSearchDB(VectorDB):
    def __init__(self, index_name:str, embedding_dim:int, host:str ="localhost", port:int= 9200,):
        self.client:Elasticsearch = Elasticsearch(
            [{'host':host, 'port':port}]
        )
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self._create_index()
        
    @property
    def _mapping(self) -> Dict[str, Any]:
        return  {
            "mapping":{
                "properties":{
                    "id": {"type":"keyword"},
                    "text": {"type":"text"},
                    "embedding":{"type":"dense_vector","dims":self.embedding_dim}
                }
            }
        }
    
    @property
    def _doc(self) -> Dict[str, Any]:
        return {
            "id": None,
            "text": "",
            "embedding": []
        }


    def _create_index(self) -> None:
        
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=self._mapping)
            print(f"index '{self.index_name}' created successfully.")
        


    def save_vectors(self, vectors:List[Dict[str, Any]]):
        for vector in vectors:
            document_id = vector.get("id")
            if not document_id:
                raise ValueError("Vector Must have an 'id field")
            
            doc = self._doc.copy()
            doc['id'] = document_id
            doc['text'] = vector.get("text","")
            doc['embedding'] = vector.get("embedding", [])
            
           
            self.client.index(index=self.index_name, id=document_id, body=doc)
            print(f"saved Vector {document_id}")

    