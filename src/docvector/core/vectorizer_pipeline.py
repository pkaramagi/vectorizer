import hashlib
from pathlib import Path
from core.document_loaders.factory import DocumentLoaderFactory
from core.embedders.base import Embedder
from core.text_splitters.base import TextSplitter
from core.vectors.base import VectorDB


class VectorizationPipeline:
    def __init__(self, splitter: TextSplitter, embedder: Embedder, vector_db: VectorDB):
        self.splitter = splitter
        self.embedder = embedder
        self.vector_db = vector_db

    def process_document(self, path:str) -> None:
        doc_loader = DocumentLoaderFactory.get_loader(path)
        text = doc_loader.load(path)
        chunks = self.splitter.split(text)

        vectors = []
        for i, chunk in enumerate(chunks):
            vector_id = hashlib.md5(f"{path}_{i}".encode()).hexdigest()
            embedding = self.embedder.embed(chunk)
            vectors.append({
                'id': vector_id,
                'document': chunk,
                'embedding': embedding
            })

        self.vector_db.save_vectors(vectors)

    def process_directory(self, directory_path:str) -> None:
        path = Path(directory_path)
        for file_path in path.glob('*'):
            if file_path.suffix in ['.pdf','.txt']:
                self.process_document(str(file_path))
