from typing import List
from src.docvector.core.text_splitters.base import TextSplitter

class CharacterTextSplitter(TextSplitter):
    def __init_(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("Chunk Size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("Chunk Overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError ("Chunk Overlap must be less than Chunk Size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text:str) -> List[str]:
        chunks = []
        start_point = 0
        while start_point < len(text):
            end_point = start_point + self.chunk_size
            chunks.append(text[start_point:end_point])
            start_point += (self.chunk_size - self.chunk_overlap)
        return chunks