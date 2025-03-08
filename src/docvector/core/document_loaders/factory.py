from core.document_loaders.base import DocumentLoader
from core.document_loaders.pdf_loader import PdfLoader
from core.document_loaders.text_loader import TextLoader


class DocumentLoaderFactory:
    @staticmethod
    def get_loader(path:str) -> DocumentLoader:
        if path.endswith('txt'):
            return TextLoader()
        elif path.endswith('pdf'):
            return PdfLoader()
        raise ValueError("Unsupported File Format")
    
    