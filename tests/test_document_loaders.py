import pytest
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.docvector.core.document_loaders.base import DocumentLoader
from src.docvector.core.document_loaders.text_loader import TextLoader
from src.docvector.core.document_loaders.pdf_loader import PdfLoader
from src.docvector.core.document_loaders.factory import DocumentLoaderFactory

@pytest.fixture
def create_text_file():
    test_file = "test.txt"
    with open(test_file, 'w') as file:
        file.write("Test: Hello, This is File Read Test")
    yield test_file
    os.remove(test_file)

@pytest.fixture
def create_pdf_file():
    pdf_file = "test.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    c.drawString(100,750,"Test: Hello, This is File Read Test")
    c.save()
    yield pdf_file
    os.remove(pdf_file)

def test_load_text_file(create_text_file):
    text_loader = TextLoader()
    file_text = text_loader.load(create_text_file)
    assert file_text == "Test: Hello, This is File Read Test"

def test_load_pdf_file(create_pdf_file):
    pdf_loader = PdfLoader()
    file_text = pdf_loader.load(create_pdf_file)
    assert file_text == "Test: Hello, This is File Read Test\n"

def test_document_factory(create_text_file, create_pdf_file):
    factory = DocumentLoaderFactory()
    text_loader = factory.get_loader(create_text_file)
    assert isinstance(text_loader, DocumentLoader)
    assert isinstance(text_loader, TextLoader)

    pdf_loader = factory.get_loader(create_pdf_file)
    assert isinstance(pdf_loader, DocumentLoader)
    assert isinstance(pdf_loader, PdfLoader)

def test_unsupported_file_format():
    factory = DocumentLoaderFactory()
    with pytest.raises(ValueError, match="Unsupported File Format"):
        factory.get_loader("unsupported_file_format.docx")

