import pytest
import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.docvector.core.document_loaders.text_loader import TextLoader
from src.docvector.core.document_loaders.pdf_loader import PdfLoader

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
