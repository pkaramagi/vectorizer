from typing import List
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.docvector.core.text_splitters.character_splitter import CharacterTextSplitter



@pytest.fixture
def create_character_splitter()-> CharacterTextSplitter:
    char_splitter = CharacterTextSplitter()

    return char_splitter

@pytest.fixture
def create_test_data() -> str:
    text = """Practical Scenarios of Mimesis in Simulated Data Generation

Data simulation is an essential part of proejct development. It involves generating fictitious data that mimics actual data patterns, useful for testing, documentation, teaching, or any scenario where real user data isn’t required.

Mimesis is a powerful Python library for generating various types of simulated data. The following are some typical use cases for Mimesis:

Generating Test Data Sample
During development, testing is crucial to ensuring software quality. Mimesis can generate test data, such as user information, order details, financial transaction records, and more. This data helps developers create comprehensive test cases to validate software behavior across different scenarios.
"""
    return text

def test_character_splitter(create_character_splitter, create_test_data):

    char_splitter = create_character_splitter
    char_splitter.chunk_size = 10
    char_splitter.chunk_overlap = 5
    splits = char_splitter.split(create_test_data)
    assert splits[0] == "Practical "