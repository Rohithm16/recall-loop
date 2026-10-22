import pytest
from dotenv import load_dotenv

load_dotenv()

def pytest_configure(config):
    config.addinivalue_line("markers", "live: marks tests as requiring a live Gemini API key")
