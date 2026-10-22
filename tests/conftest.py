import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "live: marks tests as requiring a live Gemini API key")
