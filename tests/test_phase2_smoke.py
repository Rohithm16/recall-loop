import pytest
import os
from recall_loop.config import get_settings

# Smoke test: only runs when GEMINI_API_KEY is set
pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set"
)

@pytest.mark.live
def test_llm_structured_output():
    from recall_loop.llm import call_structured
    from recall_loop.models import TopicLabels
    settings = get_settings()
    result = call_structured("Label these chunks: Chunk 1: Python is fun.", TopicLabels, settings)
    assert isinstance(result.labels, list)

@pytest.mark.live
def test_topic_labeling_live(tmp_path, monkeypatch):
    import recall_loop.store as store_mod
    from recall_loop.topics import label_chunks
    settings = get_settings()
    chunks = [
        {"id": "aaa", "text": "Python variables are dynamically typed.", "topic": "General",
         "source_file": "test.txt", "page": None, "line_start": 1},
    ]
    labeled = label_chunks(chunks, {}, settings)
    assert labeled[0]["topic"] != "" and labeled[0]["topic"] is not None
