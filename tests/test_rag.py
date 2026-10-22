import pytest
from recall_loop.rag import answer_why, build_rag_prompt
from recall_loop.models import RagAnswer
from recall_loop.config import get_settings

def test_rag_prompt_includes_sources():
    chunks = [
        {"id": "a", "text": "Python is fun.", "source_file": "notes.txt", "page": None, "line_start": 1},
    ]
    prompt = build_rag_prompt("What is Python?", chunks)
    assert "notes.txt" in prompt
    assert "Python is fun." in prompt

def test_why_not_covered_prints_message(capsys):
    settings = get_settings()
    settings.why_top_k = 2
    chunks = [
        {"id": "a", "text": "Python is fun.", "source_file": "notes.txt", "page": None, "line_start": 1},
    ]

    def fake_retrieve(q, col, s, flagged_ids=None, always_include_id=None):
        return chunks

    def fake_llm(prompt, model_class, s):
        return RagAnswer(answer="I don't know.", used_sources=[], covered=False)

    answer_why("What is gravity?", "test", "a", set(), settings,
               retrieve_fn=fake_retrieve, llm_call=fake_llm)
    captured = capsys.readouterr()
    assert "don't seem to cover" in captured.out

def test_why_always_includes_current_chunk():
    settings = get_settings()
    included = []

    def fake_retrieve(q, col, s, flagged_ids=None, always_include_id=None):
        included.append(always_include_id)
        return [{"id": always_include_id, "text": "t", "source_file": "f", "page": None, "line_start": 1}]

    def fake_llm(prompt, model_class, s):
        return RagAnswer(answer="Yes.", used_sources=["f:L1"], covered=True)

    answer_why("Why?", "test", "mychunk", set(), settings,
               retrieve_fn=fake_retrieve, llm_call=fake_llm)
    assert "mychunk" in included

def test_why_excludes_flagged_chunks():
    settings = get_settings()
    retrieved_flagged = []

    def fake_retrieve(q, col, s, flagged_ids=None, always_include_id=None):
        retrieved_flagged.append(flagged_ids)
        return []

    def fake_llm(prompt, model_class, s):
        return RagAnswer(answer="", used_sources=[], covered=False)

    flagged = {"bad_chunk"}
    answer_why("Why?", "test", "current", flagged, settings,
               retrieve_fn=fake_retrieve, llm_call=fake_llm)
    assert "bad_chunk" in retrieved_flagged[0]
