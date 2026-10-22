import pytest
from recall_loop.models import AppState, MCQ, Option
from recall_loop.mistakes import MistakeType
from recall_loop.tracker import record_result
from recall_loop.graph import (
    print_feedback, chunk_source_label, build_graph,
    node_route_answer, node_route_post, SessionState
)

def make_valid_mcq() -> MCQ:
    return MCQ(
        question="What is readability?",
        options=[
            Option(text="A Python priority", mistake_type=None),
            Option(text="Speed", mistake_type=MistakeType.wrong_detail),
            Option(text="Complexity", mistake_type=MistakeType.reversed),
            Option(text="All languages", mistake_type=MistakeType.overgeneralized),
        ],
        correct_index=0,
        explanation="Python emphasizes readability.",
    )

def test_pattern_line_shown_for_wrong_answer(capsys):
    chunk = {"source_file": "f.txt", "page": None, "line_start": 1}
    print_feedback(make_valid_mcq(), chosen_index=1, correct=False, chunk=chunk)
    out = capsys.readouterr().out
    assert "Your choice looked like:" in out
    assert "wrong detail" in out

def test_pattern_line_not_shown_for_correct(capsys):
    chunk = {"source_file": "f.txt", "page": None, "line_start": 1}
    print_feedback(make_valid_mcq(), chosen_index=0, correct=True, chunk=chunk)
    assert "Your choice looked like" not in capsys.readouterr().out

def test_pattern_line_not_shown_for_other(capsys):
    mcq = MCQ(
        question="Q?",
        options=[
            Option(text="Right", mistake_type=None),
            Option(text="Wrong", mistake_type=MistakeType.other),
            Option(text="Also", mistake_type=MistakeType.wrong_detail),
            Option(text="More", mistake_type=MistakeType.reversed),
        ],
        correct_index=0,
        explanation="E",
    )
    chunk = {"source_file": "f.txt", "page": None, "line_start": 1}
    print_feedback(mcq, chosen_index=1, correct=False, chunk=chunk)
    assert "Your choice looked like" not in capsys.readouterr().out

def test_wrong_answer_records_mistake():
    state = AppState()
    record_result(state, "abc", correct=False, mistake_type=MistakeType.concept_confusion.value, session_id=1)
    assert state.chunk_stats["abc"].mistakes[MistakeType.concept_confusion.value] == 1

def test_correct_answer_records_no_mistake():
    state = AppState()
    record_result(state, "abc", correct=True, mistake_type=None, session_id=1)
    assert all(v == 0 for v in state.chunk_stats["abc"].mistakes.values())

def test_save_reload_mid_session(tmp_path, monkeypatch):
    import recall_loop.store as store_mod
    monkeypatch.setattr(store_mod, "state_path", lambda name: tmp_path / "state.json")
    monkeypatch.setattr(store_mod, "collection_dir", lambda name: tmp_path)
    state = AppState()
    record_result(state, "chunk1", correct=False, mistake_type=MistakeType.reversed.value, session_id=1)
    store_mod.save_state("test", state)
    loaded = store_mod.load_state("test")
    assert loaded.chunk_stats["chunk1"].asked == 1
    assert loaded.chunk_stats["chunk1"].mistakes[MistakeType.reversed.value] == 1

def test_route_answer_skip(tmp_path, monkeypatch):
    import recall_loop.store as store_mod
    monkeypatch.setattr(store_mod, "state_path", lambda name: tmp_path / "state.json")
    monkeypatch.setattr(store_mod, "chunks_path", lambda name: tmp_path / "chunks.json")
    state: SessionState = {
        "collection": "test",
        "session_id": 1,
        "questions_total": 5,
        "questions_asked": 0,
        "questions_correct": 0,
        "current_chunk_id": "chunk1",
        "current_mcq": make_valid_mcq().model_dump(),
        "session_recent": [],
        "skipped_chunks": [],
        "done": False,
        "answer_revealed": False,
        "chosen_index": None,
        "_answer_input": "skip",
        "_post_input": None,
        "_retry_answer": False,
        "_stay_in_post": False,
        "settings_dict": {},
    }
    res = node_route_answer(state)
    assert res.get("current_chunk_id") is None
    assert res.get("answer_revealed") is False

def test_route_answer_invalid_prompts_retry(tmp_path, monkeypatch, capsys):
    import recall_loop.store as store_mod
    monkeypatch.setattr(store_mod, "state_path", lambda name: tmp_path / "state.json")
    monkeypatch.setattr(store_mod, "chunks_path", lambda name: tmp_path / "chunks.json")
    state: SessionState = {
        "collection": "test",
        "session_id": 1,
        "questions_total": 5,
        "questions_asked": 0,
        "questions_correct": 0,
        "current_chunk_id": "chunk1",
        "current_mcq": make_valid_mcq().model_dump(),
        "session_recent": [],
        "skipped_chunks": [],
        "done": False,
        "answer_revealed": False,
        "chosen_index": None,
        "_answer_input": "invalid_input",
        "_post_input": None,
        "_retry_answer": False,
        "_stay_in_post": False,
        "settings_dict": {},
    }
    res = node_route_answer(state)
    assert res.get("_retry_answer") is True
    out = capsys.readouterr().out
    assert "Please enter A, B, C, D" in out

def test_route_answer_flag(tmp_path, monkeypatch, capsys):
    import recall_loop.store as store_mod
    monkeypatch.setattr(store_mod, "state_path", lambda name: tmp_path / "state.json")
    monkeypatch.setattr(store_mod, "collection_dir", lambda name: tmp_path)
    monkeypatch.setattr(store_mod, "chunks_path", lambda name: tmp_path / "chunks.json")
    store_mod.save_state("test", AppState())
    state: SessionState = {
        "collection": "test",
        "session_id": 1,
        "questions_total": 5,
        "questions_asked": 0,
        "questions_correct": 0,
        "current_chunk_id": "chunk1",
        "current_mcq": make_valid_mcq().model_dump(),
        "session_recent": [],
        "skipped_chunks": [],
        "done": False,
        "answer_revealed": False,
        "chosen_index": None,
        "_answer_input": "flag",
        "_post_input": None,
        "_retry_answer": False,
        "_stay_in_post": False,
        "settings_dict": {},
    }
    res = node_route_answer(state)
    assert res.get("current_chunk_id") is None
    loaded = store_mod.load_state("test")
    assert "chunk1" in loaded.flagged

def test_route_answer_quit(tmp_path, monkeypatch):
    import recall_loop.store as store_mod
    monkeypatch.setattr(store_mod, "state_path", lambda name: tmp_path / "state.json")
    monkeypatch.setattr(store_mod, "chunks_path", lambda name: tmp_path / "chunks.json")
    state: SessionState = {
        "collection": "test",
        "session_id": 1,
        "questions_total": 5,
        "questions_asked": 0,
        "questions_correct": 0,
        "current_chunk_id": "chunk1",
        "current_mcq": make_valid_mcq().model_dump(),
        "session_recent": [],
        "skipped_chunks": [],
        "done": False,
        "answer_revealed": False,
        "chosen_index": None,
        "_answer_input": "quit",
        "_post_input": None,
        "_retry_answer": False,
        "_stay_in_post": False,
        "settings_dict": {},
    }
    res = node_route_answer(state)
    assert res.get("done") is True
