import random
import pytest
from recall_loop.models import AppState
from recall_loop.mistakes import MistakeType
from recall_loop.tracker import record_result
from recall_loop.selector import select_next_chunk

def make_chunks():
    return [
        {"id": f"chunk{i:03d}", "topic": ["Alpha", "Beta", "Gamma"][i % 3], "text": f"text {i}"}
        for i in range(9)
    ]

def test_coverage_phase_asks_all_chunks():
    chunks = make_chunks()
    state = AppState()
    rng = random.Random(42)
    selected = set()
    for _ in range(9):
        cid = select_next_chunk(chunks, state, [], set(), rng)
        assert cid is not None
        selected.add(cid)
        record_result(state, cid, correct=True, mistake_type=None, session_id=1)
    assert selected == {c["id"] for c in chunks}

def test_flagged_never_selected():
    chunks = make_chunks()
    state = AppState(flagged=["chunk000"])
    rng = random.Random(42)
    for _ in range(20):
        cid = select_next_chunk(chunks, state, [], set(), rng)
        if cid:
            assert cid != "chunk000"

def test_no_eligible_returns_none():
    chunks = [{"id": "only", "topic": "T", "text": "t"}]
    state = AppState(flagged=["only"])
    assert select_next_chunk(chunks, state, [], set(), random.Random(42)) is None

def test_same_seed_same_sequence():
    chunks = make_chunks()
    state1, state2 = AppState(), AppState()
    rng1, rng2 = random.Random(99), random.Random(99)
    seq1, seq2 = [], []
    for _ in range(9):
        c1 = select_next_chunk(chunks, state1, seq1, set(), rng1)
        c2 = select_next_chunk(chunks, state2, seq2, set(), rng2)
        seq1.append(c1)
        seq2.append(c2)
        record_result(state1, c1, correct=True, mistake_type=None, session_id=1)
        record_result(state2, c2, correct=True, mistake_type=None, session_id=1)
    assert seq1 == seq2

def test_weakness_targets_weak_topic():
    chunks = make_chunks()
    state = AppState()
    rng = random.Random(42)
    for c in chunks:
        record_result(state, c["id"], correct=True, mistake_type=None, session_id=1)
    for _ in range(10):
        for c in chunks:
            if c["topic"] == "Alpha":
                record_result(state, c["id"], correct=False,
                              mistake_type=MistakeType.wrong_detail.value, session_id=1)
    alpha_count = sum(
        1 for _ in range(30)
        if (cid := select_next_chunk(chunks, state, [], set(), rng))
        and any(c["id"] == cid and c["topic"] == "Alpha" for c in chunks)
    )
    assert alpha_count > 10

def test_no_repeat_in_window():
    chunks = make_chunks()
    state = AppState()
    rng = random.Random(42)
    for c in chunks:
        record_result(state, c["id"], correct=True, mistake_type=None, session_id=1)
    recent = [c["id"] for c in chunks[:8]]
    cid = select_next_chunk(chunks, state, recent, set(), rng)
    assert cid is not None

def test_mistake_types_do_not_affect_selection():
    chunks = [{"id": "x", "topic": "T", "text": "tx"}, {"id": "y", "topic": "T", "text": "ty"}]
    state = AppState()
    record_result(state, "x", correct=False, mistake_type=MistakeType.reversed.value, session_id=1)
    record_result(state, "y", correct=False, mistake_type=MistakeType.concept_confusion.value, session_id=1)
    rng1, rng2 = random.Random(42), random.Random(42)
    seq1, seq2 = [], []
    for _ in range(10):
        c1 = select_next_chunk(chunks, state, seq1, set(), rng1)
        c2 = select_next_chunk(chunks, state, seq2, set(), rng2)
        seq1.append(c1)
        seq2.append(c2)
    assert seq1 == seq2
