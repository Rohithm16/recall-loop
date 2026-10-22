import pytest
from recall_loop.models import AppState, ChunkStats
from recall_loop.mistakes import MistakeType
from recall_loop.tracker import (
    record_result, chunk_weakness, topic_weakness_score, get_topic_stats,
    coverage_fraction, get_mistake_counts, get_topic_mistake_counts,
)

def make_chunks():
    return [
        {"id": "aaa", "topic": "Python", "text": "t1"},
        {"id": "bbb", "topic": "Python", "text": "t2"},
        {"id": "ccc", "topic": "Algorithms", "text": "t3"},
    ]

def test_weakness_formula():
    stats = ChunkStats(asked=4, correct=2)
    assert abs(chunk_weakness(stats) - 0.5) < 0.001

def test_weakness_unseen():
    stats = ChunkStats(asked=0, correct=0)
    assert abs(chunk_weakness(stats) - 0.5) < 0.001

def test_record_result_correct():
    state = AppState()
    record_result(state, "aaa", correct=True, mistake_type=None, session_id=1)
    assert state.chunk_stats["aaa"].asked == 1
    assert state.chunk_stats["aaa"].correct == 1
    assert all(v == 0 for v in state.chunk_stats["aaa"].mistakes.values())

def test_record_result_wrong_increments_mistake():
    state = AppState()
    record_result(state, "aaa", correct=False, mistake_type=MistakeType.reversed.value, session_id=1)
    assert state.chunk_stats["aaa"].mistakes[MistakeType.reversed.value] == 1
    assert state.chunk_stats["aaa"].correct == 0

def test_coverage_fraction():
    chunks = make_chunks()
    state = AppState()
    record_result(state, "aaa", correct=True, mistake_type=None, session_id=1)
    assert abs(coverage_fraction(chunks, state) - 1/3) < 0.001

def test_coverage_excludes_flagged():
    chunks = make_chunks()
    state = AppState(flagged=["aaa"])
    record_result(state, "bbb", correct=True, mistake_type=None, session_id=1)
    cov = coverage_fraction(chunks, state)
    assert abs(cov - 0.5) < 0.001

def test_topic_stats_derived():
    chunks = make_chunks()
    state = AppState()
    record_result(state, "aaa", correct=True, mistake_type=None, session_id=1)
    record_result(state, "bbb", correct=False, mistake_type=MistakeType.wrong_detail.value, session_id=1)
    stats = get_topic_stats(chunks, state)
    assert stats["Python"]["asked"] == 2
    assert stats["Python"]["correct"] == 1

def test_mistake_counts_aggregate():
    chunks = make_chunks()
    state = AppState()
    record_result(state, "aaa", correct=False, mistake_type=MistakeType.reversed.value, session_id=1)
    record_result(state, "ccc", correct=False, mistake_type=MistakeType.reversed.value, session_id=1)
    counts = get_mistake_counts(chunks, state)
    assert counts[MistakeType.reversed.value] == 2

def test_topic_mistake_counts():
    chunks = make_chunks()
    state = AppState()
    record_result(state, "aaa", correct=False, mistake_type=MistakeType.concept_confusion.value, session_id=1)
    record_result(state, "bbb", correct=False, mistake_type=MistakeType.concept_confusion.value, session_id=1)
    topic_counts = get_topic_mistake_counts(chunks, state)
    assert topic_counts["Python"][MistakeType.concept_confusion.value] == 2
