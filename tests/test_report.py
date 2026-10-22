import sys
from io import StringIO
from recall_loop.models import AppState
from recall_loop.mistakes import MistakeType
from recall_loop.tracker import record_result
from recall_loop.report import print_report, FOOTNOTE, MINIMUM_NOTABLE_COUNT

def make_chunks():
    return [
        {"id": "a", "topic": "Python", "text": "t1", "source_file": "f.txt", "page": None, "line_start": 1},
        {"id": "b", "topic": "Python", "text": "t2", "source_file": "f.txt", "page": None, "line_start": 10},
        {"id": "c", "topic": "Algorithms", "text": "t3", "source_file": "f.txt", "page": None, "line_start": 20},
    ]

def capture(chunks, state):
    buf = StringIO()
    old, sys.stdout = sys.stdout, buf
    print_report(chunks, state)
    sys.stdout = old
    return buf.getvalue()

def test_footnote_present():
    assert FOOTNOTE[:30] in capture(make_chunks(), AppState())

def test_notable_pattern_shown():
    chunks, state = make_chunks(), AppState()
    for _ in range(MINIMUM_NOTABLE_COUNT):
        record_result(state, "a", correct=False, mistake_type=MistakeType.concept_confusion.value, session_id=1)
    out = capture(chunks, state)
    assert "concept confusion" in out
    assert "notable" in out

def test_small_pattern_not_notable():
    chunks, state = make_chunks(), AppState()
    for _ in range(2):
        record_result(state, "a", correct=False, mistake_type=MistakeType.reversed.value, session_id=1)
    out = capture(chunks, state)
    assert "<- notable" not in out

def test_other_counted_but_no_description():
    chunks, state = make_chunks(), AppState()
    for _ in range(MINIMUM_NOTABLE_COUNT):
        record_result(state, "a", correct=False, mistake_type=MistakeType.other.value, session_id=1)
    out = capture(chunks, state)
    assert "other" in out  # counted
