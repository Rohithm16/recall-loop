import random
import pytest
from recall_loop.models import MCQ, Option, SolverResult
from recall_loop.mistakes import MistakeType
from recall_loop.generator import generate_question, shuffle_options, build_solver_prompt
from recall_loop.config import get_settings

def make_valid_mcq() -> MCQ:
    return MCQ(
        question="What is the correct answer?",
        options=[
            Option(text="Correct answer", mistake_type=None),
            Option(text="Wrong A", mistake_type=MistakeType.wrong_detail),
            Option(text="Wrong B", mistake_type=MistakeType.reversed),
            Option(text="Wrong C", mistake_type=MistakeType.incomplete),
        ],
        correct_index=0,
        explanation="Because the notes say so.",
    )

def test_valid_mcq_passes_generation():
    settings = get_settings()
    settings.solver_check = False
    chunk = {"id": "abc123def456", "text": "Some content.", "topic": "T"}
    result = generate_question(chunk, [], random.Random(42), settings,
                               llm_call=lambda p, m, s: make_valid_mcq())
    assert result is not None
    assert len(result.options) == 4

def test_invalid_mcq_retries_then_skips():
    settings = get_settings()
    settings.solver_check = False
    chunk = {"id": "abc123def456", "text": "Some content.", "topic": "T"}
    # Bad MCQ: only 3 options
    bad = MCQ(
        question="Q?",
        options=[Option(text="A", mistake_type=None),
                 Option(text="B", mistake_type=MistakeType.wrong_detail),
                 Option(text="C", mistake_type=MistakeType.reversed)],
        correct_index=0, explanation="E",
    )
    call_count = [0]
    def fake(p, m, s):
        call_count[0] += 1
        return bad
    result = generate_question(chunk, [], random.Random(42), settings, llm_call=fake)
    assert result is None
    assert call_count[0] >= 2

def test_solver_check_mismatch_fails():
    settings = get_settings()
    settings.solver_check = True
    chunk = {"id": "abc123def456", "text": "Content.", "topic": "T"}
    result = generate_question(
        chunk, [], random.Random(42), settings,
        llm_call=lambda p, m, s: make_valid_mcq(),
        solver_call=lambda p, m, s: SolverResult(chosen_index=1),  # wrong
    )
    assert result is None

def test_solver_off_skips_solver():
    settings = get_settings()
    settings.solver_check = False
    chunk = {"id": "abc123def456", "text": "Content.", "topic": "T"}
    called = [False]
    def fake_solver(p, m, s):
        called[0] = True
        return SolverResult(chosen_index=0)
    generate_question(chunk, [], random.Random(42), settings,
                      llm_call=lambda p, m, s: make_valid_mcq(),
                      solver_call=fake_solver)
    assert not called[0]

def test_solver_prompt_has_no_tags_or_explanation():
    prompt = build_solver_prompt("Notes.", "What is X?", ["A", "B", "C", "D"])
    assert "mistake_type" not in prompt
    assert "explanation" not in prompt
    assert "What is X?" in prompt

def test_shuffle_preserves_tags():
    mcq = make_valid_mcq()
    shuffled = shuffle_options(mcq, random.Random(42))
    assert shuffled.options[shuffled.correct_index].mistake_type is None
    wrong = [o for i, o in enumerate(shuffled.options) if i != shuffled.correct_index]
    assert all(o.mistake_type is not None for o in wrong)

def test_shuffle_correct_index_follows():
    mcq = make_valid_mcq()
    original_correct_text = mcq.options[0].text
    shuffled = shuffle_options(mcq, random.Random(42))
    assert shuffled.options[shuffled.correct_index].text == original_correct_text
