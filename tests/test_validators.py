import pytest
from recall_loop.models import MCQ, Option
from recall_loop.mistakes import MistakeType
from recall_loop.validators import validate_mcq, word_jaccard

def make_valid_mcq() -> MCQ:
    return MCQ(
        question="What is 2 + 2?",
        options=[
            Option(text="4", mistake_type=None),
            Option(text="3", mistake_type=MistakeType.wrong_detail),
            Option(text="5", mistake_type=MistakeType.wrong_detail),
            Option(text="6", mistake_type=MistakeType.concept_confusion),
        ],
        correct_index=0,
        explanation="2+2=4",
    )

def test_valid_mcq_passes():
    assert validate_mcq(make_valid_mcq(), [], "chunk1") == []

def test_wrong_option_count():
    mcq = make_valid_mcq()
    mcq.options = mcq.options[:3]
    errors = validate_mcq(mcq, [], "chunk1")
    assert any("4 options" in e for e in errors)

def test_duplicate_option_texts():
    mcq = make_valid_mcq()
    mcq.options[1] = mcq.options[1].model_copy(update={"text": "4"})
    errors = validate_mcq(mcq, [], "chunk1")
    assert any("Duplicate" in e for e in errors)

def test_bad_correct_index():
    mcq = make_valid_mcq()
    mcq = mcq.model_copy(update={"correct_index": 5})
    errors = validate_mcq(mcq, [], "chunk1")
    assert any("out of range" in e for e in errors)

def test_all_of_the_above_rejected():
    mcq = make_valid_mcq()
    mcq.options[1] = mcq.options[1].model_copy(update={"text": "All of the above"})
    errors = validate_mcq(mcq, [], "chunk1")
    assert any("meta-option" in e for e in errors)

def test_empty_question():
    mcq = make_valid_mcq()
    mcq = mcq.model_copy(update={"question": ""})
    errors = validate_mcq(mcq, [], "chunk1")
    assert any("empty" in e for e in errors)

def test_duplicate_question_exact():
    mcq = make_valid_mcq()
    recent = [{"chunk_id": "chunk1", "text": mcq.question}]
    errors = validate_mcq(mcq, recent, "chunk1")
    assert any("exact" in e for e in errors)

def test_near_duplicate_question():
    mcq = make_valid_mcq()
    # High Jaccard similarity
    recent = [{"chunk_id": "chunk1", "text": "What is 2 + 2 ?"}]
    errors = validate_mcq(mcq, recent, "chunk1")
    assert any("duplicate" in e.lower() for e in errors)

def test_correct_option_must_have_no_tag():
    mcq = make_valid_mcq()
    mcq.options[0] = mcq.options[0].model_copy(update={"mistake_type": MistakeType.wrong_detail})
    errors = validate_mcq(mcq, [], "chunk1")
    assert any("Correct option" in e for e in errors)

def test_wrong_option_needs_tag():
    mcq = make_valid_mcq()
    mcq.options[1] = mcq.options[1].model_copy(update={"mistake_type": None})
    errors = validate_mcq(mcq, [], "chunk1")
    assert any("no mistake_type" in e for e in errors)

def test_too_many_other_tags():
    mcq = make_valid_mcq()
    mcq.options[1] = mcq.options[1].model_copy(update={"mistake_type": MistakeType.other})
    mcq.options[2] = mcq.options[2].model_copy(update={"mistake_type": MistakeType.other})
    errors = validate_mcq(mcq, [], "chunk1")
    assert any("other" in e for e in errors)

def test_word_jaccard_identical():
    assert word_jaccard("the cat sat", "the cat sat") == 1.0

def test_word_jaccard_different():
    assert word_jaccard("the cat sat", "a dog ran") < 0.3
