import re
from recall_loop.models import MCQ
from recall_loop.mistakes import MistakeType

MAX_QUESTION_LEN = 500
MAX_OPTION_LEN = 300
JACCARD_THRESHOLD = 0.8

# Compute word-set Jaccard similarity between two strings
def word_jaccard(a: str, b: str) -> float:
    words_a = set(re.findall(r"\w+", a.lower()))
    words_b = set(re.findall(r"\w+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)

# Normalize question text for duplicate comparison
def normalize_q(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

# Return True if an option text looks like "all of the above" or "none of the above"
def is_meta_option(text: str) -> bool:
    t = text.strip().lower()
    return "all of the above" in t or "none of the above" in t

# Validate a single MCQ; return a list of error strings (empty list = valid)
def validate_mcq(
    mcq: MCQ,
    recent_questions: list[dict],   # list of {chunk_id, text}
    chunk_id: str,
) -> list[str]:
    errors = []

    # Question non-empty and within length
    if not mcq.question or not mcq.question.strip():
        errors.append("Question is empty")
    elif len(mcq.question) > MAX_QUESTION_LEN:
        errors.append(f"Question too long ({len(mcq.question)} > {MAX_QUESTION_LEN})")

    # Exactly 4 options
    if len(mcq.options) != 4:
        errors.append(f"Expected 4 options, got {len(mcq.options)}")
        return errors

    # All options non-empty and within length
    texts = []
    for i, opt in enumerate(mcq.options):
        if not opt.text or not opt.text.strip():
            errors.append(f"Option {i} is empty")
        elif len(opt.text) > MAX_OPTION_LEN:
            errors.append(f"Option {i} too long")
        else:
            texts.append(opt.text)

    # All option texts distinct (case-insensitive)
    lower_texts = [t.lower().strip() for t in texts]
    if len(set(lower_texts)) < len(lower_texts):
        errors.append("Duplicate option texts")

    # No meta-options
    for i, opt in enumerate(mcq.options):
        if is_meta_option(opt.text):
            errors.append(f"Option {i} is a meta-option ('all/none of the above')")

    # correct_index in range
    if not (0 <= mcq.correct_index < 4):
        errors.append(f"correct_index {mcq.correct_index} out of range")
        return errors

    # Correct option must have no mistake_type
    correct_opt = mcq.options[mcq.correct_index]
    if correct_opt.mistake_type is not None:
        errors.append("Correct option must have mistake_type=None")

    # Each wrong option must have a valid mistake_type; at most 1 may be 'other'
    other_count = 0
    for i, opt in enumerate(mcq.options):
        if i == mcq.correct_index:
            continue
        if opt.mistake_type is None:
            errors.append(f"Wrong option {i} has no mistake_type")
        elif opt.mistake_type == MistakeType.other:
            other_count += 1
    if other_count > 1:
        errors.append(f"{other_count} wrong options tagged 'other' (max 1)")

    # Duplicate check against recent questions for the same chunk
    norm_q = normalize_q(mcq.question)
    for recent in recent_questions:
        if recent["chunk_id"] != chunk_id:
            continue
        recent_norm = normalize_q(recent["text"])
        if norm_q == recent_norm:
            errors.append("Duplicate question (exact match)")
            break
        if word_jaccard(mcq.question, recent["text"]) > JACCARD_THRESHOLD:
            errors.append("Duplicate question (near-duplicate)")
            break

    return errors
