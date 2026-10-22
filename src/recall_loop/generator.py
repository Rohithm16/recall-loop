import random
from typing import Optional, Callable
from recall_loop.models import MCQ, SolverResult
from recall_loop.mistakes import build_taxonomy_prompt
from recall_loop.validators import validate_mcq

MAX_GENERATION_ATTEMPTS = 2

# Build the MCQ generation prompt from a chunk of text
def build_generation_prompt(chunk_text: str) -> str:
    taxonomy = build_taxonomy_prompt()
    return (
        f"You are a quiz question writer. Generate a multiple-choice question from the notes below.\n\n"
        f"Rules:\n"
        f"- Use ONLY information from the provided notes. Do not add outside knowledge.\n"
        f"- Exactly 4 options. Exactly 1 correct. 3 are wrong distractors.\n"
        f"- The question must be self-contained. Do NOT say 'according to the text/passage/chunk'.\n"
        f"- No 'all of the above' or 'none of the above' options.\n"
        f"- For the correct option, set mistake_type to null.\n"
        f"- For each wrong option, assign exactly one mistake_type from the taxonomy below.\n"
        f"  Vary the types naturally. Use 'other' only when nothing else fits. At most 1 wrong option may use 'other'.\n"
        f"- explanation: 1-2 sentences citing what in the notes supports the correct answer.\n\n"
        f"{taxonomy}\n\n"
        f"Study notes:\n{chunk_text}"
    )

# Build the solver-check prompt: no correct answer, explanation, or tags
def build_solver_prompt(chunk_text: str, question: str, option_texts: list[str]) -> str:
    opts = "\n".join(f"{i}: {t}" for i, t in enumerate(option_texts))
    return (
        f"Read the notes and pick the index (0-3) of the most correct answer.\n\n"
        f"Notes:\n{chunk_text}\n\n"
        f"Question: {question}\n\nOptions:\n{opts}"
    )

# Shuffle options together with their mistake_type tags; update correct_index accordingly
def shuffle_options(mcq: MCQ, rng: random.Random) -> MCQ:
    indexed = list(enumerate(mcq.options))
    rng.shuffle(indexed)
    old_to_new = {old: new for new, (old, _) in enumerate(indexed)}
    new_options = [opt for _, opt in indexed]
    new_correct = old_to_new[mcq.correct_index]
    return mcq.model_copy(update={"options": new_options, "correct_index": new_correct})

# Generate a validated MCQ for a chunk; returns None if all attempts fail
def generate_question(
    chunk: dict,
    recent_questions: list[dict],
    rng: random.Random,
    settings,
    llm_call: Optional[Callable] = None,
    solver_call: Optional[Callable] = None,
) -> Optional[MCQ]:
    if llm_call is None:
        from recall_loop.llm import call_structured
        llm_call = call_structured
    if solver_call is None:
        from recall_loop.llm import call_structured
        solver_call = call_structured

    for _attempt in range(MAX_GENERATION_ATTEMPTS + 1):
        try:
            mcq = llm_call(build_generation_prompt(chunk["text"]), MCQ, settings)
            errors = validate_mcq(mcq, recent_questions, chunk["id"])
            if errors:
                continue
            if settings.solver_check:
                option_texts = [opt.text for opt in mcq.options]
                solver_result = solver_call(
                    build_solver_prompt(chunk["text"], mcq.question, option_texts),
                    SolverResult,
                    settings,
                )
                if solver_result.chosen_index != mcq.correct_index:
                    continue
            return shuffle_options(mcq, rng)
        except Exception:
            continue
    return None
