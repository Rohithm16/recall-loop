#!/usr/bin/env python
"""Generate 30 questions from notes through the pipeline and write question_review.csv.
Also prints the automatic solver-agreement rate."""
import csv
import os
import sys
import random
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    from recall_loop.config import get_settings
    from recall_loop.store import load_chunks, load_state
    from recall_loop.generator import generate_question, build_solver_prompt
    from recall_loop.models import SolverResult

    settings = get_settings()
    if not settings.api_key:
        print("Error: GEMINI_API_KEY is not set. Please set it in your .env file.")
        sys.exit(1)

    collection = os.environ.get("COLLECTION", "default")
    chunks = load_chunks(collection)
    if not chunks:
        print(f"Error: Collection '{collection}' is empty. Run 'ingest' first.")
        sys.exit(1)

    state = load_state(collection)
    recent = [r.model_dump() for r in state.recent_questions]
    rng = random.Random(42)

    rows = []
    solver_agree = 0
    solver_total = 0
    target = 30

    chunk_list = list(chunks)
    rng.shuffle(chunk_list)
    i = 0
    # Disable solver check during question generation to evaluate solver separately
    settings.solver_check = False

    while len(rows) < target and i < len(chunk_list) * 3:
        chunk = chunk_list[i % len(chunk_list)]
        i += 1
        mcq = generate_question(chunk, recent, rng, settings)
        if mcq is None:
            continue

        # Evaluate solver check independently
        try:
            from recall_loop.llm import call_structured
            sp = build_solver_prompt(chunk["text"], mcq.question, [o.text for o in mcq.options])
            sr = call_structured(sp, SolverResult, settings)
            solver_total += 1
            if sr.chosen_index == mcq.correct_index:
                solver_agree += 1
        except Exception:
            pass

        wrong = [(idx, o) for idx, o in enumerate(mcq.options) if idx != mcq.correct_index]
        row = {
            "question": mcq.question,
            "option_a": mcq.options[0].text,
            "option_b": mcq.options[1].text,
            "option_c": mcq.options[2].text,
            "option_d": mcq.options[3].text,
            "correct_option": "ABCD"[mcq.correct_index],
            "wrong1_text": wrong[0][1].text if len(wrong) > 0 else "",
            "wrong1_tag": wrong[0][1].mistake_type.value if len(wrong) > 0 and wrong[0][1].mistake_type else "",
            "wrong2_text": wrong[1][1].text if len(wrong) > 1 else "",
            "wrong2_tag": wrong[1][1].mistake_type.value if len(wrong) > 1 and wrong[1][1].mistake_type else "",
            "wrong3_text": wrong[2][1].text if len(wrong) > 2 else "",
            "wrong3_tag": wrong[2][1].mistake_type.value if len(wrong) > 2 and wrong[2][1].mistake_type else "",
            "source_chunk": chunk["text"][:300],
            "source_file": chunk.get("source_file", ""),
            "answerable_from_chunk": "",
            "exactly_one_correct": "",
            "tag_reasonable_wrong1": "",
            "tag_reasonable_wrong2": "",
            "tag_reasonable_wrong3": "",
            "notes": "",
        }
        rows.append(row)

    out = Path("eval/out")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "question_review.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} questions -> {path}")
    if solver_total > 0:
        print(f"Solver agreement rate: {solver_agree}/{solver_total} = {solver_agree/solver_total:.0%}")

if __name__ == "__main__":
    main()
