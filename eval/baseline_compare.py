#!/usr/bin/env python
"""Compare pipeline vs single-prompt baseline on coverage, repetition, and groundedness."""
import os
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SIMILARITY_THRESHOLD_COVERAGE = 0.60
SIMILARITY_THRESHOLD_REPETITION = 0.85
TARGET_QUESTIONS = 30

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x ** 2 for x in a) ** 0.5
    nb = sum(x ** 2 for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0

def parse_baseline(text: str) -> list[dict]:
    questions = []
    current = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Q:"):
            if current.get("question"):
                questions.append(current)
            current = {"question": line[2:].strip(), "options": []}
        elif line.startswith(("A)", "B)", "C)", "D)")) and current:
            current.setdefault("options", []).append(line[2:].strip())
        elif line.startswith("Answer:") and current:
            current["answer"] = line[7:].strip()
    if current.get("question"):
        questions.append(current)
    return questions

def main():
    from recall_loop.config import get_settings
    from recall_loop.store import load_chunks
    from recall_loop.generator import generate_question, build_solver_prompt
    from recall_loop.llm import call_structured, call_text
    from recall_loop.models import SolverResult

    settings = get_settings()
    if not settings.api_key:
        print("Error: GEMINI_API_KEY is not set.")
        sys.exit(1)

    collection = os.environ.get("COLLECTION", "default")
    chunks = load_chunks(collection)
    if not chunks:
        print(f"Error: Collection '{collection}' is empty.")
        sys.exit(1)

    print(f"Comparing pipeline vs baseline on {len(chunks)} chunks, {TARGET_QUESTIONS} questions each.")
    print(f"Coverage threshold: {SIMILARITY_THRESHOLD_COVERAGE} | Repetition threshold: {SIMILARITY_THRESHOLD_REPETITION}")

    # Generate pipeline questions
    rng = random.Random(42)
    settings.solver_check = False
    chunk_list = list(chunks)
    rng.shuffle(chunk_list)
    pipeline_qs = []
    i = 0
    while len(pipeline_qs) < TARGET_QUESTIONS and i < len(chunk_list) * 3:
        chunk = chunk_list[i % len(chunk_list)]
        i += 1
        mcq = generate_question(chunk, [], rng, settings)
        if mcq:
            pipeline_qs.append((chunk, mcq))

    # Generate baseline questions (single prompt)
    notes_text = "\n\n".join(c["text"] for c in chunks[:15])[:8000]
    prompt = (
        f"Generate {TARGET_QUESTIONS} multiple-choice questions from these study notes.\n"
        f"Each question: exactly 4 options (A-D), exactly 1 correct.\n"
        f"Format:\nQ: <question>\nA) <opt>\nB) <opt>\nC) <opt>\nD) <opt>\nAnswer: <letter>\n\n"
        f"Notes:\n{notes_text}"
    )
    baseline_raw = call_text(prompt, settings)
    baseline_qs = parse_baseline(baseline_raw)

    print(f"Pipeline questions generated: {len(pipeline_qs)}")
    print(f"Baseline questions parsed: {len(baseline_qs)}")

    from llama_index.embeddings.gemini import GeminiEmbedding
    embed = GeminiEmbedding(model_name=settings.embed_model, api_key=settings.api_key)
    chunk_embeddings = [embed.get_text_embedding(c["text"]) for c in chunks]

    def measure_coverage(question_texts):
        covered = set()
        for qt in question_texts:
            qe = embed.get_text_embedding(qt)
            best_idx = max(range(len(chunk_embeddings)), key=lambda j: cosine_similarity(qe, chunk_embeddings[j]))
            if cosine_similarity(qe, chunk_embeddings[best_idx]) >= SIMILARITY_THRESHOLD_COVERAGE:
                covered.add(best_idx)
        return len(covered) / len(chunks) if chunks else 0.0

    def measure_repetition(question_texts):
        embs = [embed.get_text_embedding(qt) for qt in question_texts]
        pairs = 0
        for i_idx in range(len(embs)):
            for j_idx in range(i_idx + 1, len(embs)):
                if cosine_similarity(embs[i_idx], embs[j_idx]) >= SIMILARITY_THRESHOLD_REPETITION:
                    pairs += 1
        return pairs

    p_texts = [q.question for _, q in pipeline_qs]
    b_texts = [q["question"] for q in baseline_qs]

    p_cov = measure_coverage(p_texts)
    b_cov = measure_coverage(b_texts)
    p_rep = measure_repetition(p_texts)
    b_rep = measure_repetition(b_texts)

    p_agree = p_total = 0
    for chunk, mcq in pipeline_qs[:10]:
        try:
            sp = build_solver_prompt(chunk["text"], mcq.question, [o.text for o in mcq.options])
            sr = call_structured(sp, SolverResult, settings)
            p_total += 1
            if sr.chosen_index == mcq.correct_index:
                p_agree += 1
        except Exception:
            pass

    print()
    print("=" * 60)
    print("BASELINE COMPARISON TABLE")
    print(f"Chunk count: {len(chunks)}")
    print(f"{'Metric':<35} {'Pipeline':>10} {'Baseline':>10}")
    print("-" * 57)
    print(f"{'Coverage (% chunks hit)':<35} {p_cov:>10.0%} {b_cov:>10.0%}")
    print(f"{'Repetition (# pairs > threshold)':<35} {p_rep:>10} {b_rep:>10}")
    solver_str = f"{p_agree}/{p_total}" if p_total else "N/A"
    print(f"{'Solver agreement (sample)':<35} {solver_str:>10}  N/A (no gt)")

if __name__ == "__main__":
    main()
