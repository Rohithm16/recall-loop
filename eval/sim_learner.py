#!/usr/bin/env python
"""Simulate a learner with a weak topic and measure targeting behavior. No API calls."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

WEAK_TOPIC = "Alpha"
WEAK_ERROR_RATE = 0.8
OTHER_ERROR_RATE = 0.2
TOTAL_QUESTIONS = 60
BLOCK_SIZE = 10

def make_chunks(n_topics=3, chunks_per_topic=5):
    chunks = []
    topic_names = ["Alpha", "Beta", "Gamma"][:n_topics]
    for t in topic_names:
        for j in range(chunks_per_topic):
            chunks.append({"id": f"{t}_{j}", "topic": t, "text": f"Content about {t} item {j}"})
    return chunks

def simulate(chunks, seed=42):
    from recall_loop.models import AppState
    from recall_loop.mistakes import MistakeType
    from recall_loop.tracker import record_result
    from recall_loop.selector import select_next_chunk

    state = AppState()
    rng = random.Random(seed)
    session_recent = []
    skipped = set()

    blocks = []
    block_counts = {t: 0 for t in {c["topic"] for c in chunks}}

    for q_num in range(TOTAL_QUESTIONS):
        if q_num > 0 and q_num % BLOCK_SIZE == 0:
            blocks.append(dict(block_counts))
            block_counts = {t: 0 for t in block_counts}

        cid = select_next_chunk(chunks, state, session_recent, skipped, rng)
        if cid is None:
            break

        chunk = next(c for c in chunks if c["id"] == cid)
        topic = chunk["topic"]
        block_counts[topic] = block_counts.get(topic, 0) + 1
        session_recent.append(cid)

        # Simulate learner answer
        error_rate = WEAK_ERROR_RATE if topic == WEAK_TOPIC else OTHER_ERROR_RATE
        is_wrong = rng.random() < error_rate
        record_result(
            state, cid,
            correct=not is_wrong,
            mistake_type=MistakeType.wrong_detail.value if is_wrong else None,
            session_id=1,
        )

    if block_counts:
        blocks.append(dict(block_counts))

    return blocks, state

def uniform_baseline(chunks, seed=42):
    rng = random.Random(seed)
    blocks = []
    block_counts = {t: 0 for t in {c["topic"] for c in chunks}}
    for q_num in range(TOTAL_QUESTIONS):
        if q_num > 0 and q_num % BLOCK_SIZE == 0:
            blocks.append(dict(block_counts))
            block_counts = {t: 0 for t in block_counts}
        chunk = rng.choice(chunks)
        block_counts[chunk["topic"]] = block_counts.get(chunk["topic"], 0) + 1
    if block_counts:
        blocks.append(dict(block_counts))
    return blocks

def main():
    chunks = make_chunks()
    our_blocks, state = simulate(chunks)
    baseline_blocks = uniform_baseline(chunks)

    from recall_loop.tracker import coverage_fraction
    cov = coverage_fraction(chunks, state)

    print("=" * 60)
    print("LEARNER SIMULATION — Targeting Behavior")
    print(f"Weak topic: {WEAK_TOPIC} (error rate {WEAK_ERROR_RATE:.0%}), others {OTHER_ERROR_RATE:.0%}")
    print("=" * 60)
    print(f"\n{'Block (10 qs)':<15} {'Alpha (ours)':>14} {'Alpha (uniform)':>16}")
    print("-" * 47)
    for i, (our, base) in enumerate(zip(our_blocks, baseline_blocks)):
        alpha_ours = our.get(WEAK_TOPIC, 0)
        alpha_base = base.get(WEAK_TOPIC, 0)
        print(f"Block {i+1:<8} {alpha_ours:>14} {alpha_base:>16}")
    print(f"\nCoverage reached: {cov:.0%}")

if __name__ == "__main__":
    main()
