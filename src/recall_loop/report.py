from typing import Optional
from recall_loop.models import AppState
from recall_loop.mistakes import MistakeType, MISTAKE_LABELS, MISTAKE_DESCRIPTIONS
from recall_loop.tracker import (
    get_topic_stats, coverage_fraction, get_mistake_counts, get_topic_mistake_counts
)

MINIMUM_NOTABLE_COUNT = 3
FOOTNOTE = (
    "Tags describe the misconception the chosen option was designed to represent. "
    "They don't prove why you chose it."
)

# Print the full progress report for a collection
def print_report(chunks: list[dict], state: AppState) -> None:
    flagged = set(state.flagged)
    total_asked = sum(s.asked for s in state.chunk_stats.values())
    total_correct = sum(s.correct for s in state.chunk_stats.values())
    accuracy = total_correct / total_asked if total_asked > 0 else 0.0

    print(f"\n{'='*50}")
    print("RECALL LOOP — PROGRESS REPORT")
    print(f"{'='*50}")
    print(f"Total asked: {total_asked}  |  Overall accuracy: {accuracy:.0%}")
    print()

    topic_stats = get_topic_stats(chunks, state, flagged)
    print("TOPICS (sorted weakest first):")
    print(f"{'Topic':<30} {'Asked':>6} {'Correct':>8} {'Weakness':>10}")
    print("-" * 58)
    for topic, data in sorted(topic_stats.items(), key=lambda kv: kv[1]["weakness"], reverse=True):
        print(f"{topic:<30} {data['asked']:>6} {data['correct']:>8} {data['weakness']:>10.2f}")
    print()

    cov = coverage_fraction(chunks, state, flagged)
    print(f"Coverage: {cov:.0%} of unflagged chunks asked at least once")
    print()

    print_mistake_patterns(chunks, state, topic_stats)

    if state.flagged:
        print("\nFLAGGED CHUNKS:")
        chunk_by_id = {c["id"]: c for c in chunks}
        for fid in state.flagged:
            c = chunk_by_id.get(fid, {})
            preview = (c.get("text", "") or "")[:80].replace("\n", " ")
            src = f"{c.get('source_file', '?')}:{c.get('page') or c.get('line_start', '?')}"
            print(f"  [{fid}] {src} — {preview}...")

# Print the mistake-pattern section; used in both full report and session summary
def print_mistake_patterns(
    chunks: list[dict],
    state: AppState,
    topic_stats: Optional[dict] = None,
) -> None:
    mistake_counts = get_mistake_counts(chunks, state)
    total_wrong = sum(mistake_counts.values())

    print("MISTAKE PATTERNS:")
    if total_wrong == 0:
        print("  No wrong answers recorded yet.")
    else:
        for mt in MistakeType:
            count = mistake_counts[mt.value]
            if count > 0:
                pct = count / total_wrong
                notable = " <- notable" if count >= MINIMUM_NOTABLE_COUNT else ""
                print(f"  {MISTAKE_LABELS[mt]}: {count} ({pct:.0%}){notable}")
    print(f"\n  Note: {FOOTNOTE}")

    if topic_stats:
        topic_mistakes = get_topic_mistake_counts(chunks, state)
        print("\n  Most common mistake per weak topic:")
        for topic, data in sorted(topic_stats.items(), key=lambda kv: kv[1]["weakness"], reverse=True):
            mt_counts = topic_mistakes.get(topic, {})
            total = sum(mt_counts.values())
            if total < MINIMUM_NOTABLE_COUNT:
                continue
            top_mt = max(mt_counts, key=lambda k: mt_counts[k])
            if mt_counts[top_mt] >= MINIMUM_NOTABLE_COUNT:
                print(f"    {topic}: {MISTAKE_LABELS.get(top_mt, top_mt)} ({mt_counts[top_mt]}x)")

# Print the end-of-session summary
def print_session_summary(
    chunks: list[dict],
    state: AppState,
    session_asked: int,
    session_correct: int,
    session_chunk_ids: list[str],
) -> None:
    acc = session_correct / session_asked if session_asked > 0 else 0.0
    print(f"\n{'='*50}")
    print("SESSION SUMMARY")
    print(f"{'='*50}")
    print(f"Questions asked: {session_asked}  |  Correct: {session_correct}  |  Accuracy: {acc:.0%}")
    print()

    flagged = set(state.flagged)
    topic_stats = get_topic_stats(chunks, state, flagged)
    if topic_stats:
        print("Top weak topics overall:")
        for topic, data in sorted(topic_stats.items(), key=lambda kv: kv[1]["weakness"], reverse=True)[:3]:
            print(f"  {topic}: weakness {data['weakness']:.2f}")
    print()
    print_mistake_patterns(chunks, state)
