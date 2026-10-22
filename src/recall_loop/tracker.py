from typing import Optional
from recall_loop.models import AppState, ChunkStats
from recall_loop.mistakes import MistakeType

# Ensure a chunk has a stats entry in state; initialize to zeros if missing
def ensure_stats(state: AppState, chunk_id: str) -> None:
    if chunk_id not in state.chunk_stats:
        state.chunk_stats[chunk_id] = ChunkStats()

# Record the result of one answered question; increment mistake counter if wrong
def record_result(
    state: AppState,
    chunk_id: str,
    correct: bool,
    mistake_type: Optional[str],   # MistakeType.value or None
    session_id: int,
) -> None:
    ensure_stats(state, chunk_id)
    stats = state.chunk_stats[chunk_id]
    stats.asked += 1
    stats.last_asked_session = session_id
    if correct:
        stats.correct += 1
    else:
        if mistake_type and mistake_type in stats.mistakes:
            stats.mistakes[mistake_type] += 1
        else:
            stats.mistakes[MistakeType.other.value] += 1

# Weakness score for a single chunk: (wrong+1)/(asked+2). Smooths small samples.
def chunk_weakness(stats: ChunkStats) -> float:
    wrong = stats.asked - stats.correct
    return (wrong + 1) / (stats.asked + 2)

# Weakness score for a topic: sum of chunk weakness scores over its unflagged chunks.
def topic_weakness_score(
    topic: str,
    chunks: list[dict],
    state: AppState,
    flagged: Optional[set[str]] = None,
) -> float:
    flagged = set(state.flagged) if flagged is None else flagged
    topic_chunks = [c for c in chunks if c.get("topic") == topic and c["id"] not in flagged]
    if not topic_chunks:
        return 0.5
    return sum(chunk_weakness(state.chunk_stats.get(c["id"], ChunkStats())) for c in topic_chunks)

# Derive per-topic stats dict from chunk_stats: {topic: {asked, correct, weakness}}
def get_topic_stats(
    chunks: list[dict],
    state: AppState,
    flagged: Optional[set[str]] = None,
) -> dict[str, dict]:
    flagged = set(state.flagged) if flagged is None else flagged
    topics: dict[str, dict] = {}
    for c in chunks:
        if c["id"] in flagged:
            continue
        topic = c.get("topic", "General")
        stats = state.chunk_stats.get(c["id"], ChunkStats())
        if topic not in topics:
            topics[topic] = {"asked": 0, "correct": 0, "chunks": []}
        topics[topic]["asked"] += stats.asked
        topics[topic]["correct"] += stats.correct
        topics[topic]["chunks"].append(c["id"])
    for topic, data in topics.items():
        wrong = data["asked"] - data["correct"]
        data["weakness"] = (wrong + 1) / (data["asked"] + 2)
    return topics

# Coverage: fraction of unflagged chunks that have been asked at least once
def coverage_fraction(
    chunks: list[dict],
    state: AppState,
    flagged: Optional[set[str]] = None,
) -> float:
    flagged = set(state.flagged) if flagged is None else flagged
    unflagged = [c for c in chunks if c["id"] not in flagged]
    if not unflagged:
        return 1.0
    asked = sum(1 for c in unflagged if state.chunk_stats.get(c["id"], ChunkStats()).asked > 0)
    return asked / len(unflagged)

# Aggregate mistake counts across all chunks
def get_mistake_counts(chunks: list[dict], state: AppState) -> dict[str, int]:
    totals: dict[str, int] = {mt.value: 0 for mt in MistakeType}
    for c in chunks:
        for mt_val, count in state.chunk_stats.get(c["id"], ChunkStats()).mistakes.items():
            if mt_val in totals:
                totals[mt_val] += count
    return totals

# Get mistake counts per topic: {topic: {mistake_type_value: count}}
def get_topic_mistake_counts(chunks: list[dict], state: AppState) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for c in chunks:
        topic = c.get("topic", "General")
        stats = state.chunk_stats.get(c["id"], ChunkStats())
        if topic not in result:
            result[topic] = {mt.value: 0 for mt in MistakeType}
        for mt_val, count in stats.mistakes.items():
            if mt_val in result[topic]:
                result[topic][mt_val] += count
    return result
