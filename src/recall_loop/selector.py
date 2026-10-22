import random
from typing import Optional
from recall_loop.models import AppState, ChunkStats
from recall_loop.tracker import topic_weakness_score, chunk_weakness

# Select the next chunk to quiz on; returns chunk_id or None if none eligible
def select_next_chunk(
    chunks: list[dict],
    state: AppState,
    session_recent: list[str],      # chunk ids asked this session (oldest first)
    skipped_this_session: set[str], # chunk ids that failed generation this session
    rng: random.Random,
) -> Optional[str]:
    flagged = set(state.flagged)
    eligible = [c for c in chunks if c["id"] not in flagged and c["id"] not in skipped_this_session]
    if not eligible:
        return None

    # Avoid repeating the last 5 chunks in the session unless no other option
    recent_window = set(session_recent[-5:]) if len(session_recent) >= 5 else set()
    not_recent = [c for c in eligible if c["id"] not in recent_window]
    candidates = not_recent if not_recent else eligible

    # Coverage phase: any unflagged chunk with asked==0 exists
    unasked = [c for c in candidates if state.chunk_stats.get(c["id"], ChunkStats()).asked == 0]
    if unasked:
        return _pick_from_weakest_topic(unasked, chunks, state, flagged, rng)

    # Weakness phase: weighted random draw by topic weakness score
    return _pick_by_weakness(candidates, chunks, state, flagged, rng)

# Pick one chunk from the topic with the highest weakness score among candidates
def _pick_from_weakest_topic(
    candidates: list[dict],
    all_chunks: list[dict],
    state: AppState,
    flagged: set[str],
    rng: random.Random,
) -> str:
    topics = list({c["topic"] for c in candidates})
    scores = {t: topic_weakness_score(t, all_chunks, state, flagged) for t in topics}
    max_score = max(scores.values())
    best_topics = sorted(t for t, s in scores.items() if s == max_score)
    chosen_topic = rng.choice(best_topics)
    topic_candidates = [c for c in candidates if c["topic"] == chosen_topic]
    return rng.choice(topic_candidates)["id"]

# Weighted random topic draw, then pick least-recently-asked chunk in that topic
def _pick_by_weakness(
    candidates: list[dict],
    all_chunks: list[dict],
    state: AppState,
    flagged: set[str],
    rng: random.Random,
) -> Optional[str]:
    topics = list({c["topic"] for c in candidates})
    scores = [topic_weakness_score(t, all_chunks, state, flagged) for t in topics]
    total = sum(scores)
    if total == 0:
        chosen_topic = rng.choice(topics)
    else:
        r = rng.random() * total
        cumulative, chosen_topic = 0.0, topics[-1]
        for topic, score in zip(topics, scores):
            cumulative += score
            if r <= cumulative:
                chosen_topic = topic
                break
    topic_candidates = [c for c in candidates if c["topic"] == chosen_topic]
    topic_candidates.sort(
        key=lambda c: (state.chunk_stats.get(c["id"], ChunkStats()).last_asked_session or 0)
    )
    return topic_candidates[0]["id"]
