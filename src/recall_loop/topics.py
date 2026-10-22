from typing import Any
from recall_loop.models import TopicLabels

BATCH_SIZE = 10  # chunks per Gemini call for topic labeling

# Normalize a topic label: strip, collapse spaces, lowercase for comparison
def normalize_label(label: str) -> str:
    import re
    return re.sub(r"\s+", " ", label.strip()).lower()

# Merge labels that are the same when normalized (keep the first casing seen)
def dedupe_labels(labels: list[str]) -> dict[str, str]:
    # Returns a mapping from normalized label to the canonical form
    canonical: dict[str, str] = {}
    for label in labels:
        norm = normalize_label(label)
        if norm not in canonical:
            canonical[norm] = label.strip()
    return canonical

# Build the prompt for batch topic labeling
def build_topic_prompt(chunks_text: list[str], existing_labels: list[str]) -> str:
    existing_str = ", ".join(existing_labels) if existing_labels else "(none yet)"
    chunks_str = "\n\n".join(
        f"Chunk {i+1}:\n{text[:500]}" for i, text in enumerate(chunks_text)
    )
    return (
        f"You are labeling study notes chunks with short topic labels (1-4 words).\n"
        f"Existing labels in use: {existing_str}\n"
        f"Reuse an existing label when it clearly fits. Only create a new label when none fits.\n"
        f"Return exactly {len(chunks_text)} labels, one per chunk, in order.\n\n"
        f"{chunks_str}"
    )

# Label a batch of chunks using Gemini structured output
def label_batch(chunks_text: list[str], existing_labels: list[str], settings) -> list[str]:
    from recall_loop.llm import call_structured
    prompt = build_topic_prompt(chunks_text, existing_labels)
    try:
        result = call_structured(prompt, TopicLabels, settings)
        # Ensure we have the right number of labels
        labels = result.labels
        if len(labels) != len(chunks_text):
            labels = (labels + ["General"] * len(chunks_text))[:len(chunks_text)]
        return labels
    except Exception:
        return ["General"] * len(chunks_text)

# Label all chunks, reusing existing labels for unchanged chunk ids
def label_chunks(chunks: list[dict], existing_by_id: dict[str, dict], settings) -> list[dict]:
    # Collect labels already assigned (for reuse)
    existing_labels = list({c.get("topic", "General") for c in existing_by_id.values()})
    canonical = dedupe_labels(existing_labels)

    result = []
    # Separate chunks that already have a label from those that need labeling
    to_label_indices = []
    for i, chunk in enumerate(chunks):
        chunk = dict(chunk)
        if chunk["id"] in existing_by_id and existing_by_id[chunk["id"]].get("topic", "General") != "General":
            chunk["topic"] = existing_by_id[chunk["id"]]["topic"]
        else:
            to_label_indices.append(i)
        result.append(chunk)

    # Label new chunks in batches
    for batch_start in range(0, len(to_label_indices), BATCH_SIZE):
        batch_indices = to_label_indices[batch_start:batch_start + BATCH_SIZE]
        batch_texts = [result[i]["text"] for i in batch_indices]
        current_labels = list({c.get("topic", "General") for c in result if c.get("topic", "General") != "General"})
        raw_labels = label_batch(batch_texts, current_labels, settings)
        for idx, label in zip(batch_indices, raw_labels):
            # Normalize and deduplicate label
            norm = normalize_label(label)
            canonical_label = canonical.get(norm, label.strip())
            if norm not in canonical:
                canonical[norm] = label.strip()
            result[idx]["topic"] = canonical_label

    return result
