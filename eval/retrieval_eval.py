#!/usr/bin/env python
"""Measure hit@1, hit@3, and hit@5 for the why retrieval system."""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    labels_path = Path("eval/labels/why_questions.csv")
    if not labels_path.exists():
        print(f"Error: {labels_path} not found. Run make_label_template.py and fill in questions first.")
        sys.exit(1)

    rows = list(csv.DictReader(open(labels_path, encoding="utf-8")))
    labeled = [
        r for r in rows
        if r.get("your_question", "").strip() and r.get("expected_chunk_id", "").strip()
    ]
    if len(labeled) < 15:
        print(f"Error: Need at least 15 labeled questions in {labels_path}. Found {len(labeled)}.")
        print("Please fill in 'your_question' and 'expected_chunk_id' for at least 15 rows.")
        sys.exit(1)

    from recall_loop.config import get_settings
    from recall_loop.index import load_index
    from recall_loop.store import load_chunks

    collection = os.environ.get("COLLECTION", "default")
    settings = get_settings()
    if not settings.api_key:
        print("Error: GEMINI_API_KEY is not set.")
        sys.exit(1)

    chunks = load_chunks(collection)
    if not chunks:
        print(f"Error: Collection '{collection}' is empty.")
        sys.exit(1)

    index = load_index(collection, settings)
    retriever = index.as_retriever(similarity_top_k=5)

    hit1 = hit3 = hit5 = total = 0
    for row in labeled:
        query = row["your_question"]
        expected = row["expected_chunk_id"]
        try:
            nodes = retriever.retrieve(query)
            retrieved_ids = [n.metadata.get("chunk_id", n.node_id) for n in nodes]
            if expected in retrieved_ids[:1]:
                hit1 += 1
            if expected in retrieved_ids[:3]:
                hit3 += 1
            if expected in retrieved_ids[:5]:
                hit5 += 1
            total += 1
        except Exception as e:
            print(f"  Retrieval error for '{query}': {e}")

    if total == 0:
        print("No queries evaluated.")
        return

    print("=" * 45)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 45)
    print(f"Questions evaluated: {total}")
    print(f"hit@1: {hit1/total:.0%}")
    print(f"hit@3: {hit3/total:.0%}")
    print(f"hit@5: {hit5/total:.0%}")

if __name__ == "__main__":
    main()
