#!/usr/bin/env python
"""Write eval/labels/why_questions.csv template with chunk previews for manual labeling."""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    from recall_loop.store import load_chunks
    collection = os.environ.get("COLLECTION", "default")
    chunks = load_chunks(collection)
    if not chunks:
        print(f"Error: Collection '{collection}' is empty. Run 'ingest' first.")
        sys.exit(1)

    labels_dir = Path("eval/labels")
    labels_dir.mkdir(parents=True, exist_ok=True)
    out_path = labels_dir / "why_questions.csv"

    sample_chunks = chunks[:25]
    rows = []
    for c in sample_chunks:
        rows.append({
            "chunk_id": c["id"],
            "chunk_preview": c["text"][:200].replace("\n", " "),
            "source": f"{c.get('source_file', '?')}:{c.get('page') or c.get('line_start', '?')}",
            "your_question": "",
            "expected_chunk_id": c["id"],
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Template written to {out_path}")
    print("Please fill in 'your_question' for at least 15 rows, then run retrieval_eval.py.")

if __name__ == "__main__":
    main()
