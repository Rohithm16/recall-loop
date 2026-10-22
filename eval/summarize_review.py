#!/usr/bin/env python
"""Summarize completed question_review.csv filled in by user."""
import csv
import sys
from pathlib import Path

def yn_rate(rows, col):
    vals = [r[col].strip().lower() for r in rows if r.get(col, "").strip()]
    if not vals:
        return 0.0, 0
    yes = sum(1 for v in vals if v == "y")
    return yes / len(vals), len(vals)

def main():
    path = Path("eval/out/question_review.csv")
    if not path.exists():
        print(f"Error: {path} not found. Run review_questions.py first and fill in the judgments.")
        sys.exit(1)

    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    filled = [r for r in rows if r.get("answerable_from_chunk", "").strip()]
    if not filled:
        print("Error: 'answerable_from_chunk' is empty. Please open eval/out/question_review.csv and fill in your judgments (y/n).")
        sys.exit(1)

    ans_rate, ans_n = yn_rate(rows, "answerable_from_chunk")
    oc_rate, oc_n = yn_rate(rows, "exactly_one_correct")
    t1r, t1n = yn_rate(rows, "tag_reasonable_wrong1")
    t2r, t2n = yn_rate(rows, "tag_reasonable_wrong2")
    t3r, t3n = yn_rate(rows, "tag_reasonable_wrong3")
    total_tags_n = t1n + t2n + t3n
    total_tags_yes = int(t1r * t1n) + int(t2r * t2n) + int(t3r * t3n)
    tag_rate = total_tags_yes / total_tags_n if total_tags_n else 0.0

    print("=" * 50)
    print("QUESTION REVIEW SUMMARY")
    print("=" * 50)
    print(f"Questions reviewed: {len(rows)}")
    print(f"Answerable from chunk: {ans_rate:.0%} ({ans_n} answered)")
    print(f"Exactly one correct: {oc_rate:.0%} ({oc_n} answered)")
    print(f"Tags reasonable (overall): {tag_rate:.0%} ({total_tags_n} tags reviewed)")

if __name__ == "__main__":
    main()
