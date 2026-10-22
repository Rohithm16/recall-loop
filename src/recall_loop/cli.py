import argparse
import sys

def main():
    parser = argparse.ArgumentParser(prog="recall_loop", description="Recall Loop quiz coach")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest notes files")
    p_ingest.add_argument("paths", nargs="+", help="Files to ingest")
    p_ingest.add_argument("--collection", default="default", help="Collection name")

    p_quiz = sub.add_parser("quiz", help="Start a quiz session")
    p_quiz.add_argument("--collection", default="default", help="Collection name")
    p_quiz.add_argument("--questions", type=int, default=None, help="Number of questions")
    p_quiz.add_argument("--seed", type=int, default=None, help="Random seed")

    p_report = sub.add_parser("report", help="Show progress report")
    p_report.add_argument("--collection", default="default", help="Collection name")

    p_reset = sub.add_parser("reset", help="Reset progress (keeps chunks)")
    p_reset.add_argument("--collection", default="default", help="Collection name")

    args = parser.parse_args()

    handlers = {
        "ingest": cmd_ingest,
        "quiz": cmd_quiz,
        "report": cmd_report,
        "reset": cmd_reset,
    }
    handlers[args.command](args)

def _check_api_key():
    from recall_loop.config import get_settings
    settings = get_settings()
    if not settings.api_key:
        print("Error: GEMINI_API_KEY is not set. Please set it in your .env file.")
        sys.exit(1)
    return settings

def _check_collection_not_empty(collection: str):
    from recall_loop.store import load_chunks
    chunks = load_chunks(collection)
    if not chunks:
        print(f"Error: Collection '{collection}' is empty. Run 'ingest' first.")
        sys.exit(1)
    return chunks

def cmd_ingest(args):
    from pathlib import Path
    from recall_loop import ingest, topics, store, index
    settings = _check_api_key()
    paths = [Path(p) for p in args.paths]
    collection = args.collection
    new_chunks, errors = ingest.load_files(paths)
    for err in errors:
        print(f"Skipped: {err}")
    if not new_chunks:
        print("No chunks were created. Check your files.")
        sys.exit(1)
    existing_chunks = store.load_chunks(collection)
    merged = ingest.merge_with_existing(new_chunks, existing_chunks)
    existing_by_id = {c["id"]: c for c in existing_chunks}
    merged = topics.label_chunks(merged, existing_by_id, settings)
    store.save_chunks(collection, merged)
    state = store.load_state(collection)
    current_ids = {c["id"] for c in merged}
    state.chunk_stats = {k: v for k, v in state.chunk_stats.items() if k in current_ids}
    state.flagged = [f for f in state.flagged if f in current_ids]
    store.save_state(collection, state)
    from recall_loop.models import Chunk as ChunkModel
    chunk_objects = [ChunkModel.model_validate(c) for c in merged]
    index.build_index(chunk_objects, collection, settings)
    topic_names = sorted({c["topic"] for c in merged})
    print(f"\nIngestion complete for collection '{collection}':")
    print(f"  Files processed: {len([p for p in paths if not any(p.name in e for e in errors)])}")
    print(f"  Chunks created: {len(merged)}")
    print(f"  Topics found: {', '.join(topic_names)}")
    if errors:
        print(f"  Files rejected: {len(errors)}")

def cmd_quiz(args):
    settings = _check_api_key()
    _check_collection_not_empty(args.collection)
    n = args.questions or settings.questions_per_session
    from recall_loop.graph import run_session
    run_session(args.collection, n, seed=args.seed)

def cmd_report(args):
    _check_api_key()
    chunks = _check_collection_not_empty(args.collection)
    from recall_loop.store import load_state
    from recall_loop.report import print_report
    print_report(chunks, load_state(args.collection))

def cmd_reset(args):
    confirm = input(f"Reset all progress for collection '{args.collection}'? This cannot be undone. Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Reset cancelled.")
        return
    from recall_loop.store import save_state
    from recall_loop.models import AppState
    save_state(args.collection, AppState())
    print(f"Progress reset for collection '{args.collection}'. Chunks are preserved.")
