import pytest
import os
import random
from pathlib import Path

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — skipping live tests",
)

@pytest.mark.live
def test_ingest_and_generate_and_why(tmp_path):
    """Live smoke test: ingest sample notes, generate 3 questions, run a why query."""
    import os
    os.environ["RECALL_LOOP_DATA_DIR"] = str(tmp_path)
    import importlib
    import recall_loop.config as cfg_mod
    importlib.reload(cfg_mod)
    import recall_loop.store as store_mod
    importlib.reload(store_mod)
    settings = cfg_mod.get_settings()

    from recall_loop import ingest, topics
    import recall_loop.index as index_mod
    from recall_loop.models import AppState, Chunk
    from recall_loop.mistakes import MistakeType

    notes_path = Path("data/sample_notes/python_basics.txt")
    assert notes_path.exists(), "Sample notes file not found"

    collection = "smoke_live"
    new_chunks, errors = ingest.load_files([notes_path])
    assert not errors, f"Ingestion errors: {errors}"
    assert len(new_chunks) > 0, "No chunks created"

    merged = ingest.merge_with_existing(new_chunks, [])
    merged = topics.label_chunks(merged, {}, settings)
    store_mod.save_chunks(collection, merged)
    store_mod.save_state(collection, AppState())
    index_mod.build_index([Chunk.model_validate(c) for c in merged], collection, settings)

    from recall_loop.generator import generate_question
    chunks = store_mod.load_chunks(collection)
    rng = random.Random(42)
    generated = 0
    for chunk in chunks[:6]:
        mcq = generate_question(chunk, [], rng, settings)
        if mcq is None:
            continue
        print(f"\n[Generated Question {generated+1}]: {mcq.question}")
        for i, opt in enumerate(mcq.options):
            tag = f" [{opt.mistake_type.value}]" if opt.mistake_type else " [CORRECT]"
            print(f"  {i}: {opt.text}{tag}")
        # Verify wrong options all have valid mistake tags
        wrong = [o for i, o in enumerate(mcq.options) if i != mcq.correct_index]
        for opt in wrong:
            assert opt.mistake_type is not None, "Wrong option missing mistake tag"
            assert opt.mistake_type in MistakeType, f"Unknown mistake type {opt.mistake_type}"
        generated += 1
        if generated >= 3:
            break

    assert generated >= 1, "Failed to generate at least 1 question"

    from recall_loop.rag import answer_why
    answer_why("What does dynamically typed mean?", collection, chunks[0]["id"], set(), settings)
    print("why query completed.")
