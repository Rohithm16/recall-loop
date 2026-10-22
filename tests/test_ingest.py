import pytest
from pathlib import Path
import tempfile
import os
from recall_loop.ingest import make_chunk_id, load_files, merge_with_existing
from recall_loop.models import Chunk

def test_chunk_id_stable():
    # Same text should always produce the same id
    assert make_chunk_id("hello world") == make_chunk_id("hello world")

def test_chunk_id_changes_with_text():
    # Different text should produce a different id
    assert make_chunk_id("hello world") != make_chunk_id("hello world!")

def test_chunk_id_length():
    # Id should be 12 hex chars
    assert len(make_chunk_id("test")) == 12

def test_unsupported_file_type(tmp_path):
    # .docx files should be rejected
    f = tmp_path / "notes.docx"
    f.write_bytes(b"fake content")
    chunks, errors = load_files([f])
    assert len(errors) == 1
    assert "Unsupported" in errors[0]

def test_text_file_produces_chunks(tmp_path):
    # A normal text file should produce at least one chunk
    f = tmp_path / "notes.txt"
    f.write_text("Python is a programming language.\n" * 40, encoding="utf-8")
    chunks, errors = load_files([f])
    assert len(errors) == 0
    assert len(chunks) >= 1

def test_reingest_preserves_stats():
    # Unchanged chunk ids should keep their topic from existing_chunks
    text = "Some stable chunk text that is long enough to survive."
    chunk_id = make_chunk_id(text)
    existing = [{"id": chunk_id, "text": text, "source_file": "notes.txt",
                 "topic": "Existing Topic", "page": None, "line_start": None}]
    new_chunk = Chunk(id=chunk_id, source_file="notes.txt", text=text, topic="General")
    merged = merge_with_existing([new_chunk], existing)
    assert merged[0]["topic"] == "Existing Topic"

def test_reingest_drops_removed_ids():
    # Chunks not in new_chunks should not appear in merged result
    old_chunk = {"id": "removed000000", "text": "old text", "source_file": "a.txt",
                 "topic": "T", "page": None, "line_start": None}
    new_chunk = Chunk(id="newid0000000", source_file="a.txt", text="new text", topic="General")
    # We need a 12-char id
    from recall_loop.ingest import make_chunk_id
    new_chunk = Chunk(id=make_chunk_id("new text here x"), source_file="a.txt",
                      text="new text here x", topic="General")
    merged = merge_with_existing([new_chunk], [old_chunk])
    ids = [c["id"] for c in merged]
    assert "removed000000" not in ids
    assert new_chunk.id in ids
