import pytest
import json
from pathlib import Path
from recall_loop.store import atomic_write_json, load_state, save_state
from recall_loop.models import AppState, ChunkStats

def test_atomic_write_round_trip(tmp_path):
    # Save and reload should produce the same data
    data = {"key": "value", "num": 42}
    p = tmp_path / "test.json"
    atomic_write_json(p, data)
    loaded = json.loads(p.read_text())
    assert loaded == data

def test_atomic_write_no_partial(tmp_path, monkeypatch):
    # Even if renamed fails, original should not be partially written
    p = tmp_path / "test.json"
    data = {"ok": True}
    atomic_write_json(p, data)
    # File should exist and be valid
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded == data

def test_save_load_state_round_trip(tmp_path, monkeypatch):
    # State saved and loaded should match
    import recall_loop.store as store_mod
    monkeypatch.setattr(store_mod, "state_path", lambda name: tmp_path / "state.json")
    monkeypatch.setattr(store_mod, "collection_dir", lambda name: tmp_path)
    state = AppState()
    state.chunk_stats["abc123def456"] = ChunkStats(asked=2, correct=1)
    state.flagged = ["abc123def456"]
    save_state("test", state)
    loaded = load_state("test")
    assert loaded.flagged == ["abc123def456"]
    assert loaded.chunk_stats["abc123def456"].asked == 2

def test_corrupt_state_backed_up(tmp_path, monkeypatch):
    # A corrupt state.json should be backed up and a fresh state returned
    import recall_loop.store as store_mod
    state_file = tmp_path / "state.json"
    state_file.write_text("NOT VALID JSON{{{")
    monkeypatch.setattr(store_mod, "state_path", lambda name: state_file)
    state = load_state("test")
    assert isinstance(state, AppState)
    # Backup should exist
    backups = list(tmp_path.glob("state.json.corrupt-*"))
    assert len(backups) == 1
