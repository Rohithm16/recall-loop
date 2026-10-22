import pytest
import sys

def test_missing_api_key_exits(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    import importlib, recall_loop.config as cfg
    importlib.reload(cfg)
    with pytest.raises(SystemExit) as exc:
        from recall_loop.cli import _check_api_key
        _check_api_key()
    assert exc.value.code != 0
    assert "GEMINI_API_KEY" in capsys.readouterr().out

def test_empty_collection_exits(tmp_path, monkeypatch):
    import recall_loop.store as store_mod
    monkeypatch.setattr(store_mod, "chunks_path", lambda name: tmp_path / "chunks.json")
    with pytest.raises(SystemExit) as exc:
        from recall_loop.cli import _check_collection_not_empty
        _check_collection_not_empty("empty")
    assert exc.value.code != 0

def test_reset_cancelled_on_no(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "no")
    import argparse
    from recall_loop.cli import cmd_reset
    cmd_reset(argparse.Namespace(collection="default"))
    assert "cancelled" in capsys.readouterr().out.lower()
