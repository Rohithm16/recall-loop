import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from recall_loop.models import AppState
from recall_loop.config import get_settings

# Returns the directory for a named collection
def collection_dir(name: str) -> Path:
    settings = get_settings()
    return settings.data_dir / "collections" / name

# Returns the path to chunks.json for a collection
def chunks_path(name: str) -> Path:
    return collection_dir(name) / "chunks.json"

# Returns the path to state.json for a collection
def state_path(name: str) -> Path:
    return collection_dir(name) / "state.json"

# Returns the path to the LlamaIndex index directory
def index_dir(name: str) -> Path:
    return collection_dir(name) / "index"

# Save data to a JSON file atomically (write temp, then rename)
def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.move(str(tmp), str(path))

# Load AppState from state.json, handling corrupt files gracefully
def load_state(name: str) -> AppState:
    path = state_path(name)
    if not path.exists():
        return AppState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AppState.model_validate(data)
    except Exception:
        # Back up the corrupt file and start fresh
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        backup = path.with_name(f"state.json.corrupt-{ts}")
        shutil.copy2(str(path), str(backup))
        print(f"Warning: state.json was corrupt. Backed up to {backup.name}. Starting fresh.")
        return AppState()

# Save AppState to state.json atomically
def save_state(name: str, state: AppState) -> None:
    atomic_write_json(state_path(name), state.model_dump())

# Load chunks list from chunks.json
def load_chunks(name: str) -> list[dict]:
    path = chunks_path(name)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

# Save chunks list to chunks.json atomically
def save_chunks(name: str, chunks: list[dict]) -> None:
    atomic_write_json(chunks_path(name), chunks)
