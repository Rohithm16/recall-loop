import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Thresholds for scan/empty PDF rejection
MIN_CHARS_PER_PAGE = 200
MIN_TOTAL_CHARS = 500
# Merge chunks shorter than this character count into a neighbor
MIN_CHUNK_CHARS = 200
# Maximum recent_questions to keep in state
MAX_RECENT_QUESTIONS = 200
# Number of questions to avoid repeating within a session
NO_REPEAT_WINDOW = 5

class Settings:
    api_key: str
    chat_model: str
    embed_model: str
    data_dir: Path
    solver_check: bool
    chunk_size: int
    chunk_overlap: int
    why_top_k: int
    questions_per_session: int
    random_seed: int | None
    min_seconds_between_calls: float

def get_settings() -> Settings:
    # Build and return the Settings object from environment variables.
    s = Settings()
    s.api_key = os.environ.get("GEMINI_API_KEY", "")
    s.chat_model = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
    s.embed_model = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")
    s.data_dir = Path(os.environ.get("RECALL_LOOP_DATA_DIR", "./data"))
    s.solver_check = os.environ.get("SOLVER_CHECK", "on").lower() == "on"
    s.chunk_size = int(os.environ.get("CHUNK_SIZE", "512"))
    s.chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "64"))
    s.why_top_k = int(os.environ.get("WHY_TOP_K", "4"))
    s.questions_per_session = int(os.environ.get("QUESTIONS_PER_SESSION", "10"))
    seed = os.environ.get("RANDOM_SEED", "")
    s.random_seed = int(seed) if seed.strip() else None
    s.min_seconds_between_calls = float(os.environ.get("GEMINI_MIN_SECONDS_BETWEEN_CALLS", "0"))
    return s
