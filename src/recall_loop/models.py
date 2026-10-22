from __future__ import annotations
from pydantic import BaseModel
from typing import Optional
from recall_loop.mistakes import MistakeType

# Represents a single text chunk from the user's notes
class Chunk(BaseModel):
    id: str  # first 12 hex chars of SHA-256 of cleaned text
    source_file: str
    page: Optional[int] = None
    line_start: Optional[int] = None
    text: str
    topic: str = "General"

# One answer option in a multiple-choice question
class Option(BaseModel):
    text: str
    mistake_type: Optional[MistakeType] = None  # None for correct option

# A full multiple-choice question generated from a chunk
class MCQ(BaseModel):
    question: str
    options: list[Option]  # exactly 4
    correct_index: int  # 0 to 3
    explanation: str

# Per-chunk statistics stored in state.json
class ChunkStats(BaseModel):
    asked: int = 0
    correct: int = 0
    last_asked_session: Optional[int] = None
    mistakes: dict[str, int] = {}  # MistakeType value -> count

    def model_post_init(self, __context):
        # Ensure all mistake types are initialized to 0
        from recall_loop.mistakes import MistakeType
        for mt in MistakeType:
            if mt.value not in self.mistakes:
                self.mistakes[mt.value] = 0

# One entry in the recent_questions list
class RecentQuestion(BaseModel):
    chunk_id: str
    text: str

# One session summary stored in state.json
class SessionRecord(BaseModel):
    id: int
    date: str
    asked: int
    correct: int

# The full state.json structure
class AppState(BaseModel):
    schema_version: int = 1
    chunk_stats: dict[str, ChunkStats] = {}
    flagged: list[str] = []
    recent_questions: list[RecentQuestion] = []
    sessions: list[SessionRecord] = []

# Structured output model for topic labeling
class TopicLabels(BaseModel):
    labels: list[str]  # one label per chunk, in the same order

# Structured output model for the why/rag answer
class RagAnswer(BaseModel):
    answer: str
    used_sources: list[str]  # e.g. ["notes.txt:10", "notes.pdf:p3"]
    covered: bool  # False if the notes don't cover the question

# Structured output for solver check
class SolverResult(BaseModel):
    chosen_index: int  # the index the solver thinks is correct
