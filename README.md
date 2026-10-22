# Recall Loop

A terminal-based quiz coach for your typed study notes. Recall Loop generates multiple-choice questions from your notes, tags each wrong option with the misconception it represents, tracks your weak topics over time, guarantees coverage across all your notes, and answers follow-up questions (`why`) grounded directly in your notes with citations.

---

## 1. The Problem

For reviewing short notes in a single sitting, opening a chat with Gemini and asking "quiz me on this" gets most of the value. 

However, a raw chat model quickly breaks down for serious long-term study:
- **No persistent tracking:** Chat history forgets what you missed yesterday or last week.
- **No coverage guarantee:** A model naturally fixates on whatever was recently mentioned or whatever is easiest to summarize, skipping tricky sections.
- **Ungrounded follow-ups:** When you ask "why?", chat models easily hallucinate facts from their pre-training data that contradict or exceed what is in your notes.
- **No mistake diagnostics:** Knowing an answer was wrong is only half the battle. Knowing *why* you chose a wrong answer (e.g. confusing two concepts vs. reversing cause and effect) reveals systematic gaps in understanding.
- **No measurement:** You cannot measure whether your quiz session actually tested your entire corpus or repeated the same facts.

Recall Loop solves this by pairing Gemini's generative strength with deterministic Python session control.

---

## 2. Core Design Principle

> **Gemini generates, code controls.**

Gemini writes the questions, tags distractors with misconceptions, and formulates grounded answers. **Plain Python code controls the session**:
- What is asked next (coverage-first, then weakness-weighted).
- What is tracked and persisted (per-chunk and per-topic stats, mistake patterns).
- What is considered valid (rigid schema and grounding validation).
- What progress is saved between sessions.

The core logic modules (`cleaning`, `models`, `mistakes`, `store`, `validators`, `tracker`, `selector`, `report`) have **zero dependencies** on AI frameworks and run completely offline without an API key or network connection.

---

## 3. Architecture

### Ingestion Pipeline
```
[Notes (.txt / .md / .pdf)]
             │
             ▼
     cleaning.py (pure Python: normalize whitespace, rejoin hyphens, strip page numbers/headers)
             │
             ▼
      ingest.py (SentenceSplitter, chunk IDs = SHA-256 hex)
             │
             ▼
      topics.py (Gemini structured batch topic labeling)
             │
             ▼
      index.py (LlamaIndex VectorStoreIndex + Gemini Embeddings)
             │
             ▼
  [data/collections/<name>/] (chunks.json, state.json, index/)
```

### Quiz Session (LangGraph Loop)
```
          init_session
               │
               ▼
         select_chunk ◄────────────────────────────────┐
               │                                       │
               ▼                                       │
       generate_question                               │
               │                                       │
       [Validation Check]                              │
          ├── Invalid (< 2 retries) ──► retry          │
          └── Invalid (>= 2 retries) ─► skip chunk ───┤
               │                                       │
               ▼ Valid                                 │
        present_question                               │
               │                                       │
   [INTERRUPT: wait for answer]                        │
               │                                       │
       route_answer_input                              │
          ├── 'skip' ──────────────────────────────────┘
          ├── 'flag' ──► flag chunk ───────────────────┤
          ├── 'quit' ──► end_session                   │
          └── A/B/C/D ──► record_result & save         │
                              │                        │
                              ▼                        │
                        show_feedback                  │
                              │                        │
               [INTERRUPT: wait for post-cmd]          │
                              │                        │
                      route_post_command               │
          ├── Enter ──► check_end ─────────────────────┘
          ├── 'source' ──► print chunk ──► wait for post-cmd
          ├── 'why <q>' ──► RAG answer ──► wait for post-cmd
          ├── 'flag' ──► flag chunk ─────► check_end
          └── 'quit' ──► end_session
                              │
                              ▼
                         end_session (summary & save)
```

### Why / RAG Retrieval Path
```
[User: "why <question>"]
             │
             ▼
  Index Retrieval (top-K chunks from VectorStore, excluding flagged chunks)
             │
             ▼
  + Always include current question's source chunk
             │
             ▼
  Gemini Structured Call (RagAnswer: answer, used_sources, covered)
             │
             ├── covered == False ─► "Your notes don't seem to cover this." (+ sources)
             └── covered == True  ─► Grounded answer with citations [1], [2]
```

---

## 4. Where Each Tool Is Used and Why

| Tool | Where | Purpose | Could it be removed? |
|---|---|---|---|
| **LlamaIndex** | `index.py`, `ingest.py` | Sentence splitting, vector indexing, and chunk retrieval with `google.genai` embeddings. | Yes. Could be replaced with a custom sliding-window splitter and a local SQLite/Numpy cosine search. |
| **Google GenAI** | `llm.py`, `index.py` | Official `google-genai` SDK for structured outputs with Pydantic schema enforcement and semantic embeddings. | Yes. Could use raw HTTP requests directly. |
| **LangGraph** | `graph.py` | Quiz session state machine with `interrupt()` and `Command(resume=...)` for human-in-the-loop terminal turns. | Yes. Could be written as a standard Python `while` loop with `input()`. |

> **Note on framework independence:** None of the core algorithms (tracking, weakness scoring, coverage selection, text cleanup, mistake taxonomy, reporting) import any of these frameworks.

---

## 5. Usage

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/recall-loop.git
cd recall-loop

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Configuration
Copy the example environment file and add your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_CHAT_MODEL=gemini-2.0-flash
GEMINI_EMBED_MODEL=gemini-embedding-2-preview
RECALL_LOOP_DATA_DIR=./data
SOLVER_CHECK=on
CHUNK_SIZE=512
CHUNK_OVERLAP=64
WHY_TOP_K=4
QUESTIONS_PER_SESSION=10
```

### Commands

#### 1. Ingest Notes
```bash
python -m recall_loop ingest data/sample_notes/python_basics.txt data/sample_notes/algorithms_basics.txt --collection default
```

#### 2. Start Quiz
```bash
python -m recall_loop quiz --collection default --questions 10
```
During the quiz:
- Enter `A`, `B`, `C`, or `D` to answer.
- Type `skip` to pass without recording.
- Type `flag` to exclude the chunk from future sessions.
- Type `quit` to exit immediately and save progress.

After answering:
- Press `Enter` to move to the next question.
- Type `why <question>` to get an explanation grounded only in your notes.
- Type `source` to inspect the full source chunk text.
- Type `flag` to mark the chunk as inaccurate.

#### 3. View Progress Report
```bash
python -m recall_loop report --collection default
```

#### 4. Reset Progress
```bash
python -m recall_loop reset --collection default
```

---

## 6. Supported Input and Limitations

- **Supported input:** UTF-8 `.txt`, `.md`, and digital PDFs with selectable text.
- **Unsupported input:** Scanned documents, handwritten notes, images, or any OCR. PDFs with fewer than 200 characters per page or 500 characters total are rejected with an explicit error.
- **Flawed questions:** Generative models can produce ambiguous distractors or incorrect answers. Recall Loop mitigates this with a two-pass **Solver Check** (`SOLVER_CHECK=on`) and the `flag` command.
- **Mistake tags interpretation:** Tags (`concept_confusion`, `wrong_detail`, `incomplete`, `reversed`, `overgeneralized`, `other`) are assigned by Gemini to each wrong option at generation time. They represent the misconception the option was *designed* to embody, not proof of user intent.
- **API limits and privacy:** Rate limits on the Gemini free tier can cause throttling. Your note content is transmitted to Google's API, so do not ingest confidential data.

---

## 7. Evaluation Results

Run the scripts in `eval/` on your collection to fill in the evaluation table:

| Metric | Measured Value | Threshold / Notes |
|---|---|---|
| **Answerable from chunk** | TBD | Human judgment via `eval/review_questions.py` |
| **Exactly one correct option** | TBD | Human judgment via `eval/review_questions.py` |
| **Wrong-option tags reasonable** | TBD | Human judgment via `eval/review_questions.py` |
| **Solver agreement rate** | TBD | Automated solver check agreement rate |
| **Coverage vs baseline** | TBD | Evaluated via `eval/baseline_compare.py` |
| **Repetition vs baseline** | TBD | Evaluated via `eval/baseline_compare.py` |
| **Retrieval hit@1** | TBD | Evaluated via `eval/retrieval_eval.py` |
| **Retrieval hit@3** | TBD | Evaluated via `eval/retrieval_eval.py` |
| **Retrieval hit@5** | TBD | Evaluated via `eval/retrieval_eval.py` |

---

## 8. Choices I Made

- **Python Version:** Python 3.13.2 (detected on system).
- **PDF Extraction:** `pymupdf` (PyMuPDF / `fitz`) — high-performance, actively maintained, handles multi-column layouts and text bounding accurately.
- **Gemini Chat Model:** `gemini-3.5-flash` — fast generation, strong structured output adherence, cost-effective.
- **Gemini Embedding Model:** `gemini-embedding-2-preview` — current recommended multimodal/text embedding model for semantic retrieval via the official `google-genai` SDK.
- **LangGraph Interrupt API:** Modern `from langgraph.types import interrupt, Command` with `InMemorySaver` checkpointer.
- **Chunk Size / Overlap:** 512 tokens with 64 token overlap, merging chunks smaller than 200 characters into neighbors.
