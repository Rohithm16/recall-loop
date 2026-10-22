import random
import sys
from datetime import datetime
from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from recall_loop.models import AppState, MCQ, RecentQuestion, SessionRecord
from recall_loop.mistakes import MistakeType, MISTAKE_LABELS, MISTAKE_DESCRIPTIONS
from recall_loop.tracker import record_result
from recall_loop.selector import select_next_chunk
from recall_loop.store import load_chunks, load_state, save_state
from recall_loop.report import print_session_summary
from recall_loop.config import get_settings, MAX_RECENT_QUESTIONS

OPTION_LETTERS = ["A", "B", "C", "D"]

# The graph state holds only small serializable data
class SessionState(TypedDict):
    collection: str
    session_id: int
    questions_total: int
    questions_asked: int
    questions_correct: int
    current_chunk_id: Optional[str]
    current_mcq: Optional[dict]
    session_recent: list[str]
    skipped_chunks: list[str]
    done: bool
    answer_revealed: bool
    chosen_index: Optional[int]
    _answer_input: Optional[str]
    _post_input: Optional[str]
    _retry_answer: bool
    _stay_in_post: bool
    settings_dict: dict

# Format a readable source label from a chunk dict
def chunk_source_label(chunk: dict) -> str:
    src = chunk.get("source_file", "?")
    page = chunk.get("page")
    line = chunk.get("line_start")
    if page:
        return f"{src} (p.{page})"
    elif line:
        return f"{src} (line {line})"
    return src

# Print the question and options to the terminal
def print_question(state: SessionState, chunk: dict, mcq: MCQ) -> None:
    print(f"\n--- Question {state['questions_asked'] + 1} of {state['questions_total']} ---")
    print(f"Topic: {chunk.get('topic', 'General')}  |  Source: {chunk_source_label(chunk)}")
    print(f"\n{mcq.question}\n")
    for i, opt in enumerate(mcq.options):
        print(f"  {OPTION_LETTERS[i]}) {opt.text}")
    print()

# Print immediate feedback after answer is revealed
def print_feedback(mcq: MCQ, chosen_index: int, correct: bool, chunk: dict) -> None:
    if correct:
        print("\n✓ Correct!")
    else:
        print("\n✗ Incorrect.")
        chosen_opt = mcq.options[chosen_index]
        if chosen_opt.mistake_type and chosen_opt.mistake_type != MistakeType.other:
            mt = chosen_opt.mistake_type
            label = MISTAKE_LABELS.get(mt, mt.value)
            desc = MISTAKE_DESCRIPTIONS.get(mt, "")
            print(f"Your choice looked like: {label} — {desc}")
        correct_text = mcq.options[mcq.correct_index].text
        print(f"Correct answer: {OPTION_LETTERS[mcq.correct_index]}) {correct_text}")
    print(f"Explanation: {mcq.explanation}")
    print(f"Source: {chunk_source_label(chunk)}")

# Node: select next chunk based on coverage and weakness
def node_select_chunk(state: SessionState) -> dict:
    if state["questions_asked"] >= state["questions_total"]:
        return {"done": True}
    chunks = load_chunks(state["collection"])
    app_state = load_state(state["collection"])
    settings = _settings_from_dict(state["settings_dict"])
    seed = settings.random_seed
    rng = random.Random((seed if seed is not None else 0) + state["questions_asked"])
    cid = select_next_chunk(
        chunks, app_state,
        state["session_recent"],
        set(state["skipped_chunks"]),
        rng,
    )
    if cid is None:
        return {"done": True}
    return {"current_chunk_id": cid, "done": False}

# Node: generate a validated question for the current chunk
def node_generate_question(state: SessionState) -> dict:
    from recall_loop.generator import generate_question
    chunks = load_chunks(state["collection"])
    app_state = load_state(state["collection"])
    settings = _settings_from_dict(state["settings_dict"])
    chunk = next((c for c in chunks if c["id"] == state["current_chunk_id"]), None)
    if not chunk:
        return {"done": True}
    rng = random.Random((settings.random_seed or 0) + state["questions_asked"] + len(state["skipped_chunks"]))
    recent = [r.model_dump() for r in app_state.recent_questions]
    mcq = generate_question(chunk, recent, rng, settings)
    if mcq is None:
        skipped = list(state["skipped_chunks"]) + [state["current_chunk_id"]]
        return {"skipped_chunks": skipped, "current_chunk_id": None}
    return {"current_mcq": mcq.model_dump(), "_retry_answer": False}

# Node: display question to the user
def node_present_question(state: SessionState) -> dict:
    chunks = load_chunks(state["collection"])
    chunk = next((c for c in chunks if c["id"] == state["current_chunk_id"]), None)
    mcq = MCQ.model_validate(state["current_mcq"])
    print_question(state, chunk, mcq)
    return {"_retry_answer": False}

# Node: interrupt and wait for answer input
def node_wait_answer(state: SessionState) -> dict:
    print("Enter A/B/C/D, 'skip', 'flag', or 'quit': ", end="", flush=True)
    ans = interrupt("waiting_for_answer")
    return {"_answer_input": str(ans)}

# Node: route user answer input to appropriate handler
def node_route_answer(state: SessionState) -> dict:
    raw = (state.get("_answer_input") or "").strip().lower()
    mcq = MCQ.model_validate(state["current_mcq"])
    collection = state["collection"]
    chunks = load_chunks(collection)
    chunk = next((c for c in chunks if c["id"] == state["current_chunk_id"]), None)

    if raw in ("a", "b", "c", "d"):
        chosen_index = OPTION_LETTERS.index(raw.upper())
        correct = (chosen_index == mcq.correct_index)
        app_state = load_state(collection)
        mt = None
        if not correct:
            mt_enum = mcq.options[chosen_index].mistake_type
            mt = mt_enum.value if mt_enum else None
        record_result(app_state, state["current_chunk_id"], correct, mt, state["session_id"])
        rq = RecentQuestion(chunk_id=state["current_chunk_id"], text=mcq.question)
        app_state.recent_questions.append(rq)
        if len(app_state.recent_questions) > MAX_RECENT_QUESTIONS:
            app_state.recent_questions = app_state.recent_questions[-MAX_RECENT_QUESTIONS:]
        save_state(collection, app_state)
        print_feedback(mcq, chosen_index, correct, chunk)
        new_recent = list(state["session_recent"]) + [state["current_chunk_id"]]
        return {
            "questions_asked": state["questions_asked"] + 1,
            "questions_correct": state["questions_correct"] + (1 if correct else 0),
            "session_recent": new_recent,
            "answer_revealed": True,
            "chosen_index": chosen_index,
            "_retry_answer": False,
        }
    elif raw == "skip":
        return {"current_chunk_id": None, "current_mcq": None, "answer_revealed": False, "_retry_answer": False}
    elif raw == "flag":
        app_state = load_state(collection)
        if state["current_chunk_id"] not in app_state.flagged:
            app_state.flagged.append(state["current_chunk_id"])
            save_state(collection, app_state)
            print("Chunk flagged.")
        return {"current_chunk_id": None, "current_mcq": None, "answer_revealed": False, "_retry_answer": False}
    elif raw == "quit":
        return {"done": True, "_retry_answer": False}
    else:
        print("Please enter A, B, C, D, 'skip', 'flag', or 'quit'.")
        return {"_retry_answer": True}

# Node: interrupt and wait for post-answer follow-up command
def node_wait_post(state: SessionState) -> dict:
    print("\nPress Enter to continue, 'why <question>', 'source', 'flag', or 'quit': ", end="", flush=True)
    post_cmd = interrupt("waiting_for_post")
    return {"_post_input": str(post_cmd)}

# Node: route post-answer follow-up command
def node_route_post(state: SessionState) -> dict:
    raw = (state.get("_post_input") or "").strip()
    collection = state["collection"]
    chunks = load_chunks(collection)
    chunk = next((c for c in chunks if c["id"] == state["current_chunk_id"]), None)
    settings = _settings_from_dict(state["settings_dict"])

    if raw == "" or raw.lower() == "enter":
        return {"answer_revealed": False, "current_chunk_id": None, "current_mcq": None, "_stay_in_post": False}
    elif raw.lower().startswith("why "):
        question = raw[4:].strip()
        app_state = load_state(collection)
        from recall_loop.rag import answer_why
        answer_why(question, collection, state["current_chunk_id"], set(app_state.flagged), settings)
        return {"_stay_in_post": True}
    elif raw.lower() == "source":
        if chunk:
            print(f"\n--- Source chunk ---\n{chunk['text']}\n")
        return {"_stay_in_post": True}
    elif raw.lower() == "flag":
        app_state = load_state(collection)
        if state["current_chunk_id"] not in app_state.flagged:
            app_state.flagged.append(state["current_chunk_id"])
            save_state(collection, app_state)
            print("Chunk flagged.")
        return {"answer_revealed": False, "current_chunk_id": None, "current_mcq": None, "_stay_in_post": False}
    elif raw.lower() == "quit":
        return {"done": True, "_stay_in_post": False}
    else:
        print("Commands: Enter (next), 'why <question>', 'source', 'flag', 'quit'")
        return {"_stay_in_post": True}

# Node: record session summary and print report
def node_end_session(state: SessionState) -> dict:
    chunks = load_chunks(state["collection"])
    app_state = load_state(state["collection"])
    app_state.sessions.append(SessionRecord(
        id=state["session_id"],
        date=datetime.now().isoformat(),
        asked=state["questions_asked"],
        correct=state["questions_correct"],
    ))
    save_state(state["collection"], app_state)
    print_session_summary(
        chunks, app_state,
        state["questions_asked"],
        state["questions_correct"],
        state["session_recent"],
    )
    return {"done": True}

# Restore settings object from dictionary
def _settings_from_dict(d: dict):
    settings = get_settings()
    for k, v in d.items():
        setattr(settings, k, v)
    return settings

# Build and compile the session StateGraph
def build_graph():
    builder = StateGraph(SessionState)
    builder.add_node("select_chunk", node_select_chunk)
    builder.add_node("generate_question", node_generate_question)
    builder.add_node("present_question", node_present_question)
    builder.add_node("wait_answer", node_wait_answer)
    builder.add_node("route_answer", node_route_answer)
    builder.add_node("wait_post", node_wait_post)
    builder.add_node("route_post", node_route_post)
    builder.add_node("end_session", node_end_session)

    builder.add_edge(START, "select_chunk")

    def after_select(s):
        if s.get("done") or not s.get("current_chunk_id"):
            return "end_session"
        return "generate_question"

    def after_generate(s):
        if s.get("done"):
            return "end_session"
        if not s.get("current_chunk_id"):
            return "select_chunk"
        return "present_question"

    def after_route_answer(s):
        if s.get("done"):
            return "end_session"
        if s.get("_retry_answer"):
            return "wait_answer"
        if s.get("answer_revealed"):
            return "wait_post"
        return "select_chunk"

    def after_route_post(s):
        if s.get("done"):
            return "end_session"
        if s.get("_stay_in_post"):
            return "wait_post"
        return "select_chunk"

    builder.add_conditional_edges("select_chunk", after_select)
    builder.add_conditional_edges("generate_question", after_generate)
    builder.add_edge("present_question", "wait_answer")
    builder.add_edge("wait_answer", "route_answer")
    builder.add_conditional_edges("route_answer", after_route_answer)
    builder.add_edge("wait_post", "route_post")
    builder.add_conditional_edges("route_post", after_route_post)
    builder.add_edge("end_session", END)

    return builder.compile(checkpointer=InMemorySaver())

# Run an interactive quiz session
def run_session(collection: str, questions_total: int, seed: Optional[int] = None) -> None:
    import signal
    settings = get_settings()
    if seed is not None:
        settings.random_seed = seed

    app_state = load_state(collection)
    session_id = len(app_state.sessions) + 1

    initial: SessionState = {
        "collection": collection,
        "session_id": session_id,
        "questions_total": questions_total,
        "questions_asked": 0,
        "questions_correct": 0,
        "current_chunk_id": None,
        "current_mcq": None,
        "session_recent": [],
        "skipped_chunks": [],
        "done": False,
        "answer_revealed": False,
        "chosen_index": None,
        "_answer_input": None,
        "_post_input": None,
        "_retry_answer": False,
        "_stay_in_post": False,
        "settings_dict": {
            "api_key": settings.api_key,
            "chat_model": settings.chat_model,
            "embed_model": settings.embed_model,
            "solver_check": settings.solver_check,
            "why_top_k": settings.why_top_k,
            "min_seconds_between_calls": settings.min_seconds_between_calls,
            "random_seed": settings.random_seed,
        },
    }

    graph = build_graph()
    config = {"configurable": {"thread_id": f"{collection}-{session_id}"}}

    def handle_ctrl_c(sig, frame):
        print("\n\nInterrupted. Saving progress...")
        try:
            st = graph.get_state(config).values
            chunks = load_chunks(collection)
            ap = load_state(collection)
            ap.sessions.append(SessionRecord(
                id=st.get("session_id", session_id),
                date=datetime.now().isoformat(),
                asked=st.get("questions_asked", 0),
                correct=st.get("questions_correct", 0),
            ))
            save_state(collection, ap)
            print_session_summary(
                chunks, ap,
                st.get("questions_asked", 0),
                st.get("questions_correct", 0),
                st.get("session_recent", []),
            )
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_ctrl_c)

    graph.invoke(initial, config)
    while True:
        graph_state = graph.get_state(config)
        if not graph_state.next:
            break
        try:
            user_input = input()
        except EOFError:
            user_input = "quit"
        graph.invoke(Command(resume=user_input), config)
