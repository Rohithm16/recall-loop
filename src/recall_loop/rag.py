from recall_loop.models import RagAnswer

# Format a source label string from a chunk dict
def format_source(chunk: dict) -> str:
    src = chunk.get("source_file", "?")
    page = chunk.get("page")
    line = chunk.get("line_start")
    if page:
        return f"{src}:p{page}"
    elif line:
        return f"{src}:L{line}"
    return src

# Build the RAG prompt from a question and a list of retrieved chunk dicts
def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = [
        f"[{i+1}] ({format_source(c)})\n{c['text']}"
        for i, c in enumerate(chunks)
    ]
    context = "\n\n".join(context_parts)
    return (
        f"Answer the question using ONLY the provided notes. Do not use outside knowledge.\n"
        f"If the notes don't cover the question, set covered=false.\n"
        f"Cite sources as [1], [2], etc., matching the numbers above.\n"
        f"List the source strings you used in used_sources.\n\n"
        f"Notes:\n{context}\n\nQuestion: {question}"
    )

# Answer a 'why' follow-up question using retrieved chunks; prints result to terminal
def answer_why(
    question: str,
    collection: str,
    current_chunk_id: str,
    flagged_ids: set[str],
    settings,
    retrieve_fn=None,
    llm_call=None,
) -> None:
    if retrieve_fn is None:
        from recall_loop.index import retrieve_chunks
        retrieve_fn = retrieve_chunks
    if llm_call is None:
        from recall_loop.llm import call_structured
        llm_call = call_structured

    chunks = retrieve_fn(
        question, collection, settings,
        flagged_ids=flagged_ids,
        always_include_id=current_chunk_id,
    )
    if not chunks:
        print("No relevant chunks found in your notes.")
        return

    result = llm_call(build_rag_prompt(question, chunks), RagAnswer, settings)

    if not result.covered:
        print("Your notes don't seem to cover this.")
        print("Retrieved sources:")
        for c in chunks:
            print(f"  - {format_source(c)}")
        return

    print(f"\nAnswer: {result.answer}")
    print("Sources:")
    for i, src in enumerate(result.used_sources):
        print(f"  [{i+1}] {src}")
