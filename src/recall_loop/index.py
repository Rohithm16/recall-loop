from pathlib import Path
from typing import Optional
from recall_loop.models import Chunk

# Build a LlamaIndex vector index from our chunks and persist it
def build_index(chunks: list[Chunk], collection: str, settings) -> None:
    from llama_index.core import VectorStoreIndex, StorageContext, Settings as LISettings
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.gemini import GeminiEmbedding
    from recall_loop.store import index_dir

    idx_path = index_dir(collection)
    idx_path.mkdir(parents=True, exist_ok=True)

    # Set up Gemini embeddings
    embed_model = GeminiEmbedding(
        model_name=settings.embed_model,
        api_key=settings.api_key,
    )
    LISettings.embed_model = embed_model
    LISettings.llm = None  # We don't use LlamaIndex's LLM

    # Build nodes from our chunks, carrying our chunk ids as metadata
    nodes = []
    for chunk in chunks:
        node = TextNode(
            text=chunk.text,
            id_=chunk.id,
            metadata={
                "chunk_id": chunk.id,
                "source_file": chunk.source_file,
                "page": chunk.page,
                "line_start": chunk.line_start,
                "topic": chunk.topic,
            }
        )
        nodes.append(node)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(idx_path))

# Load a persisted LlamaIndex index
def load_index(collection: str, settings):
    from llama_index.core import StorageContext, load_index_from_storage, Settings as LISettings
    from llama_index.embeddings.gemini import GeminiEmbedding
    from recall_loop.store import index_dir

    embed_model = GeminiEmbedding(
        model_name=settings.embed_model,
        api_key=settings.api_key,
    )
    LISettings.embed_model = embed_model
    LISettings.llm = None

    idx_path = index_dir(collection)
    storage_context = StorageContext.from_defaults(persist_dir=str(idx_path))
    return load_index_from_storage(storage_context)

# Retrieve the top-k most relevant chunks for a query, excluding flagged chunk ids
def retrieve_chunks(query: str, collection: str, settings,
                    flagged_ids: Optional[set[str]] = None,
                    always_include_id: Optional[str] = None) -> list[dict]:
    from recall_loop.store import load_chunks

    flagged_ids = flagged_ids or set()
    index = load_index(collection, settings)
    retriever = index.as_retriever(similarity_top_k=settings.why_top_k + len(flagged_ids) + 1)
    nodes = retriever.retrieve(query)

    # Load all chunks to get metadata
    all_chunks = {c["id"]: c for c in load_chunks(collection)}

    results = []
    seen_ids = set()

    # Always include the current question's chunk first if requested
    if always_include_id and always_include_id in all_chunks:
        c = all_chunks[always_include_id]
        results.append(c)
        seen_ids.add(always_include_id)

    for node in nodes:
        chunk_id = node.metadata.get("chunk_id", node.node_id)
        if chunk_id in flagged_ids or chunk_id in seen_ids:
            continue
        if chunk_id in all_chunks:
            results.append(all_chunks[chunk_id])
            seen_ids.add(chunk_id)
        if len(results) >= settings.why_top_k:
            break

    return results
