from pathlib import Path
from typing import Optional
from pydantic import PrivateAttr
from google import genai
from llama_index.core.base.embeddings.base import BaseEmbedding
from recall_loop.models import Chunk

# Custom LlamaIndex embedding class using Google's current google.genai SDK
class GenAIEmbedding(BaseEmbedding):
    model_name: str
    api_key: str
    _client: genai.Client = PrivateAttr()

    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        self._client = genai.Client(api_key=api_key)

    def _get_query_embedding(self, query: str) -> list[float]:
        res = self._client.models.embed_content(model=self.model_name, contents=query)
        return res.embeddings[0].values

    def _get_text_embedding(self, text: str) -> list[float]:
        res = self._client.models.embed_content(model=self.model_name, contents=text)
        return res.embeddings[0].values

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._get_text_embedding(t) for t in texts]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

# Build a LlamaIndex vector index from chunks and persist it using google.genai embeddings
def build_index(chunks: list[Chunk], collection: str, settings) -> None:
    from llama_index.core import VectorStoreIndex, Settings as LISettings
    from llama_index.core.schema import TextNode
    from recall_loop.store import index_dir

    idx_path = index_dir(collection)
    idx_path.mkdir(parents=True, exist_ok=True)

    embed_model = GenAIEmbedding(
        model_name=settings.embed_model,
        api_key=settings.api_key,
    )
    LISettings.embed_model = embed_model
    LISettings.llm = None  # We don't use LlamaIndex's LLM

    # Build nodes carrying chunk metadata
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

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=str(idx_path))

# Load a persisted LlamaIndex index
def load_index(collection: str, settings):
    from llama_index.core import StorageContext, load_index_from_storage, Settings as LISettings
    from recall_loop.store import index_dir

    embed_model = GenAIEmbedding(
        model_name=settings.embed_model,
        api_key=settings.api_key,
    )
    LISettings.embed_model = embed_model
    LISettings.llm = None

    idx_path = index_dir(collection)
    storage_context = StorageContext.from_defaults(persist_dir=str(idx_path))
    return load_index_from_storage(storage_context, embed_model=embed_model)

# Retrieve the top-k most relevant chunks for a query, excluding flagged chunk ids
def retrieve_chunks(
    query: str,
    collection: str,
    settings,
    flagged_ids: Optional[set[str]] = None,
    always_include_id: Optional[str] = None,
) -> list[dict]:
    from recall_loop.store import load_chunks

    flagged_ids = flagged_ids or set()
    index = load_index(collection, settings)
    retriever = index.as_retriever(similarity_top_k=settings.why_top_k + len(flagged_ids) + 1)
    nodes = retriever.retrieve(query)

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
