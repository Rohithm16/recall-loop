import hashlib
import re
from pathlib import Path
from typing import Optional
from recall_loop.cleaning import clean_text, is_scan_like
from recall_loop.models import Chunk
from recall_loop.config import get_settings, MIN_CHUNK_CHARS

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# Compute the chunk id: first 12 hex chars of SHA-256 of the text
def make_chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

# Read a text or markdown file, return cleaned text and line numbers
def read_text_file(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    chunks = split_into_chunks(cleaned, str(path.name))
    return chunks

# Extract text from a PDF using PyMuPDF, one string per page
def extract_pdf_pages(path: Path) -> list[str]:
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages

# Read a PDF file, check for scans, return cleaned chunks with page numbers
def read_pdf_file(path: Path) -> list[Chunk]:
    pages_text = extract_pdf_pages(path)
    total_chars = sum(len(p) for p in pages_text)
    if is_scan_like(total_chars, len(pages_text)):
        raise ValueError(
            f"This looks like a scanned or image-based file. Please upload a text-based version."
        )
    raw_text = "\n".join(pages_text)
    cleaned = clean_text(raw_text, pages_text)
    chunks = split_into_chunks(cleaned, str(path.name))
    # Assign page numbers by finding which page each chunk text is closest to
    chunks = assign_page_numbers(chunks, pages_text, path.name)
    return chunks

# Assign page numbers to chunks by matching chunk text against per-page text
def assign_page_numbers(chunks: list[Chunk], pages_text: list[str], source_file: str) -> list[Chunk]:
    # For each chunk, find which page contains the most words from the chunk
    result = []
    for chunk in chunks:
        best_page = find_best_page(chunk.text, pages_text)
        result.append(chunk.model_copy(update={"page": best_page, "source_file": source_file}))
    return result

# Find which page (1-indexed) contains the most overlap with the chunk text
def find_best_page(chunk_text: str, pages_text: list[str]) -> Optional[int]:
    chunk_words = set(chunk_text.lower().split())
    best_score, best_page = 0, 1
    for i, page in enumerate(pages_text):
        score = len(chunk_words & set(page.lower().split()))
        if score > best_score:
            best_score, best_page = score, i + 1
    return best_page

# Split cleaned text into chunks using LlamaIndex sentence splitter
def split_into_chunks(text: str, source_file: str) -> list[Chunk]:
    try:
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core import Document
        settings = get_settings()
        splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        doc = Document(text=text)
        nodes = splitter.get_nodes_from_documents([doc])
        raw_chunks = [node.get_content() for node in nodes]
    except ImportError:
        # Simple fallback splitter if llama_index is not installed
        paragraphs = text.split("\n\n")
        raw_chunks = [p.strip() for p in paragraphs if p.strip()]

    # Merge too-short chunks into neighbors
    merged = merge_short_chunks(raw_chunks)
    result = []
    line_cursor = 1
    for chunk_text in merged:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        line_start = find_line_start(text, chunk_text, line_cursor)
        chunk_id = make_chunk_id(chunk_text)
        result.append(Chunk(
            id=chunk_id,
            source_file=source_file,
            line_start=line_start,
            text=chunk_text,
            topic="General",
        ))
        if line_start:
            line_cursor = line_start
    return result

# Merge chunks shorter than MIN_CHUNK_CHARS into the next chunk
def merge_short_chunks(chunks: list[str]) -> list[str]:
    merged = []
    pending = ""
    for chunk in chunks:
        if pending:
            chunk = pending + "\n" + chunk
            pending = ""
        if len(chunk.strip()) < MIN_CHUNK_CHARS and len(chunk.strip()) > 0:
            pending = chunk.strip()
        else:
            merged.append(chunk)
    if pending:
        if merged:
            merged[-1] = merged[-1] + "\n" + pending
        else:
            merged.append(pending)
    return merged

# Find the 1-indexed line number where chunk_text starts in full_text
def find_line_start(full_text: str, chunk_text: str, hint: int = 1) -> Optional[int]:
    lines = full_text.splitlines()
    first_line = chunk_text.splitlines()[0].strip() if chunk_text.splitlines() else ""
    for i, line in enumerate(lines):
        if first_line and first_line in line:
            return i + 1
    return None

# Load all chunks from the given file paths, return chunks and a list of errors
def load_files(paths: list[Path]) -> tuple[list[Chunk], list[str]]:
    all_chunks: list[Chunk] = []
    errors: list[str] = []
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            errors.append(f"Unsupported file type: {path.name} (supported: .txt, .md, .pdf)")
            continue
        try:
            if path.suffix.lower() == ".pdf":
                chunks = read_pdf_file(path)
            else:
                chunks = read_text_file(path)
            all_chunks.extend(chunks)
        except ValueError as e:
            errors.append(f"{path.name}: {e}")
        except Exception as e:
            errors.append(f"{path.name}: unexpected error: {e}")
    return all_chunks, errors

# Merge new chunks with existing chunks, preserving stats for unchanged ids
def merge_with_existing(new_chunks: list[Chunk], existing_chunks: list[dict]) -> list[dict]:
    # Keep chunks whose text (and thus id) is unchanged, add new ones
    existing_by_id = {c["id"]: c for c in existing_chunks}
    result = []
    for chunk in new_chunks:
        d = chunk.model_dump()
        if chunk.id in existing_by_id:
            # Preserve topic if already labeled
            d["topic"] = existing_by_id[chunk.id].get("topic", "General")
        result.append(d)
    return result
