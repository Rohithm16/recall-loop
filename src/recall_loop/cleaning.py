import re
from collections import Counter
from typing import Optional

# Regex to detect standalone page numbers like "42" or "Page 42" on their own line
PAGE_NUMBER_RE = re.compile(r"^\s*(Page\s+)?\d+\s*$", re.IGNORECASE)

# Join words hyphenated across line breaks (PDF artifact like "infor-\nmation" -> "information")
def rejoin_hyphenated(text: str) -> str:
    return re.sub(r"-\r?\n\s*([a-zA-Z])", r"\1", text)

# Remove lines that look like standalone page numbers
def remove_page_numbers(lines: list[str]) -> list[str]:
    return [line for line in lines if not PAGE_NUMBER_RE.match(line)]

# Find lines that appear on more than half the pages and return them as a set (headers/footers)
def find_repeated_lines(pages_text: list[str]) -> set[str]:
    # A repeated line is one that appears verbatim on more than half the pages
    if len(pages_text) < 2:
        return set()
    page_count = len(pages_text)
    threshold = page_count / 2
    line_counts: Counter = Counter()
    for page in pages_text:
        for line in set(page.splitlines()):
            stripped = line.strip()
            if stripped:
                line_counts[stripped] += 1
    return {line for line, count in line_counts.items() if count > threshold}

# Remove header/footer lines from a list of lines
def remove_repeated_lines(lines: list[str], repeated: set[str]) -> list[str]:
    return [line for line in lines if line.strip() not in repeated]

# Collapse multiple blank lines into one, normalize whitespace within lines
def normalize_whitespace(text: str) -> str:
    # Normalize line endings, collapse repeated spaces, keep paragraph breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    # Collapse more than 2 consecutive blank lines into 2
    result, blank_count = [], 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                result.append("")
        else:
            blank_count = 0
            result.append(line)
    return "\n".join(result).strip()

# Full cleanup pipeline for text from a single file (pages_text is list of per-page text for PDFs)
def clean_text(raw_text: str, pages_text: Optional[list[str]] = None) -> str:
    # Run all cleanup steps in order; pages_text is needed for header/footer detection
    text = rejoin_hyphenated(raw_text)
    repeated = find_repeated_lines(pages_text) if pages_text else set()
    lines = text.splitlines()
    lines = remove_page_numbers(lines)
    lines = remove_repeated_lines(lines, repeated)
    return normalize_whitespace("\n".join(lines))

# Check if PDF text is too sparse (likely a scanned/image PDF)
def is_scan_like(total_chars: int, num_pages: int) -> bool:
    # Returns True if the file looks like a scan and should be rejected
    from recall_loop.config import MIN_CHARS_PER_PAGE, MIN_TOTAL_CHARS
    if total_chars < MIN_TOTAL_CHARS:
        return True
    if num_pages > 0 and (total_chars / num_pages) < MIN_CHARS_PER_PAGE:
        return True
    return False
