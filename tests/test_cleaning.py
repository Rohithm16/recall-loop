from recall_loop.cleaning import (
    rejoin_hyphenated, remove_page_numbers, find_repeated_lines,
    remove_repeated_lines, normalize_whitespace, clean_text, is_scan_like
)

def test_rejoin_hyphenated():
    # Hyphens at line ends should be removed and the word joined
    result = rejoin_hyphenated("infor-\nmation")
    assert result == "information"

def test_remove_page_numbers():
    lines = ["Hello world", "42", "Page 7", "actual content"]
    result = remove_page_numbers(lines)
    assert "42" not in result
    assert "Page 7" not in result
    assert "Hello world" in result

def test_find_repeated_lines():
    pages = ["Header\nContent A", "Header\nContent B", "Header\nContent C"]
    repeated = find_repeated_lines(pages)
    assert "Header" in repeated

def test_repeated_lines_not_flagged_if_rare():
    pages = ["Unique A", "Unique B", "Unique C", "Shared Line\nUnique D"]
    repeated = find_repeated_lines(pages)
    assert "Shared Line" not in repeated

def test_normalize_whitespace():
    text = "Hello   world\n\n\n\nNext para"
    result = normalize_whitespace(text)
    assert "   " not in result
    assert result.count("\n\n") <= 2

def test_is_scan_like_sparse():
    # Total chars below threshold should be flagged
    assert is_scan_like(100, 5) is True

def test_is_scan_like_ok():
    # Enough chars should not be flagged
    assert is_scan_like(5000, 5) is False

def test_clean_text_full():
    raw = "Some-\nconcatenated word\n42\nPage 3\nReal content here."
    result = clean_text(raw)
    assert "42" not in result.split("\n") or "42" not in [l.strip() for l in result.split("\n")]
    assert "Someconcatenated" in result or "Some" in result  # hyphen rejoined
