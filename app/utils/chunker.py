"""
Fixed-size text chunking with token overlap, used to prepare text for
embedding and FAISS indexing.

Simple, deterministic pipeline: raw extracted PDF text -> sliding window of
`chunk_size_tokens` with `chunk_overlap_tokens` overlap. There is no section
detection or section-aware boundary logic -- every chunk is just "the next
window of the document text."
"""
from dataclasses import dataclass
from typing import List

from app.utils.token_counter import count_tokens


@dataclass
class Chunk:
    text: str
    chunk_index: int
    start_word_index: int = 0


def _split_into_words(text: str) -> List[str]:
    return text.split()


def chunk_text(
    full_text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> List[Chunk]:
    """
    Split full_text into overlapping chunks of approximately
    chunk_size_tokens tokens each, with overlap_tokens of overlap between
    consecutive chunks. Uses a word-based sliding window scaled by the
    tokens-per-word ratio of the document, which is fast and avoids
    re-tokenizing every window with tiktoken.
    """
    words = _split_into_words(full_text)
    if not words:
        return []

    total_tokens = count_tokens(full_text)
    if total_tokens == 0:
        return []

    words_per_token = len(words) / max(total_tokens, 1)
    window_words = max(1, int(chunk_size_tokens * words_per_token))
    overlap_words = max(0, int(overlap_tokens * words_per_token))
    step = max(1, window_words - overlap_words)

    chunks: List[Chunk] = []
    idx = 0
    chunk_index = 0
    while idx < len(words):
        window = words[idx: idx + window_words]
        if not window:
            break
        chunk_text_value = " ".join(window)
        chunks.append(Chunk(text=chunk_text_value, chunk_index=chunk_index, start_word_index=idx))
        chunk_index += 1
        if idx + window_words >= len(words):
            break
        idx += step

    return chunks
