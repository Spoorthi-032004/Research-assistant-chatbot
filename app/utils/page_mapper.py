"""
Approximate word-index -> page-number mapping.

The chunker operates on the full document text (all pages joined), so a
chunk's exact character offset within its source page is not tracked. This
module gives an approximate page number for a chunk by counting words per
page and locating which page's word range a chunk's starting word falls
into. It is approximate because chunk windows can straddle a page boundary;
the chunk is attributed to the page its FIRST word belongs to.
"""
import bisect
from typing import List


def compute_page_word_boundaries(pages: List[str]) -> List[int]:
    """
    Returns a list where entry i is the cumulative word count of all pages
    before page i (0-indexed). E.g. for page word counts [10, 20, 5]:
    -> [0, 10, 30]
    """
    boundaries = []
    cumulative = 0
    for page_text in pages:
        boundaries.append(cumulative)
        cumulative += len(page_text.split())
    return boundaries


def word_index_to_page(word_index: int, page_word_boundaries: List[int]) -> int:
    """
    Maps a word index (0-indexed, into the full joined document text) to a
    1-indexed page number, using the cumulative boundaries produced by
    compute_page_word_boundaries().
    """
    if not page_word_boundaries:
        return 1
    page = bisect.bisect_right(page_word_boundaries, word_index)
    return max(1, min(page, len(page_word_boundaries)))
