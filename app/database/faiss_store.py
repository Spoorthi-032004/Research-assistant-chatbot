"""
FAISS-backed vector store.

Architecture (updated):
  - ONE global FAISS index (IndexIDMap wrapping IndexFlatIP over
    L2-normalized vectors == cosine similarity) holds chunk vectors for
    every uploaded paper.
  - A sidecar metadata file (metadata.json) maps each FAISS vector_id to the
    paper/chunk it came from: paper_id, paper_name, page_number,
    chunk_index, chunk_text.
  - On search, FAISS returns vector_ids + similarity scores; the metadata
    mapping is used to recover the chunk text / paper info needed to build
    the LLM context. Results can optionally be filtered down to a single
    paper_id (used by /chat, which answers questions about one paper).

Embeddings and chunk text are never stored in MongoDB, only here on disk.
"""
import json
import os
import threading
from typing import Dict, List, Optional

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

_lock = threading.Lock()

_index: Optional[faiss.IndexIDMap] = None
_metadata: Dict[int, Dict] = {}
_next_vector_id: int = 0

_INDEX_FILENAME = "global.index"
_METADATA_FILENAME = "metadata.json"


def _index_path() -> str:
    return os.path.join(settings.FAISS_INDEX_DIR, _INDEX_FILENAME)


def _metadata_path() -> str:
    return os.path.join(settings.FAISS_INDEX_DIR, _METADATA_FILENAME)


def _ensure_dir() -> None:
    os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)


def _load_state_locked() -> None:
    """Load the global index + metadata from disk into memory. Caller must hold _lock."""
    global _index, _metadata, _next_vector_id
    _ensure_dir()

    if os.path.exists(_index_path()):
        loaded = faiss.read_index(_index_path())
        _index = loaded if isinstance(loaded, faiss.IndexIDMap) else faiss.IndexIDMap(loaded)
    else:
        _index = faiss.IndexIDMap(faiss.IndexFlatIP(settings.FAISS_DIM))

    if os.path.exists(_metadata_path()):
        with open(_metadata_path(), "r", encoding="utf-8") as f:
            raw = json.load(f)
        _metadata = {int(k): v for k, v in raw.items()}
        _next_vector_id = (max(_metadata.keys()) + 1) if _metadata else 0
    else:
        _metadata = {}
        _next_vector_id = 0


def _ensure_loaded() -> None:
    if _index is None:
        with _lock:
            if _index is None:
                _load_state_locked()


def _persist_locked() -> None:
    """Write the in-memory index + metadata to disk. Caller must hold _lock."""
    faiss.write_index(_index, _index_path())
    with open(_metadata_path(), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in _metadata.items()}, f)


def add_vectors(paper_id: str, paper_name: str, vectors: np.ndarray, chunk_records: List[Dict]) -> List[int]:
    """
    Add one paper's chunk vectors to the global index.

    vectors: float32 array of shape (n_chunks, dim), NOT yet normalized.
    chunk_records: list of dicts aligned with vectors' rows, each with keys
        "page_number", "chunk_index", "chunk_text".

    Returns the list of vector_ids assigned (in row order).
    """
    _ensure_loaded()
    if vectors.shape[0] != len(chunk_records):
        raise ValueError("vectors and chunk_records length mismatch")
    if vectors.shape[0] == 0:
        return []

    normalized = vectors.astype("float32").copy()
    faiss.normalize_L2(normalized)

    with _lock:
        global _next_vector_id
        ids = np.arange(_next_vector_id, _next_vector_id + vectors.shape[0]).astype("int64")
        _index.add_with_ids(normalized, ids)

        for vector_id, record in zip(ids.tolist(), chunk_records):
            _metadata[vector_id] = {
                "vector_id": vector_id,
                "paper_id": paper_id,
                "paper_name": paper_name,
                "page_number": record.get("page_number"),
                "chunk_index": record.get("chunk_index"),
                "chunk_text": record.get("chunk_text"),
            }

        _next_vector_id += vectors.shape[0]
        _persist_locked()

    logger.info(
        "Added %d vectors for paper_id=%s (%s) to global FAISS index (total=%d)",
        vectors.shape[0], paper_id, paper_name, _index.ntotal,
    )
    return ids.tolist()


def search(
    query_vector: np.ndarray,
    top_k: int,
    paper_id: Optional[str] = None,
    paper_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Search the global index. If paper_id (or paper_ids list) is given, results are
    filtered to those papers only (chunks from other papers are excluded).

    Returns a list of metadata dicts (vector_id, paper_id, paper_name,
    page_number, chunk_index, chunk_text) each enriched with a 'score' key
    (cosine similarity, higher is better).
    """
    _ensure_loaded()
    if _index.ntotal == 0:
        return []

    # Consolidate filtering targets into a set of allowed paper_ids
    allowed_paper_ids = None
    if paper_id is not None:
        allowed_paper_ids = {paper_id}
    elif paper_ids is not None:
        allowed_paper_ids = set(paper_ids)

    query = query_vector.astype("float32").reshape(1, -1).copy()
    faiss.normalize_L2(query)

    # When filtering to specific papers, over-fetch from the global index since it
    # mixes chunks from every paper, then filter down to top_k matches.
    fetch_k = top_k if allowed_paper_ids is None else min(_index.ntotal, max(top_k * 10, 50))
    scores, ids = _index.search(query, fetch_k)

    results: List[Dict] = []
    for score, vector_id in zip(scores[0], ids[0]):
        if vector_id == -1:
            continue
        record = _metadata.get(int(vector_id))
        if record is None:
            continue
        if allowed_paper_ids is not None and record["paper_id"] not in allowed_paper_ids:
            continue
        item = dict(record)
        item["score"] = float(score)
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def paper_has_vectors(paper_id: str) -> bool:
    """True if at least one chunk for this paper is present in the global index."""
    _ensure_loaded()
    return any(record["paper_id"] == paper_id for record in _metadata.values())
