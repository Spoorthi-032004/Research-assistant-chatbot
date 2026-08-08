"""
Embedding generation using sentence-transformers/all-MiniLM-L6-v2, run on CPU.

The model is loaded once (lazily, thread-safe) and reused across requests.
"""
import threading
from typing import List

import numpy as np

from app.core.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

_model = None
_init_lock = threading.Lock()


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _init_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s (CPU)...", settings.EMBEDDING_MODEL_NAME)
            _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")
            logger.info("Embedding model loaded")
    return _model


def preload_model() -> None:
    """
    Eagerly load the embedding model. Intended to be called once during app
    startup (see app.main lifespan) so the first upload/chat request doesn't
    pay the model-load cost, and so the same in-memory model instance is
    reused for every request for the lifetime of the process.
    """
    _get_model()


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns a float32 numpy array of shape (len(texts), dim).

    Texts are embedded in batches (see settings.EMBEDDING_BATCH_SIZE, default
    32) rather than one at a time -- sentence-transformers' encode() batches
    internally, which is significantly faster than calling embed once per
    chunk, especially on CPU.
    """
    if not texts:
        return np.zeros((0, settings.FAISS_DIM), dtype="float32")
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings.astype("float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
