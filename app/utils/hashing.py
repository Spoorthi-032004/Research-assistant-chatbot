"""
Hashing helpers used for duplicate detection.
"""
import hashlib


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of raw bytes."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def sha256_file(path: str, chunk_size: int = 8192) -> str:
    """Return the SHA-256 hex digest of a file on disk, streamed in chunks."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
