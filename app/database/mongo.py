"""
MongoDB connection management and CRUD helpers using Motor (async driver).

Collections:
  - papers: cached AI-generated results per uploaded paper (see spec: only
    paper_name, sha256, summary, structured_information, research_gap,
    timestamp are stored — never embeddings).
  - analytics: token/latency/cost events for GET /analytics.
  - chat_sessions: persistent per-paper conversation history for the
    ChatGPT-style /chat endpoint (session_id, paper_id, messages[], created_at,
    updated_at). Never stores embeddings/chunks either — those stay in FAISS.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongo() -> None:
    global _client, _db
    _client = AsyncIOMotorClient(settings.MONGO_URI)
    _db = _client[settings.MONGO_DB_NAME]
    # Fail fast if Mongo is unreachable
    await _client.admin.command("ping")
    await _db.papers.create_index("sha256", unique=True)
    await _db.analytics.create_index("timestamp")
    await _db.chat_sessions.create_index("session_id", unique=True)
    await _db.chat_sessions.create_index("paper_id")
    logger.info("Connected to MongoDB at %s (db=%s)", settings.MONGO_URI, settings.MONGO_DB_NAME)


async def close_mongo_connection() -> None:
    global _client
    if _client is not None:
        _client.close()
        logger.info("MongoDB connection closed")


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB has not been initialized. Call connect_to_mongo() first.")
    return _db


# ---------------------------------------------------------------------------
# Papers collection
# ---------------------------------------------------------------------------
async def get_paper_by_hash(sha256: str) -> Optional[Dict[str, Any]]:
    return await get_db().papers.find_one({"sha256": sha256}, {"_id": 0})


async def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    # paper_id IS the sha256 hash in this system
    return await get_paper_by_hash(paper_id)


async def insert_paper(document: Dict[str, Any]) -> None:
    document = dict(document)
    document.setdefault("timestamp", datetime.now(timezone.utc))
    await get_db().papers.insert_one(document)


async def get_papers_by_ids(paper_ids: List[str]) -> List[Dict[str, Any]]:
    cursor = get_db().papers.find({"sha256": {"$in": paper_ids}}, {"_id": 0})
    return [doc async for doc in cursor]


async def list_all_papers() -> List[Dict[str, Any]]:
    cursor = get_db().papers.find({}, {"_id": 0})
    return [doc async for doc in cursor]


# ---------------------------------------------------------------------------
# Analytics collection
# ---------------------------------------------------------------------------
async def insert_analytics_event(event: Dict[str, Any]) -> None:
    event = dict(event)
    event.setdefault("timestamp", datetime.now(timezone.utc))
    await get_db().analytics.insert_one(event)


async def get_all_analytics_events() -> List[Dict[str, Any]]:
    cursor = get_db().analytics.find({}, {"_id": 0}).sort("timestamp", -1)
    return [doc async for doc in cursor]


# ---------------------------------------------------------------------------
# Chat sessions collection
# ---------------------------------------------------------------------------
async def get_chat_session(session_id: str) -> Optional[Dict[str, Any]]:
    return await get_db().chat_sessions.find_one({"session_id": session_id}, {"_id": 0})


async def create_chat_session(
    session_id: str,
    paper_id: Optional[str] = None,
    uploaded_paper_metadata: Optional[Dict[str, Any]] = None,
    session_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    if uploaded_paper_metadata:
        uploaded_paper_metadata = dict(uploaded_paper_metadata)
        uploaded_paper_metadata["paper_id"] = paper_id

    if session_metadata is None:
        title = "New Research Session"
        if uploaded_paper_metadata:
            title = uploaded_paper_metadata.get("paper_name", "Paper Analysis")
        session_metadata = {
            "title": title,
            "created_at": now,
            "updated_at": now
        }
    else:
        session_metadata = dict(session_metadata)
        session_metadata.setdefault("title", "New Research Session")
        session_metadata.setdefault("created_at", now)
        session_metadata.setdefault("updated_at", now)

    document = {
        "session_id": session_id,
        "paper_id": paper_id,
        "uploaded_paper_metadata": uploaded_paper_metadata,
        "messages": [],
        "recommended_papers": [],
        "comparison_results": [],
        "conversation_summary": None,
        "session_metadata": session_metadata,
        "created_at": now,
        "updated_at": now,
    }
    await get_db().chat_sessions.insert_one(dict(document))
    return document


async def delete_chat_session(session_id: str) -> bool:
    """Permanently deletes a chat session from MongoDB."""
    result = await get_db().chat_sessions.delete_one({"session_id": session_id})
    return result.deleted_count > 0


async def append_chat_messages(session_id: str, user_content: str, assistant_content: str) -> None:
    """Appends one user turn + one assistant turn to an existing session."""
    now = datetime.now(timezone.utc)
    await get_db().chat_sessions.update_one(
        {"session_id": session_id},
        {
            "$push": {
                "messages": {
                    "$each": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            },
            "$set": {
                "updated_at": now,
                "session_metadata.updated_at": now
            },
        },
    )


async def list_all_chat_sessions() -> List[Dict[str, Any]]:
    """Returns metadata of all chat sessions sorted by updated timestamp."""
    cursor = get_db().chat_sessions.find(
        {},
        {
            "_id": 0,
            "session_id": 1,
            "paper_id": 1,
            "session_metadata": 1,
            "updated_at": 1,
            "created_at": 1,
        }
    ).sort("updated_at", -1)
    return [doc async for doc in cursor]


async def save_recommended_papers(session_id: str, papers: List[Dict[str, Any]]) -> None:
    """Saves recommended scholarly papers list in the session."""
    now = datetime.now(timezone.utc)
    await get_db().chat_sessions.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "recommended_papers": papers,
                "updated_at": now,
                "session_metadata.updated_at": now
            }
        }
    )


async def save_comparison_result(session_id: str, result: Dict[str, Any]) -> None:
    """Appends a new comparison report to the session's comparisons list."""
    now = datetime.now(timezone.utc)
    await get_db().chat_sessions.update_one(
        {"session_id": session_id},
        {
            "$push": {"comparison_results": result},
            "$set": {
                "updated_at": now,
                "session_metadata.updated_at": now
            }
        }
    )


async def update_conversation_summary(session_id: str, summary: str) -> None:
    """Updates the conversation summary field in the session."""
    now = datetime.now(timezone.utc)
    await get_db().chat_sessions.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "conversation_summary": summary,
                "updated_at": now,
                "session_metadata.updated_at": now
            }
        }
    )


async def bind_paper_to_session(session_id: str, paper_id: str, paper_metadata: Dict[str, Any]) -> None:
    """Binds an uploaded paper to an existing chat session (capped at 3 papers)."""
    now = datetime.now(timezone.utc)
    title = paper_metadata.get("paper_name", "Paper Analysis")
    paper_metadata = dict(paper_metadata)
    paper_metadata["paper_id"] = paper_id
    
    # Fetch current session to check existing uploaded papers
    session = await get_db().chat_sessions.find_one({"session_id": session_id})
    uploaded_papers = []
    if session:
        uploaded_papers = session.get("uploaded_papers") or []
        if session.get("uploaded_paper_metadata") and not uploaded_papers:
            uploaded_papers = [session["uploaded_paper_metadata"]]
            
    # Check if already present, update it, else append with max limit of 3
    existing_idx = next((i for i, p in enumerate(uploaded_papers) if p.get("paper_id") == paper_id or p.get("sha256") == paper_id), None)
    if existing_idx is not None:
        uploaded_papers[existing_idx] = paper_metadata
    else:
        if len(uploaded_papers) >= 3:
            uploaded_papers.pop(0)
        uploaded_papers.append(paper_metadata)
        
    await get_db().chat_sessions.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "paper_id": paper_id,
                "uploaded_paper_metadata": paper_metadata,
                "uploaded_papers": uploaded_papers,
                "session_metadata.title": title,
                "updated_at": now,
                "session_metadata.updated_at": now
            }
        }
    )
