"""
POST /chat -- RAG chat with an uploaded paper, with ChatGPT-style session
memory. Omit session_id to start a new conversation (the response returns
the new session_id); pass it back on follow-up questions to continue.
"""
from typing import List, Optional
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.core.logging_config import get_logger
from app.models.schemas import (
    ChatRequest, ChatResponse, ErrorResponse,
    SessionListResponse, SessionListItem, SessionDetailResponse,
    CreateSessionRequest, SearchResponse, SearchResultItem, AppendMessagesRequest, MessageItem,
    SaveRecommendationsRequest
)
from app.services import chat_service, search_service
from app.database import mongo
from app.guardrails.input_guard import GuardrailBlockedException

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    summary="Ask a question about an uploaded paper (RAG, session-based conversational memory)",
)
async def chat_with_paper(request: ChatRequest) -> ChatResponse:
    try:
        # Note: request.paper_id is passed. Can be empty/None if user passes it as optional
        return await chat_service.answer_question(
            request.paper_id, request.question, request.top_k, request.session_id
        )
    except GuardrailBlockedException as exc:
        raise HTTPException(status_code=403, detail=f"Request blocked by guardrails ({exc.stage}): {', '.join(exc.reasons)}") from exc
    except chat_service.SessionMismatchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except chat_service.ChatError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during chat")
        raise HTTPException(status_code=500, detail=f"Internal error during chat: {exc}") from exc


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="Get all research sessions sorted by last activity time"
)
async def get_all_sessions() -> SessionListResponse:
    try:
        sessions = await mongo.list_all_chat_sessions()
        items = []
        for s in sessions:
            meta = s.get("session_metadata") or {}
            items.append(
                SessionListItem(
                    session_id=s["session_id"],
                    paper_id=s.get("paper_id"),
                    title=meta.get("title", "New Research Session"),
                    created_at=s.get("created_at") or meta.get("created_at"),
                    updated_at=s.get("updated_at") or meta.get("updated_at")
                )
            )
        return SessionListResponse(sessions=items)
    except Exception as exc:
        logger.exception("Unexpected error listing sessions")
        raise HTTPException(status_code=500, detail=f"Internal error listing sessions: {exc}") from exc


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get the full conversation and context of a specific session"
)
async def get_session_detail(session_id: str) -> SessionDetailResponse:
    try:
        session = await mongo.get_chat_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session with id={session_id} not found.")
        
        return SessionDetailResponse(
            session_id=session["session_id"],
            paper_id=session.get("paper_id"),
            uploaded_paper_metadata=session.get("uploaded_paper_metadata"),
            uploaded_papers=session.get("uploaded_papers", []),
            messages=session.get("messages", []),
            recommended_papers=session.get("recommended_papers", []),
            comparison_results=session.get("comparison_results", []),
            conversation_summary=session.get("conversation_summary"),
            created_at=session["created_at"],
            updated_at=session["updated_at"]
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error getting session details")
        raise HTTPException(status_code=500, detail=f"Internal error getting session: {exc}") from exc


@router.delete(
    "/sessions/{session_id}",
    responses={404: {"model": ErrorResponse}},
    summary="Permanently delete a specific research session"
)
async def delete_session(session_id: str):
    try:
        deleted = await mongo.delete_chat_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Session with id={session_id} not found.")
        return {"status": "success", "message": f"Session {session_id} successfully deleted."}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error deleting session")
        raise HTTPException(status_code=500, detail=f"Internal error deleting session: {exc}") from exc


@router.post(
    "/sessions",
    response_model=SessionDetailResponse,
    summary="Create a new research session"
)
async def create_session(request: CreateSessionRequest) -> SessionDetailResponse:
    try:
        session_id = uuid.uuid4().hex
        uploaded_paper_metadata = None
        if request.paper_id:
            paper = await mongo.get_paper_by_id(request.paper_id)
            if paper:
                uploaded_paper_metadata = paper
        
        doc = await mongo.create_chat_session(
            session_id=session_id,
            paper_id=request.paper_id,
            uploaded_paper_metadata=uploaded_paper_metadata
        )
        return SessionDetailResponse(
            session_id=doc["session_id"],
            paper_id=doc.get("paper_id"),
            uploaded_paper_metadata=doc.get("uploaded_paper_metadata"),
            messages=[],
            recommended_papers=[],
            comparison_results=[],
            conversation_summary=None,
            created_at=doc["created_at"],
            updated_at=doc["updated_at"]
        )
    except Exception as exc:
        logger.exception("Unexpected error creating session")
        raise HTTPException(status_code=500, detail=f"Internal error creating session: {exc}") from exc


@router.post(
    "/sessions/{session_id}/recommendations",
    response_model=List[SearchResultItem],
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Discover recent related papers for the current uploaded paper or save custom recommendations"
)
async def discover_related_papers(session_id: str, request: Optional[SaveRecommendationsRequest] = None) -> List[SearchResultItem]:
    try:
        session = await mongo.get_chat_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session with id={session_id} not found.")
        
        # If papers are passed in the request body, save them directly to session
        if request is not None and request.papers:
            serialized_results = [r.model_dump() for r in request.papers]
            await mongo.save_recommended_papers(session_id, serialized_results)
            return request.papers

        paper_id = session.get("paper_id")
        if not paper_id:
            raise HTTPException(status_code=400, detail="Please upload a research paper first to find recommendations.")
        
        # Determine query topic from paper metadata title
        paper_metadata = session.get("uploaded_paper_metadata")
        if paper_metadata:
            struct_info = paper_metadata.get("structured_information") or {}
            title = struct_info.get("title") or paper_metadata.get("paper_name")
        else:
            paper = await mongo.get_paper_by_id(paper_id)
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found in database.")
            struct_info = paper.get("structured_information") or {}
            title = struct_info.get("title") or paper.get("paper_name")
            
        logger.info("Discovering papers related to: %s", title)
        # Search scholarly APIs
        results = await search_service.search_all_sources(title, limit_per_source=3)
        
        # Save to database
        serialized_results = [r.model_dump() for r in results]
        await mongo.save_recommended_papers(session_id, serialized_results)
        
        return results
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error finding recommendations")
        raise HTTPException(status_code=500, detail=f"Internal error generating recommendations: {exc}") from exc


@router.post(
    "/sessions/{session_id}/paper/{paper_id}",
    response_model=SessionDetailResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Bind a paper to an existing session"
)
async def bind_paper(session_id: str, paper_id: str) -> SessionDetailResponse:
    try:
        paper = await mongo.get_paper_by_id(paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found.")
        await mongo.bind_paper_to_session(session_id, paper_id, paper)
        
        session = await mongo.get_chat_session(session_id)
        return SessionDetailResponse(
            session_id=session["session_id"],
            paper_id=session.get("paper_id"),
            uploaded_paper_metadata=session.get("uploaded_paper_metadata"),
            uploaded_papers=session.get("uploaded_papers", []),
            messages=session.get("messages", []),
            recommended_papers=session.get("recommended_papers", []),
            comparison_results=session.get("comparison_results", []),
            conversation_summary=session.get("conversation_summary"),
            created_at=session["created_at"],
            updated_at=session["updated_at"]
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error binding paper")
        raise HTTPException(status_code=500, detail=f"Internal error binding paper: {exc}") from exc


@router.delete(
    "/sessions/{session_id}/paper/{paper_id}",
    response_model=SessionDetailResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Unbind a paper from the session"
)
async def unbind_paper(session_id: str, paper_id: str) -> SessionDetailResponse:
    try:
        session = await mongo.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        
        uploaded_papers = session.get("uploaded_papers") or []
        uploaded_papers = [p for p in uploaded_papers if p.get("paper_id") != paper_id and p.get("sha256") != paper_id]
        
        active_paper_id = session.get("paper_id")
        uploaded_paper_metadata = session.get("uploaded_paper_metadata")
        
        if active_paper_id == paper_id:
            if uploaded_papers:
                active_paper_id = uploaded_papers[0].get("paper_id") or uploaded_papers[0].get("sha256")
                uploaded_paper_metadata = uploaded_papers[0]
            else:
                active_paper_id = None
                uploaded_paper_metadata = None
                
        now = datetime.now(timezone.utc)
        await mongo.get_db().chat_sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "paper_id": active_paper_id,
                    "uploaded_paper_metadata": uploaded_paper_metadata,
                    "uploaded_papers": uploaded_papers,
                    "updated_at": now,
                    "session_metadata.updated_at": now
                }
            }
        )
        
        updated_session = await mongo.get_chat_session(session_id)
        return SessionDetailResponse(
            session_id=updated_session["session_id"],
            paper_id=updated_session.get("paper_id"),
            uploaded_paper_metadata=updated_session.get("uploaded_paper_metadata"),
            uploaded_papers=updated_session.get("uploaded_papers", []),
            messages=updated_session.get("messages", []),
            recommended_papers=updated_session.get("recommended_papers", []),
            comparison_results=updated_session.get("comparison_results", []),
            conversation_summary=updated_session.get("conversation_summary"),
            created_at=updated_session["created_at"],
            updated_at=updated_session["updated_at"]
        )
    except Exception as exc:
        logger.exception("Unexpected error unbinding paper")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")




@router.post(
    "/sessions/{session_id}/messages",
    response_model=SessionDetailResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Append manual messages to a session history"
)
async def append_manual_messages(session_id: str, request: AppendMessagesRequest) -> SessionDetailResponse:
    try:
        session = await mongo.get_chat_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        
        # Append each message
        for msg in request.messages:
            await mongo.get_db().chat_sessions.update_one(
                {"session_id": session_id},
                {
                    "$push": {
                        "messages": {
                            "$each": [msg.model_dump()]
                        }
                    },
                    "$set": {
                        "updated_at": datetime.now(timezone.utc),
                        "session_metadata.updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            
        session = await mongo.get_chat_session(session_id)
        return SessionDetailResponse(
            session_id=session["session_id"],
            paper_id=session.get("paper_id"),
            uploaded_paper_metadata=session.get("uploaded_paper_metadata"),
            messages=session.get("messages", []),
            recommended_papers=session.get("recommended_papers", []),
            comparison_results=session.get("comparison_results", []),
            conversation_summary=session.get("conversation_summary"),
            created_at=session["created_at"],
            updated_at=session["updated_at"]
        )
    except Exception as exc:
        logger.exception("Unexpected error appending messages")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
