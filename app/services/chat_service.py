"""
Manual RAG pipeline for POST /chat, with ChatGPT-style session memory:

  load/create session -> application guardrails -> input guardrails -> embed query -> FAISS top-k ->
  retrieval validation -> build prompt (system -> trimmed history -> RAG context -> question) ->
  Groq -> output guardrails -> persist turn -> update session summary -> answer + citations + session_id
"""
import uuid
from typing import List, Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.database import mongo, faiss_store
from app.models.schemas import ChatResponse, CitationChunk
from app.services import embedding_service, llm_service, analytics_service
from app.guardrails.input_guard import run_input_guardrails, GuardrailBlockedException
from app.guardrails.output_guard import run_output_guardrails
from app.guardrails import application_guard

settings = get_settings()
logger = get_logger(__name__)


class ChatError(Exception):
    """Maps to 404 in the API layer (paper/session not found)."""


class SessionMismatchError(Exception):
    """Maps to 400 in the API layer (session_id exists but belongs to a different paper)."""


_RAG_SYSTEM_PROMPT = """You are a research assistant having an ongoing conversation with a user \
about a specific uploaded paper. Use ONLY the paper excerpts provided for the CURRENT question to \
answer it. If the excerpts do not contain enough information to answer confidently, say so \
explicitly rather than guessing. You may use the prior conversation turns to resolve references \
(e.g. "they", "it", "that dataset") to what was discussed earlier, but ground every factual claim \
in the excerpts. Keep answers concise and reference excerpt numbers inline (e.g. "(Excerpt 2)") \
where relevant."""

_GENERAL_SYSTEM_PROMPT = """You are a strict academic research assistant. If the user's query is \
unrelated to scientific literature, academic concepts, research methodologies, or general science/engineering, \
you MUST refuse to answer. Say: "I am designed specifically for research paper analysis, discovery, \
and literature review. Please ask a research-related question or upload a paper." Do not provide general \
knowledge, personal details, or trivia about individuals, pop culture, or unrelated facts."""


def _build_rag_messages(
    question: str,
    history: List[dict],
    retrieved_chunks: List[dict],
    conversation_summary: Optional[str] = None,
    is_general: bool = False,
    author_context: str = ""
) -> List[dict]:
    """
    ChatGPT-style prompt assembly:
        System Prompt -> Conversation History -> Retrieved RAG Context -> Current Question
    `history` is already trimmed.
    """
    if is_general:
        system_prompt = _GENERAL_SYSTEM_PROMPT
        current_turn_content = question
    else:
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            paper_name = chunk.get("paper_name", "Unknown Paper")
            page_num = chunk.get("page_number")
            page_info = f", Page {page_num}" if page_num else ""
            context_blocks.append(f"[Excerpt {i}] (From Paper: '{paper_name}'{page_info}): {chunk['chunk_text']}")
        context_text = "\n\n".join(context_blocks)
        current_turn_content = f"Paper excerpts:\n\n{context_text}\n\nQuestion: {question}"
        system_prompt = _RAG_SYSTEM_PROMPT

    if author_context:
        system_prompt += f"\n\n{author_context}"

    if conversation_summary:
        system_prompt += f"\n\nHere is a summary of the prior conversation topics and context: {conversation_summary}"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": current_turn_content})
    return messages


def _trim_history(messages: List[dict]) -> List[dict]:
    """Keep only the last MAX_CHAT_HISTORY_EXCHANGES user+assistant pairs."""
    max_messages = settings.MAX_CHAT_HISTORY_EXCHANGES * 2
    return messages[-max_messages:] if len(messages) > max_messages else messages


def _get_effective_history(messages: List[dict], has_summary: bool) -> List[dict]:
    """
    If conversation_summary is maintained, send only the summary plus the recent messages
    (last 2 turns, which is 4 messages) to the LLM instead of the entire history.
    """
    if has_summary:
        return messages[-4:] if len(messages) > 4 else messages
    return _trim_history(messages)


async def _load_or_create_session(session_id: Optional[str], paper_id: Optional[str]) -> tuple[str, List[dict], bool]:
    """
    Returns (session_id, history, is_new_session).
    """
    if session_id:
        logger.info("Loading existing chat session session_id=%s", session_id)
        session = await mongo.get_chat_session(session_id)
        if session is None:
            raise ChatError(f"No chat session found with session_id={session_id}.")
        if paper_id and session.get("paper_id") and session["paper_id"] != paper_id:
            raise SessionMismatchError(
                f"session_id={session_id} belongs to a different paper_id "
                f"({session['paper_id']}), not {paper_id}."
            )
        # Choose effective history based on presence of summary
        has_summary = bool(session.get("conversation_summary"))
        history = _get_effective_history(session.get("messages", []), has_summary)
        return session_id, history, False

    new_session_id = uuid.uuid4().hex
    logger.info("Creating new chat session session_id=%s for paper_id=%s", new_session_id, paper_id)

    uploaded_paper_metadata = None
    if paper_id:
        paper = await mongo.get_paper_by_id(paper_id)
        if paper:
            uploaded_paper_metadata = paper

    await mongo.create_chat_session(new_session_id, paper_id, uploaded_paper_metadata=uploaded_paper_metadata)
    return new_session_id, [], True


async def _maybe_update_summary(session_id: str, messages: List[dict], current_summary: Optional[str]) -> Optional[str]:
    """
    If the conversation is getting long, generate/update a conversation summary.
    """
    # Trigger summarization only when we have more than 4 messages (2 turns)
    if len(messages) <= 4:
        return current_summary

    logger.info("Conversation has %d messages, updating summary...", len(messages))
    history_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
    prompt = (
        "You are an assistant summarizing a research chat history. "
        "Create a concise 2-3 sentence summary summarizing the main topics, questions, "
        "and findings discussed in this conversation so far. Be objective, direct, and factual. "
        "Do not include personal opinions.\n\n"
        f"Prior Summary: {current_summary or 'None'}\n\n"
        f"New Messages:\n{history_text}\n\n"
        "Summary:"
    )

    summary_messages = [{"role": "user", "content": prompt}]
    try:
        result = await llm_service.generate(summary_messages, max_tokens=150, temperature=0.3)
        summary_content = result.content.strip()
        await mongo.update_conversation_summary(session_id, summary_content)
        return summary_content
    except Exception as exc:
        logger.exception("Failed to update conversation summary: %s", exc)
        return current_summary


def _get_session_authors(session: Optional[dict]) -> List[dict]:
    """
    Extracts author names and paper details from the session's papers.
    Returns a list of dicts: [{"name": "Saanchi", "title": "...", "type": "uploaded"}]
    """
    authors_info = []
    if not session:
        return authors_info

    # 1. Check uploaded papers list
    for paper in session.get("uploaded_papers", []):
        title = paper.get("structured_information", {}).get("title") or paper.get("paper_name") or "Uploaded Paper"
        authors_data = paper.get("structured_information", {}).get("authors") or []
        if isinstance(authors_data, str):
            names = [n.strip() for n in authors_data.split(",") if n.strip()]
        elif isinstance(authors_data, list):
            names = []
            for a in authors_data:
                if isinstance(a, str):
                    names.append(a.strip())
                elif isinstance(a, dict) and a.get("name"):
                    names.append(a["name"].strip())
        else:
            names = []
        
        for name in names:
            if name:
                authors_info.append({"name": name, "title": title, "type": "uploaded"})

    # 2. Also check single active paper
    active_meta = session.get("uploaded_paper_metadata")
    if active_meta:
        title = active_meta.get("structured_information", {}).get("title") or active_meta.get("paper_name") or "Active Paper"
        authors_data = active_meta.get("structured_information", {}).get("authors") or []
        if isinstance(authors_data, str):
            names = [n.strip() for n in authors_data.split(",") if n.strip()]
        elif isinstance(authors_data, list):
            names = []
            for a in authors_data:
                if isinstance(a, str):
                    names.append(a.strip())
                elif isinstance(a, dict) and a.get("name"):
                    names.append(a["name"].strip())
        else:
            names = []
        
        for name in names:
            if name and not any(a["name"] == name and a["title"] == title for a in authors_info):
                authors_info.append({"name": name, "title": title, "type": "uploaded"})

    # 3. Check recommended papers list
    for paper in session.get("recommended_papers", []):
        title = paper.get("title") or "Recommended Paper"
        authors_data = paper.get("authors") or []
        if isinstance(authors_data, str):
            names = [n.strip() for n in authors_data.split(",") if n.strip()]
        elif isinstance(authors_data, list):
            names = [a.strip() for a in authors_data if isinstance(a, str) and a.strip()]
        else:
            names = []
            
        for name in names:
            if name and not any(a["name"] == name and a["title"] == title for a in authors_info):
                authors_info.append({"name": name, "title": title, "type": "recommended"})

    return authors_info


async def answer_question(
    paper_id: Optional[str],
    question: str,
    top_k: Optional[int] = None,
    session_id: Optional[str] = None,
) -> ChatResponse:
    logger.info("Received question for paper_id=%s (session_id=%s)", paper_id or "<none>", session_id or "<new>")

    # Initialize or load session
    session_id, history, _ = await _load_or_create_session(session_id, paper_id)
    session = await mongo.get_chat_session(session_id)

    # 1. Run Application Guardrails
    guard_warning = application_guard.run_application_guardrails(question, session)
    if guard_warning:
        logger.info("Blocked by Application Guardrails: %s", guard_warning)
        # Store in conversation history
        await mongo.append_chat_messages(session_id, question, guard_warning)
        return ChatResponse(
            session_id=session_id,
            answer=guard_warning,
            citations=[],
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0
        )

    # 2. Run existing Input guardrails (Regex, Presidio, NeMo Guardrails)
    logger.info("Running input guardrails (session_id=%s)", session_id)
    try:
        guarded = await run_input_guardrails(question)
    except GuardrailBlockedException as exc:
        logger.info("Blocked by Input Guardrails (%s): %s", exc.stage, exc.reasons)
        if exc.stage == "regex_validation":
            if "Harmful content detected" in exc.reasons:
                friendly_warning = "I can't assist with requests involving harmful or illegal activities."
            else:
                friendly_warning = "I can't disclose or override my internal instructions, implementation, or system configuration. I'm designed to assist with research paper analysis, research paper discovery, paper comparison, and literature review tasks."
        elif exc.stage == "presidio_pii_detection":
            friendly_warning = (
                "For privacy and security, I cannot process queries containing personally identifiable "
                "information (PII) such as phone numbers, email addresses, or credit cards. "
                "Please rephrase your query without these details."
            )
        else:
            friendly_warning = (
                "This assistant is designed specifically for research paper analysis, discovery, "
                "and literature review. I can't help with requests outside this scope."
            )
            
        await mongo.append_chat_messages(session_id, question, friendly_warning)
        return ChatResponse(
            session_id=session_id,
            answer=friendly_warning,
            citations=[],
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0
        )

    uploaded_papers = session.get("uploaded_papers") or []
    if not uploaded_papers and session.get("uploaded_paper_metadata"):
        uploaded_papers = [session["uploaded_paper_metadata"]]
    
    paper_ids = [p.get("paper_id") or p.get("sha256") for p in uploaded_papers if p.get("paper_id") or p.get("sha256")]
    # Backwards compatibility: add paper_id from param or session.paper_id if not already there
    extra_id = paper_id or (session.get("paper_id") if session else None)
    if extra_id and extra_id not in paper_ids:
        paper_ids.append(extra_id)

    retrieved = []

    if paper_ids:
        # Check vectors exist for all target papers (log warnings if missing)
        for pid in paper_ids:
            if not faiss_store.paper_has_vectors(pid):
                logger.warning("No vector index found for paper_id=%s, skipping verification.", pid)

        # Generate query embedding
        logger.info("Generating query embedding (session_id=%s)", session_id)
        query_vector = embedding_service.embed_query(guarded.sanitized_text)
        k = top_k or settings.TOP_K_RETRIEVAL

        logger.info("Searching FAISS for session papers %s (session_id=%s, top_k=%d)", paper_ids, session_id, k)
        raw_retrieved = faiss_store.search(query_vector, k, paper_ids=paper_ids)

        # 3. Retrieval Validation: Validate chunks have a relevance score >= 0.15
        relevance_threshold = 0.15
        retrieved = [chunk for chunk in raw_retrieved if chunk.get("score", 0) >= relevance_threshold]

        if not retrieved:
            logger.info("Retrieval validation failed -- falling back to using cached paper summaries/profiles as context.")
            for fallback_idx, p in enumerate(uploaded_papers, start=1):
                p_name = p.get("paper_name", "Unknown Paper")
                p_summary = p.get("summary") or ""
                p_gap = p.get("research_gap") or ""
                p_struct = p.get("structured_information") or {}
                
                fallback_chunk_text = (
                    f"Paper Title/Name: {p_name}\n"
                    f"Summary: {p_summary}\n"
                    f"Problem Statement: {p_struct.get('problem_statement') or 'N/A'}\n"
                    f"Methodology: {p_struct.get('methodology') or 'N/A'}\n"
                    f"Key Results: {p_struct.get('results') or 'N/A'}\n"
                    f"Research Gap: {p_gap}\n"
                )
                
                retrieved.append({
                    "chunk_index": fallback_idx,
                    "paper_id": p.get("paper_id") or p.get("sha256"),
                    "paper_name": p_name,
                    "page_number": None,
                    "chunk_text": fallback_chunk_text,
                    "score": 1.0
                })

        logger.info("Retrieved %d valid chunks after validation (session_id=%s)", len(retrieved), session_id)

    # 4. Build prompt: system (with summary if exists) -> history -> RAG context -> question
    conversation_summary = session.get("conversation_summary") if session else None
    is_general = not bool(paper_ids)
    
    # Check if any authors are mentioned in the question
    matched_authors = []
    session_authors = _get_session_authors(session)
    for auth in session_authors:
        if auth["name"].lower() in guarded.sanitized_text.lower() or any(part.lower() in guarded.sanitized_text.lower() for part in auth["name"].split() if len(part) > 2):
            matched_authors.append(auth)
            
    author_context = ""
    if matched_authors:
        context_lines = []
        for auth in matched_authors:
            context_lines.append(f"- '{auth['name']}' is an author of the {auth['type']} research paper titled '{auth['title']}' in the current session.")
        author_context = "For your context, the following author matches were found in this session's papers:\n" + "\n".join(context_lines)

    messages = _build_rag_messages(guarded.sanitized_text, history, retrieved, conversation_summary, is_general=is_general, author_context=author_context)

    logger.info("Calling Groq (session_id=%s)", session_id)
    llm_result = await llm_service.generate(messages, max_tokens=settings.LLM_MAX_TOKENS)
    await analytics_service.record_llm_usage("/chat", llm_result)

    # 5. Output guardrails: NeMo -> Presidio -> Regex
    logger.info("Running output guardrails (session_id=%s)", session_id)
    try:
        output_guarded = await run_output_guardrails(llm_result.content)
    except GuardrailBlockedException as exc:
        logger.info("Blocked by Output Guardrails (%s): %s", exc.stage, exc.reasons)
        friendly_warning = (
            "This assistant is designed specifically for research paper analysis, discovery, "
            "and literature review. I can't help with requests outside this scope."
        )
        await mongo.append_chat_messages(session_id, guarded.sanitized_text, friendly_warning)
        return ChatResponse(
            session_id=session_id,
            answer=friendly_warning,
            citations=[],
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens,
            latency_ms=llm_result.latency_ms
        )

    # 6. Save turn & update conversation summary
    logger.info("Saving conversation turn (session_id=%s)", session_id)
    await mongo.append_chat_messages(session_id, guarded.sanitized_text, output_guarded.sanitized_text)

    # Check and update conversation summary in background/sequential flow
    updated_session = await mongo.get_chat_session(session_id)
    if updated_session:
        all_msgs = updated_session.get("messages", [])
        current_sum = updated_session.get("conversation_summary")
        await _maybe_update_summary(session_id, all_msgs, current_sum)

    citations = [
        CitationChunk(
            chunk_index=chunk["chunk_index"],
            page_number=chunk.get("page_number"),
            snippet=(chunk["chunk_text"][:280] + ("..." if len(chunk["chunk_text"]) > 280 else "")),
            score=round(chunk["score"], 4),
        )
        for chunk in retrieved
    ]

    logger.info("Returning response (session_id=%s)", session_id)
    return ChatResponse(
        session_id=session_id,
        answer=output_guarded.sanitized_text,
        citations=citations,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        latency_ms=llm_result.latency_ms,
    )
