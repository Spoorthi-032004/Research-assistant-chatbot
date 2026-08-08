import streamlit as st
import requests
import uuid
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

st.set_page_config(page_title="AI Research Assistant", page_icon="🧬", layout="wide")

BACKEND_URL = "http://127.0.0.1:8000"

# Premium purple and white theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap');

    .stApp {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Main container background and bottom padding for fixed input bar */
    .main .block-container {
        background-color: #FFFFFF !important;
        padding-top: 0.5rem !important;
        padding-bottom: 220px !important;
    }
    
    .main {
        overflow-y: auto !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] h3 {
        color: #1F2937 !important;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Global text visibility overrides (fixes faint/invisible text) */
    p, span, li, ul, ol, label {
        color: #1F2937 !important;
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: #7C3AED !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        transition: all 0.2s ease !important;
    }
    
    div.stButton > button:hover {
        border-color: #7C3AED !important;
        color: #7C3AED !important;
        background-color: #F5F3FF !important;
    }

    /* Primary button style overrides */
    div.stButton > button[kind="primary"] {
        background-color: #7C3AED !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    
    div.stButton > button[kind="primary"] * {
        color: #FFFFFF !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background-color: #6D28D9 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 6px -1px rgba(124, 58, 237, 0.2) !important;
    }
    
    /* Alert / Context Banner overrides */
    .stAlert {
        background-color: #F5F3FF !important;
        color: #6D28D9 !important;
        border: 1px solid #DDD6FE !important;
        border-radius: 8px !important;
    }
    
    .stAlert * {
        color: #6D28D9 !important;
    }
    
    /* Chat history scroll container */
    .stChatMessage {
        background-color: transparent !important;
        padding: 8px 0 !important;
        border: none !important;
    }
    
    /* Chat message text colors */
    .stChatMessage [data-testid="stMarkdownContainer"] p, 
    .stChatMessage [data-testid="stMarkdownContainer"] li,
    .stChatMessage [data-testid="stMarkdownContainer"] span {
        color: #1F2937 !important;
    }
    
    .stChatMessage a {
        color: #7C3AED !important;
        font-weight: 600 !important;
        text-decoration: underline !important;
    }
    
    /* User message bubble */
    div[data-testid="chatAvatarIcon-user"] {
        background-color: #F5F3FF !important;
    }
    
    .stChatMessage:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: #F5F3FF !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        border: 1px solid #E9D5FF !important;
    }
    
    /* Assistant message bubble */
    div[data-testid="chatAvatarIcon-assistant"] {
        background-color: #7C3AED !important;
    }
    
    /* Sticky Bottom Input Bar Layout CSS anchor hack - aligned next to the sidebar */
    div[data-testid="stVerticalBlock"]:has(div.bottom-anchor) {
        position: fixed !important;
        bottom: 0 !important;
        left: 21.5rem !important;
        right: 2rem !important;
        width: auto !important;
        background-color: #FFFFFF !important;
        padding: 15px 30px !important;
        border-top: 1px solid #E5E7EB !important;
        box-shadow: 0 -8px 24px rgba(0, 0, 0, 0.05) !important;
        z-index: 9999 !important;
        border-radius: 16px 16px 0 0 !important;
    }
    
    /* Paper metadata card */
    .paper-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    .paper-card * {
        color: #1F2937 !important;
    }
    
    .paper-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .paper-authors {
        font-size: 0.8rem;
        color: #6B7280 !important;
        margin-bottom: 8px;
    }
    
    .paper-abstract {
        font-size: 0.85rem;
        color: #4B5563 !important;
        line-height: 1.4;
        margin-bottom: 12px;
    }
    
    .paper-tag {
        display: inline-block;
        background-color: #F5F3FF !important;
        color: #7C3AED !important;
        border: 1px solid #DDD6FE !important;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-bottom: 6px;
        font-weight: 500;
    }
    
    .paper-tag a {
        color: #7C3AED !important;
        text-decoration: none;
    }
    
    /* Custom File Uploader */
    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 1px dashed #7C3AED !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    
    div[data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
    }

    /* Pinned Chat Input Container */
    .stChatInputContainer {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
    
    .stChatInputContainer textarea {
        color: #111827 !important;
        background-color: transparent !important;
    }
    
    /* Sidebar Session Items formatting */
    div.element-container:has(button[key^="session_btn_"]) button {
        text-align: left !important;
        justify-content: flex-start !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    /* Scrollbars - styled custom purple scrollbar for scrollable divs */
    ::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
        display: block !important;
    }
    ::-webkit-scrollbar-track {
        background: #F3F4F6 !important;
        border-radius: 4px !important;
    }
    ::-webkit-scrollbar-thumb {
        background: #7C3AED !important;
        border-radius: 4px !important;
        border: 2px solid #F3F4F6 !important;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #6D28D9 !important;
    }
    
    /* Target Streamlit's vertical block containers specifically for scrollbar style */
    div[data-testid="stVerticalBlock"]::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
        display: block !important;
    }
    div[data-testid="stVerticalBlock"]::-webkit-scrollbar-track {
        background: #F3F4F6 !important;
        border-radius: 4px !important;
    }
    div[data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
        background: #7C3AED !important;
        border-radius: 4px !important;
        border: 2px solid #F3F4F6 !important;
    }
    div[data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb:hover {
        background: #6D28D9 !important;
    }
    
    /* Force vertical scrollbar and bottom padding inside the chat container so it displays like a standard chat box */
    div[data-testid="stChatMessageContainer"] {
        overflow-y: scroll !important;
        padding-bottom: 150px !important;
    }

    /* Change header background from black to clean white */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #E5E7EB !important;
        color: #1F2937 !important;
    }

    /* Streamlit top decoration line */
    div[data-testid="stDecoration"] {
        background-image: none !important;
        background-color: #7C3AED !important;
        height: 3px !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session States
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "active_paper_metadata" not in st.session_state:
    st.session_state.active_paper_metadata = None
if "uploaded_papers" not in st.session_state:
    st.session_state.uploaded_papers = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recommended_papers" not in st.session_state:
    st.session_state.recommended_papers = []
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = None


def load_recent_sessions() -> List[Dict[str, Any]]:
    try:
        response = requests.get(f"{BACKEND_URL}/chat/sessions")
        if response.status_code == 200:
            return response.json().get("sessions", [])
    except Exception as exc:
        st.sidebar.warning("Could not connect to FastAPI backend server.")
    return []


def load_session(session_id: str):
    try:
        response = requests.get(f"{BACKEND_URL}/chat/sessions/{session_id}")
        if response.status_code == 200:
            data = response.json()
            st.session_state.current_session_id = data["session_id"]
            st.session_state.active_paper_metadata = data.get("uploaded_paper_metadata")
            st.session_state.uploaded_papers = data.get("uploaded_papers") or ([data["uploaded_paper_metadata"]] if data.get("uploaded_paper_metadata") else [])
            st.session_state.messages = data.get("messages", [])
            st.session_state.recommended_papers = data.get("recommended_papers", [])
            st.session_state.comparison_results = data.get("comparison_results", [])
            st.session_state.conversation_summary = data.get("conversation_summary")
            st.rerun()
    except Exception as exc:
        st.error(f"Error loading session: {exc}")


def create_new_session(paper_id: Optional[str] = None):
    try:
        response = requests.post(f"{BACKEND_URL}/chat/sessions", json={"paper_id": paper_id})
        if response.status_code == 200:
            data = response.json()
            st.session_state.current_session_id = data["session_id"]
            st.session_state.active_paper_metadata = data.get("uploaded_paper_metadata")
            st.session_state.messages = []
            st.session_state.recommended_papers = []
            st.session_state.comparison_results = []
            st.session_state.conversation_summary = None
            st.rerun()
    except Exception as exc:
        st.error(f"Error creating session: {exc}")


# -------------------------------------------------------------
# SIDEBAR: Research Sessions & Paper Upload
# -------------------------------------------------------------
# Custom sidebar logo/header
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding: 10px 0;">
    <div style="background-color: #EEF2FF; padding: 8px; border-radius: 10px; display: flex; align-items: center; justify-content: center; border: 1px solid #E0E7FF;">
        <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-brain"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M12 5v14"/><path d="M12 9h4a1 1 0 0 1 0 2h-4"/><path d="M12 14h-4a1 1 0 0 1 0-2h4"/></svg>
    </div>
    <div style="font-family: 'Outfit', sans-serif; font-size: 1.2rem; font-weight: 700; color: #4F46E5; line-height: 1.2;">
        AI Research<br/><span style="color: #6366f1; font-size: 1.15rem; font-weight: 600;">Assistant</span>
    </div>
</div>
""", unsafe_allow_html=True)

# New Chat Button
if st.sidebar.button("➕ New Chat", type="primary", use_container_width=True):
    create_new_session()

st.sidebar.markdown("---")

# Recent Chats Sidebar
st.sidebar.subheader("🕒 Recent Chats")
recent_sessions = load_recent_sessions()

if not recent_sessions:
    st.sidebar.info("No recent research chats found.")
else:
    for session in recent_sessions:
        # Check active session highlight
        is_active = session["session_id"] == st.session_state.current_session_id
        btn_label = f"💬 {session['title'][:25]}"
        btn_type = "primary" if is_active else "secondary"
            
        if st.sidebar.button(btn_label, key=f"session_btn_{session['session_id']}", type=btn_type, use_container_width=True):
            load_session(session["session_id"])

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Delete Current Chat", type="primary", use_container_width=True):
        session_id = st.session_state.current_session_id
        if session_id:
            try:
                resp = requests.delete(f"{BACKEND_URL}/chat/sessions/{session_id}")
                if resp.status_code == 200:
                    st.session_state.current_session_id = None
                    st.session_state.active_paper_metadata = None
                    st.session_state.messages = []
                    st.session_state.recommended_papers = []
                    st.session_state.comparison_results = []
                    st.session_state.conversation_summary = None
                    st.toast("Chat deleted permanently.")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to delete chat session.")
            except Exception as e:
                st.sidebar.error(f"Error deleting chat: {e}")

    # -------------------------------------------------------------
    # SIDEBAR: Upload Research Papers
    # -------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Upload Research Papers")
    uploaded_papers = st.session_state.get("uploaded_papers") or []
    if not uploaded_papers and st.session_state.active_paper_metadata:
        uploaded_papers = [st.session_state.active_paper_metadata]

    if len(uploaded_papers) < 3:
        uploaded_files = st.sidebar.file_uploader(
            "Upload PDF Paper(s) for analysis (Max 3)",
            type=["pdf"],
            accept_multiple_files=True,
            key="sidebar_file_uploader"
        )
        if uploaded_files:
            file_keys = [f"uploaded_{f.name}_{len(f.getvalue())}" for f in uploaded_files]
            new_files = [f for f, key in zip(uploaded_files, file_keys) if key not in st.session_state]
            
            if new_files:
                if len(uploaded_papers) + len(new_files) > 3:
                    st.sidebar.error("⚠️ Adding these files would exceed the limit of 3 PDF uploads per session.")
                else:
                    for key in file_keys:
                        st.session_state[key] = True
                        
                    # Ensure we have an active session
                    if not st.session_state.current_session_id:
                        st.session_state.current_session_id = str(uuid.uuid4().hex)
                        try:
                            requests.post(f"{BACKEND_URL}/chat/sessions", json={"session_id": st.session_state.current_session_id})
                        except Exception:
                            pass
                    
                    session_id = st.session_state.current_session_id
                    
                    with st.spinner("Processing PDF paper(s)..."):
                        try:
                            # Construct payload for multi-file upload
                            files_payload = []
                            for f in new_files:
                                files_payload.append(("files", (f.name, f.getvalue(), "application/pdf")))
                                
                            upload_response = requests.post(f"{BACKEND_URL}/upload", files=files_payload)
                            
                            if upload_response.status_code == 200:
                                upload_data = upload_response.json()
                                
                                # Bind each paper to the session
                                for paper_detail in upload_data.get("papers", []):
                                    paper_id = paper_detail["paper_id"]
                                    requests.post(f"{BACKEND_URL}/chat/sessions/{session_id}/paper/{paper_id}")
                                    st.toast(f"Paper loaded: {paper_detail['paper_name']}")
                                    
                                # Check if combined summary was generated (related papers)
                                combined_summary = upload_data.get("combined_summary")
                                if combined_summary:
                                    summary_content = f"### 📑 Combined Summary of Related Papers\n\n{combined_summary}"
                                    requests.post(f"{BACKEND_URL}/chat/sessions/{session_id}/messages", json={
                                        "messages": [
                                            {"role": "assistant", "content": summary_content}
                                        ]
                                    })
                                    
                                load_session(session_id)
                            else:
                                st.sidebar.error(f"Failed to process PDF: {upload_response.json().get('detail')}")
                        except Exception as e:
                            st.sidebar.error(f"Error processing PDF: {e}")
    else:
        st.sidebar.warning("⚠️ Maximum limit of 3 uploaded papers reached. Remove a paper first to upload another.")





# Ensure a session is initialized at startup
if not st.session_state.current_session_id:
    # If sessions list is not empty, load the most recent one. Otherwise create new.
    if recent_sessions:
        load_session(recent_sessions[0]["session_id"])
    else:
        create_new_session()


# Quick helper to identify paper search intent inline in chat
def detect_search_intent(query: str) -> bool:
    lowered = query.lower()
    search_indicators = [
        "find papers", "search for papers", "search papers", "discover papers", 
        "recommend papers", "find recent papers", "search for related papers", "scholarly search",
        "give me some research paper", "give me research papers", "suggest papers", "show me papers",
        "search for a paper", "find a paper", "give me some papers", "recommend a paper"
    ]
    return any(indicator in lowered for indicator in search_indicators) or (
        lowered.startswith("search ") or 
        lowered.startswith("find ") or 
        lowered.startswith("give me ") or
        lowered.startswith("recommend ") or
        lowered.startswith("show me ")
    )


def detect_compare_intent(query: str) -> bool:
    lowered = query.lower()
    compare_indicators = [
        "compare papers", "compare uploaded papers", "compare the papers", 
        "compare the uploaded papers", "run comparison", "compare them", "comparison report",
        "compare my uploaded papers"
    ]
    return any(indicator in lowered for indicator in compare_indicators)


# MAIN LAYOUT: Single Conversation Interface
# -------------------------------------------------------------

tab_chat, tab_rec = st.tabs(["💬 Research Chat", "📚 Recommended Papers"])

with tab_chat:
    # Active context check and banner
    uploaded_papers = st.session_state.get("uploaded_papers") or []
    if not uploaded_papers and st.session_state.active_paper_metadata:
        uploaded_papers = [st.session_state.active_paper_metadata]

    if uploaded_papers:
        paper_names = [p.get("paper_name", "Unknown Paper") for p in uploaded_papers]
        papers_str = ", ".join(f"**{name}**" for name in paper_names)
        st.info(f"📍 **Active Context**: Analyzing {len(uploaded_papers)} paper(s): {papers_str}")
    else:
        st.warning("📍 **Active Context**: General research chat. Upload a PDF in the sidebar or search to begin paper analysis.")

    # Create a container for messages so that new messages/errors are always rendered above the chat input
    chat_placeholder = st.container(height=580)

    # Displays previous messages
    with chat_placeholder:
        for index, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])
        # Leave some bottom space inside the scroll container so the question/answer is fully visible
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # STICKY BOTTOM INPUT BAR CONTAINER
    # -------------------------------------------------------------
    bottom_bar = st.container()
    
    with bottom_bar:
        # Hack to target this container in CSS and make it fixed bottom
        st.markdown('<div class="bottom-anchor"></div>', unsafe_allow_html=True)
        
        # 1. Horizontally list uploaded papers as cards/chips matching the 3rd snapshot
        uploaded_papers = st.session_state.get("uploaded_papers") or []
        if not uploaded_papers and st.session_state.active_paper_metadata:
            uploaded_papers = [st.session_state.active_paper_metadata]
            
        if uploaded_papers:
            # Render a horizontal layout of cards
            # We can create up to 3 columns to fit up to 3 papers horizontally!
            cols_chips = st.columns(3)
            for idx, paper in enumerate(uploaded_papers[:3]):
                paper_id = paper.get("paper_id") or paper.get("sha256")
                paper_name = paper.get("paper_name", "Unknown Paper")
                
                # Truncate long name to look elegant
                truncated_name = paper_name
                if len(truncated_name) > 30:
                    truncated_name = truncated_name[:27] + "..."
                    
                border_style = "border: 1px solid #7C3AED;"
                bg_style = "background-color: #F5F3FF;"
                
                with cols_chips[idx]:
                    # Horizontal chip: Red document icon, text, and close ✕ button
                    st.markdown(f"""
                    <div style="{border_style} {bg_style} padding: 8px 12px; border-radius: 12px; display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                        <div style="background-color: #EF4444; color: white; padding: 4px 6px; border-radius: 6px; font-weight: bold; font-size: 10px; line-height: 1;">PDF</div>
                        <div style="font-size: 11px; font-weight: 500; color: #1F2937; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 120px;" title="{paper_name}">{truncated_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Center the delete trigger
                    if st.button("✕ Delete", key=f"del_paper_{paper_id}", use_container_width=True):
                        requests.delete(f"{BACKEND_URL}/chat/sessions/{st.session_state.current_session_id}/paper/{paper_id}")
                        load_session(st.session_state.current_session_id)

        # 3. Chat Input
        user_query = st.chat_input("Ask anything...")

    # If the user submitted a query, update the session state and rerun immediately to show the user message
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.processing_query = user_query
        st.rerun()

    # Process the active query if present
    if st.session_state.get("processing_query"):
        query_to_process = st.session_state.processing_query
        # Clear the query flag immediately so we don't double-process on subsequent runs
        st.session_state.processing_query = None
        session_id = st.session_state.current_session_id
        
        with chat_placeholder:
            # Check if user asks to compare papers in session explicitly
            if detect_compare_intent(query_to_process):
                with st.chat_message("assistant"):
                    with st.spinner("Comparing uploaded papers..."):
                        try:
                            uploaded_papers = st.session_state.get("uploaded_papers") or []
                            if len(uploaded_papers) < 2:
                                ans = "Please upload at least 2 PDF papers before asking to compare them."
                                st.markdown(ans)
                                requests.post(f"{BACKEND_URL}/chat/sessions/{session_id}/messages", json={
                                    "messages": [
                                        {"role": "user", "content": query_to_process},
                                        {"role": "assistant", "content": ans}
                                    ]
                                })
                            else:
                                paper_ids = [p.get("paper_id") or p.get("sha256") for p in uploaded_papers]
                                response = requests.post(f"{BACKEND_URL}/compare", json={"paper_ids": paper_ids})
                                
                                if response.status_code == 200:
                                    report_data = response.json()
                                    report_md = report_data.get("comparison_report")
                                    st.markdown(report_md)
                                    
                                    requests.post(f"{BACKEND_URL}/chat/sessions/{session_id}/messages", json={
                                        "messages": [
                                            {"role": "user", "content": query_to_process},
                                            {"role": "assistant", "content": report_md}
                                        ]
                                    })
                                else:
                                    st.error("Failed to generate comparison report from the backend.")
                            load_session(session_id)
                        except Exception as e:
                            st.error(f"Error running comparison: {e}")

            # Check if user query has scholarly search/discovery intent
            elif detect_search_intent(query_to_process):
                with st.chat_message("assistant"):
                    with st.spinner("Searching scholarly APIs (Semantic Scholar, OpenAlex, arXiv)..."):
                        try:
                            topic = query_to_process
                            for prefix in [
                                "find papers on the topic", "search for papers on the topic",
                                "find papers on", "search for papers on", "search papers on", 
                                "discover papers on", "recommend papers on", "give me some research paper on the topic", 
                                "give me some research paper on", "give me research papers on", "give me some papers on",
                                "give me ", "search ", "find ", "recommend ", "show me "
                            ]:
                                if topic.lower().startswith(prefix):
                                    topic = topic[len(prefix):].strip()
                                    break
                                    
                            response = requests.post(f"{BACKEND_URL}/search", json={"topic": topic, "limit_per_source": 3})
                            if response.status_code == 200:
                                search_results = response.json().get("results", [])
                                
                                if not search_results:
                                    ans = "No recent papers found matching your query."
                                    st.markdown(ans)
                                    requests.post(f"{BACKEND_URL}/chat/sessions/{session_id}/messages", json={
                                        "messages": [
                                            {"role": "user", "content": query_to_process},
                                            {"role": "assistant", "content": ans}
                                        ]
                                    })
                                else:
                                    st.markdown(f"### 🔍 Scholarly Papers Discovered on **'{topic}'**:")
                                    assistant_summary_content = f"Discovered recommended papers on **{topic}**:\n\n"
                                    
                                    for idx, paper in enumerate(search_results[:5], start=1):
                                        title = paper.get("title", "Untitled")
                                        authors = ", ".join(paper.get("authors") or [])
                                        year = paper.get("year") or "n.d."
                                        pdf_url = paper.get("pdf_url")
                                        paper_url = paper.get("paper_url")
                                        doi = paper.get("doi")
                                        
                                        display_url = pdf_url or paper_url
                                        if not display_url and doi:
                                            display_url = f"https://doi.org/{doi}"
                                        
                                        tags_html = ""
                                        if pdf_url:
                                            tags_html += f"<span class='paper-tag'><a href='{pdf_url}' target='_blank' style='color:#7C3AED; text-decoration:none;'>📄 PDF Link</a></span>"
                                        if paper_url:
                                            tags_html += f"<span class='paper-tag'><a href='{paper_url}' target='_blank' style='color:#7C3AED; text-decoration:none;'>🔗 Paper Page</a></span>"
                                        if not pdf_url and not paper_url and doi:
                                            tags_html += f"<span class='paper-tag'><a href='https://doi.org/{doi}' target='_blank' style='color:#7C3AED; text-decoration:none;'>🔗 Publisher Page</a></span>"
                                        if doi:
                                            tags_html += f"<span class='paper-tag'>DOI: {doi}</span>"
                                            
                                        paper_card_html = f"""
                                        <div class="paper-card">
                                            <div class="paper-title">[{idx}] {title} ({year})</div>
                                            <div class="paper-authors">Authors: {authors}</div>
                                            <div>
                                                {tags_html}
                                            </div>
                                        </div>
                                        """
                                        st.markdown(paper_card_html, unsafe_allow_html=True)
                                        
                                        url_str = f" | [Link]({display_url})" if display_url else ""
                                        assistant_summary_content += f"{idx}. **{title} ({year})** by *{authors}*{url_str}\n"
                                    
                                    assistant_summary_content += "\n*You can view details and run comparisons in the **Recommended Papers** tab.*"
                                    
                                    requests.post(f"{BACKEND_URL}/chat/sessions/{session_id}/recommendations", json={"papers": search_results})
                                    requests.post(f"{BACKEND_URL}/chat/sessions/{session_id}/messages", json={
                                        "messages": [
                                            {"role": "user", "content": query_to_process},
                                            {"role": "assistant", "content": assistant_summary_content}
                                        ]
                                    })
                                    load_session(session_id)
                            else:
                                st.error(f"Search API returned error: {response.json().get('detail')}")
                        except Exception as e:
                            st.error(f"Scholarly API search error: {e}")
                            
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            paper_id = ""
                            if st.session_state.active_paper_metadata:
                                paper_id = st.session_state.active_paper_metadata.get("paper_id") or st.session_state.active_paper_metadata.get("sha256") or ""
                            
                            payload = {
                                "session_id": session_id,
                                "paper_id": paper_id,
                                "question": query_to_process
                            }
                            response = requests.post(f"{BACKEND_URL}/chat", json=payload)
                            
                            if response.status_code == 200:
                                data = response.json()
                                st.markdown(data["answer"])
                                
                                citations = data.get("citations", [])
                                st.session_state.active_citations = citations  # Save to session state
                                
                                if citations:
                                    with st.expander("📚 Retrieved Grounding Context & Citations"):
                                        for cit in citations:
                                            st.write(f"- **Excerpt {cit['chunk_index']}** (Page {cit.get('page_number') or 'N/A'}, Similarity: {cit['score']}):")
                                            st.caption(f"\"{cit['snippet']}\"")
                                            
                                load_session(session_id)
                            else:
                                detail = response.json().get("detail", "Error communicating with backend.")
                                st.error(f"Backend blocked or returned error: {detail}")
                        except Exception as e:
                            st.error(f"Error querying assistant: {e}")

with tab_rec:
    if st.session_state.recommended_papers:
        st.subheader("📚 Recommended Papers in this Session")
        st.caption("You can compare any of the scholarly papers discovered below with your uploaded paper.")
        
        for idx, paper in enumerate(st.session_state.recommended_papers, start=1):
            title = paper.get("title", "Untitled")
            authors = ", ".join(paper.get("authors") or [])
            year = paper.get("year") or "n.d."
            abstract = paper.get("abstract") or "No abstract available."
            pdf_url = paper.get("pdf_url")
            paper_url = paper.get("paper_url")
            doi = paper.get("doi")
            
            url_markdown_links = []
            if pdf_url:
                url_markdown_links.append(f"[📄 PDF Link]({pdf_url})")
            if paper_url:
                url_markdown_links.append(f"[🔗 Paper Page]({paper_url})")
            if not pdf_url and not paper_url and doi:
                url_markdown_links.append(f"[🔗 Publisher Page](https://doi.org/{doi})")
                
            url_str = " | ".join(url_markdown_links) if url_markdown_links else "No link available"
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**[{idx}] {title} ({year})**")
                st.markdown(f"**Links:** {url_str}")
                st.caption(f"Authors: {authors} | DOI: {doi or 'N/A'}")
                with st.expander("Abstract Excerpt"):
                    st.write(abstract)
                    
            with col2:
                # Comparison Button
                if st.button("⚖️ Compare", key=f"comp_rec_{idx}_{st.session_state.current_session_id}"):
                    if not st.session_state.active_paper_metadata:
                        st.error("Please upload a PDF paper before running comparisons.")
                    else:
                        uploaded_paper_id = st.session_state.active_paper_metadata.get("paper_id") or st.session_state.active_paper_metadata.get("sha256") or ""
                        session_id = st.session_state.current_session_id
                        
                        with st.spinner("Downloading PDF and running three-level comparison strategy..."):
                            try:
                                compare_payload = {
                                    "session_id": session_id,
                                    "paper_id": uploaded_paper_id,
                                    "recommended_paper": paper
                                }
                                comp_response = requests.post(f"{BACKEND_URL}/compare/recommendation", json=compare_payload)
                                
                                if comp_response.status_code == 200:
                                    comp_data = comp_response.json()
                                    level = comp_data["comparison_level"]
                                    report = comp_data["comparison_report"]
                                    
                                    # Format visual comparison message
                                    comparison_msg_content = f"### ⚖️ Comparison Report: **{title}**\n"
                                    comparison_msg_content += f"- **Comparison Level**: `{level.upper()}`\n\n"
                                    comparison_msg_content += report
                                    
                                    # Append to session messages history in MongoDB
                                    user_msg = {"role": "user", "content": f"Compare recommended paper: {title}"}
                                    assistant_msg = {"role": "assistant", "content": comparison_msg_content}
                                    
                                    requests.post(f"{BACKEND_URL}/chat/sessions/{session_id}/messages", json={
                                        "messages": [user_msg, assistant_msg]
                                    })
                                    
                                    # Refresh and reload updated chat
                                    load_session(session_id)
                                else:
                                    err = comp_response.json().get("detail", "Failed comparison.")
                                    st.error(f"Comparison error: {err}")
                            except Exception as e:
                                st.error(f"Error executing comparison: {e}")
                                
            st.markdown("<hr style='border:1px solid #E5E7EB'/>", unsafe_allow_html=True)
    else:
        st.info("No recommended papers found in this session. Ask to search/find papers on a topic to discover related literature.")
