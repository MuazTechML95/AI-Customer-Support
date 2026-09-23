
"""
streamlit_app.py
----------------
NovaCart Customer Support - Streamlit UI

Run:
    streamlit run app/ui/streamlit_app.py
"""

import sys
import os
import uuid
from datetime import datetime

import streamlit as st


# ==========================================================================
# PROJECT ROOT
# ==========================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================================
# APPLICATION IMPORTS
# ==========================================================================

from app.config import settings
from app.main import run_startup_checks
from app.services import support_service
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import rebuild_index, get_index_status
from app.rag.ingestion import load_knowledge_base


# ==========================================================================
# UI CONSTANTS
# ==========================================================================

ACCENT_COLOR = "#4F46E5"
ACCENT_SOFT = "#EEF2FF"

SUGGESTED_QUESTIONS = [
    "How long does shipping take?",
    "What is your return policy?",
    "Do you accept cash on delivery?",
    "What's the warranty on laptops?",
]


# ==========================================================================
# PAGE CONFIG
# ==========================================================================

st.set_page_config(
    page_title=f"{settings.app_name} · {settings.company_name}",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# GLOBAL CSS
# ==========================================================================

st.markdown(
    f"""
    <style>

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }}

    #MainMenu,
    footer {{
        visibility: hidden;
    }}

    /* ==============================================================
       BRAND HEADER
       ============================================================== */

    .brand-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
        background: white;
        padding: 18px 22px;
        border-radius: 16px;
        border: 1px solid #ECECF3;
        box-shadow: 0 1px 3px rgba(17, 24, 39, 0.04);
    }}

    .brand-logo {{
        width: 48px;
        height: 48px;
        border-radius: 13px;
        background: linear-gradient(
            135deg,
            {ACCENT_COLOR},
            #7C3AED
        );
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        color: white;
        flex-shrink: 0;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
    }}

    .brand-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.15;
        letter-spacing: -0.01em;
    }}

    .brand-subtitle {{
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 2px;
    }}

    /* ==============================================================
       CHAT
       ============================================================== */

    div[data-testid="stChatMessage"] {{
        background: white;
        border: 1px solid #ECECF3;
        border-radius: 14px;
        padding: 4px 6px;
        margin-bottom: 8px;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.03);
    }}

    div[data-testid="stChatInput"] {{
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(17, 24, 39, 0.06);
    }}

    /* ==============================================================
       SOURCE BADGES
       ============================================================== */

    .source-badge {{
        display: inline-block;
        background-color: {ACCENT_SOFT};
        color: #3730a3;
        border-radius: 999px;
        padding: 3px 12px;
        margin: 3px 6px 3px 0;
        font-size: 0.78em;
        font-weight: 500;
        border: 1px solid #e0e7ff;
    }}

    /* ==============================================================
       STATUS
       ============================================================== */

    .status-pill {{
        display: inline-block;
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.78em;
        font-weight: 600;
    }}

    .status-ok {{
        background: #dcfce7;
        color: #166534;
    }}

    .status-warn {{
        background: #fef9c3;
        color: #854d0e;
    }}

    .status-bad {{
        background: #fee2e2;
        color: #991b1b;
    }}

    /* ==============================================================
       BUTTONS
       ============================================================== */

    div[data-testid="stButton"] > button {{
        border-radius: 999px !important;
        border: 1px solid #E0E0EA !important;
        background: white !important;
        color: #374151 !important;
        font-weight: 500 !important;
        transition: all 0.15s ease;
    }}

    div[data-testid="stButton"] > button:hover {{
        border-color: {ACCENT_COLOR} !important;
        color: {ACCENT_COLOR} !important;
        background: {ACCENT_SOFT} !important;
    }}

    div[data-testid="stButton"] > button[kind="primary"] {{
        background: linear-gradient(
            135deg,
            {ACCENT_COLOR},
            #7C3AED
        ) !important;
        color: white !important;
        border: none !important;
    }}

    /* ==============================================================
       SIDEBAR
       ============================================================== */

    section[data-testid="stSidebar"] {{
        background: white;
        border-right: 1px solid #ECECF3;
    }}

    /* ==============================================================
       METRICS
       ============================================================== */

    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid #ECECF3;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.03);
    }}

    button[data-baseweb="tab"] {{
        font-weight: 600;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# SESSION STATE
# ==========================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "escalated" not in st.session_state:
    st.session_state.escalated = False


# ==========================================================================
# HELPERS
# ==========================================================================

def _submit_question(text: str):
    """Queue a question for processing."""
    st.session_state.pending_question = text


def _count_unanswered_messages(messages):
    """Count assistant responses that were not grounded."""

    return sum(
        1
        for message in messages
        if (
            message.get("role") == "assistant"
            and message.get("grounded") is False
        )
    )


def render_brand_header(icon, title, subtitle):
    """
    Render a brand header using Streamlit's native HTML renderer.
    """

    st.html(
        f"""
        <div class="brand-header">

            <div class="brand-logo">
                {icon}
            </div>

            <div>
                <div class="brand-title">
                    {title}
                </div>

                <div class="brand-subtitle">
                    {subtitle}
                </div>
            </div>

        </div>
        """
    )


# ==========================================================================
# SIDEBAR
# ==========================================================================

with st.sidebar:

    render_brand_header(
        "💬",
        settings.app_name,
        f"AI Support · {settings.company_name}",
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "💬  Chat",
            "🛠️  Admin Dashboard",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:

        if st.button(
            "🗑️ Clear chat",
            use_container_width=True,
        ):

            support_service.get_session_store().clear_session(
                st.session_state.session_id
            )

            st.session_state.display_messages = []
            st.session_state.escalated = False
            st.session_state.pending_question = None

            st.rerun()

    with col_b:

        if st.button(
            "🙋 Human help",
            use_container_width=True,
        ):

            st.session_state.escalated = True
            st.rerun()

    st.divider()

    st.caption(
        f"Session `{st.session_state.session_id[:8]}…`"
    )

    st.caption(
        "Grounded RAG · Retrieval-Augmented Generation"
    )

    environment = getattr(
        settings,
        "environment",
        "development",
    )

    st.caption(
        f"Environment: `{environment}`"
    )


# ==========================================================================
# STARTUP CHECKS
# ==========================================================================

startup = run_startup_checks()

if startup["config_errors"]:

    st.error(
        "⚠️ Configuration problem(s) detected:\n\n"
        + "\n".join(
            f"- {error}"
            for error in startup["config_errors"]
        )
    )

    st.info(
        "Fix your `.env` file using `.env.example` "
        "and restart the application."
    )

    st.stop()


index_ready = startup["index_status"]["exists"]

if not index_ready:

    st.warning(
        "⚠️ The knowledge base index has not been built yet. "
        "Go to **Admin Dashboard → Upload / Rebuild** "
        "and click **Rebuild Index** before chatting."
    )


# ==========================================================================
# CHAT PAGE
# ==========================================================================

def render_chat_page():

    render_brand_header(
        "🛍️",
        f"{settings.company_name} Customer Support",
        "Ask about orders, shipping, returns, warranty, or our products.",
    )

    st.write("")

    # ----------------------------------------------------------------------
    # HUMAN ESCALATION
    # ----------------------------------------------------------------------

    if st.session_state.escalated:

        st.info(
            "🙋 **You've requested a human agent.** "
            "A support representative will follow up with you shortly. "
            "You can keep chatting with the assistant in the meantime.",
            icon="🙋",
        )

    # ----------------------------------------------------------------------
    # EMPTY STATE
    # ----------------------------------------------------------------------

    if not st.session_state.display_messages:

        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            st.write(
                f"👋 Hi! I'm the **{settings.app_name}** assistant "
                f"for **{settings.company_name}**. "
                "I can only answer using our official support "
                "knowledge base, so my answers stay accurate. "
                "What can I help you with?"
            )

        st.caption("Try one of these:")

        chip_cols = st.columns(
            len(SUGGESTED_QUESTIONS)
        )

        for col, question in zip(
            chip_cols,
            SUGGESTED_QUESTIONS,
        ):

            with col:

                if st.button(
                    question,
                    key=f"chip_{question}",
                    use_container_width=True,
                ):

                    _submit_question(question)

    # ----------------------------------------------------------------------
    # CHAT HISTORY
    # ----------------------------------------------------------------------

    for index, message in enumerate(
        st.session_state.display_messages
    ):

        avatar = (
            "🧑"
            if message["role"] == "user"
            else "🤖"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar,
        ):

            st.write(
                message["content"]
            )

            # Sources
            if message.get("sources"):

                source_html = " ".join(
                    f'<span class="source-badge">'
                    f'📄 {source}'
                    f'</span>'
                    for source in message["sources"]
                )

                st.markdown(
                    source_html,
                    unsafe_allow_html=True,
                )

            # Feedback
            if message["role"] == "assistant":

                fcol1, fcol2, _ = st.columns(
                    [1, 1, 8]
                )

                feedback = message.get(
                    "feedback"
                )

                with fcol1:

                    if st.button(
                        "👍",
                        key=f"up_{index}",
                        disabled=feedback is not None,
                    ):

                        st.session_state.display_messages[
                            index
                        ]["feedback"] = "up"

                        st.rerun()

                with fcol2:

                    if st.button(
                        "👎",
                        key=f"down_{index}",
                        disabled=feedback is not None,
                    ):

                        st.session_state.display_messages[
                            index
                        ]["feedback"] = "down"

                        st.rerun()

                if feedback == "up":

                    st.caption(
                        "Thanks for the feedback! 🙏"
                    )

                elif feedback == "down":

                    st.caption(
                        "Thanks — we'll use this to improve. 🙏"
                    )

    # ----------------------------------------------------------------------
    # CHAT INPUT
    # ----------------------------------------------------------------------

    user_input = st.chat_input(
        "Type your question…",
        disabled=not index_ready,
    )

    if user_input:

        _submit_question(user_input)

    # ----------------------------------------------------------------------
    # PROCESS QUESTION
    # ----------------------------------------------------------------------

    if st.session_state.pending_question:

        question = st.session_state.pending_question

        st.session_state.pending_question = None

        # User message
        st.session_state.display_messages.append(
            {
                "role": "user",
                "content": question,
                "sources": [],
                "ts": datetime.now(),
            }
        )

        with st.chat_message(
            "user",
            avatar="🧑",
        ):

            st.write(question)

        # AI response
        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            with st.spinner("Thinking…"):

                try:

                    result = support_service.ask(
                        st.session_state.session_id,
                        question,
                    )

                except Exception as exc:

                    result = {
                        "answer": (
                            "Something went wrong while "
                            "processing your question. "
                            "Please try again."
                        ),
                        "sources": [],
                        "grounded": False,
                    }

                    if settings.debug:
                        st.exception(exc)

            st.write(
                result["answer"]
            )

            # Sources
            if result.get("sources"):

                source_html = " ".join(
                    f'<span class="source-badge">'
                    f'📄 {source}'
                    f'</span>'
                    for source in result["sources"]
                )

                st.markdown(
                    source_html,
                    unsafe_allow_html=True,
                )

        # Save assistant message
        st.session_state.display_messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get(
                    "sources",
                    [],
                ),
                "grounded": result.get(
                    "grounded",
                    False,
                ),
                "ts": datetime.now(),
                "feedback": None,
            }
        )

        st.rerun()


# ==========================================================================
# ADMIN DASHBOARD
# ==========================================================================

def render_admin_page():

    render_brand_header(
        "🛠️",
        "Admin Dashboard",
        "Knowledge base, index health, and live session metrics.",
    )

    st.write("")

    tab_overview, tab_kb, tab_upload, tab_settings = st.tabs(
        [
            "📊 Overview",
            "📚 Knowledge Base",
            "⬆️ Upload / Rebuild",
            "⚙️ Settings",
        ]
    )

    # ======================================================================
    # OVERVIEW
    # ======================================================================

    with tab_overview:

        status = get_index_status()

        current_messages = (
            st.session_state.display_messages
        )

        total_msgs = len(
            current_messages
        )

        user_msgs = sum(
            1
            for message in current_messages
            if message.get("role") == "user"
        )

        escalations = (
            1
            if st.session_state.escalated
            else 0
        )

        thumbs_up = sum(
            1
            for message in current_messages
            if message.get("feedback") == "up"
        )

        thumbs_down = sum(
            1
            for message in current_messages
            if message.get("feedback") == "down"
        )

        unanswered = _count_unanswered_messages(
            current_messages
        )

        # Metrics
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Conversations (this session)",
            1 if user_msgs else 0,
        )

        c2.metric(
            "Messages exchanged",
            total_msgs,
        )

        c3.metric(
            "Escalations",
            escalations,
        )

        c4.metric(
            "Unanswered / fallback replies",
            unanswered,
        )

        c5, c6, c7 = st.columns(3)

        c5.metric(
            "👍 Helpful",
            thumbs_up,
        )

        c6.metric(
            "👎 Not helpful",
            thumbs_down,
        )

        c7.metric(
            "Chunks indexed",
            status.get(
                "chunk_count",
                0,
            ),
        )

        st.caption(
            "These metrics reflect the current browser session only. "
            "In a production deployment they would be aggregated "
            "from persistent Conversation, Message, and Feedback data."
        )

        st.divider()

        st.markdown(
            "##### Index Health"
        )

        pill_class = (
            "status-ok"
            if status.get("exists")
            else "status-bad"
        )

        pill_text = (
            "Ready"
            if status.get("exists")
            else "Not built"
        )

        st.markdown(
            f"""
            <span class="status-pill {pill_class}">
                ● {pill_text}
            </span>
            """,
            unsafe_allow_html=True,
        )

    # ======================================================================
    # KNOWLEDGE BASE
    # ======================================================================

    with tab_kb:

        try:

            documents = load_knowledge_base(
                settings.knowledge_base_path
            )

        except Exception as exc:

            st.error(
                f"Could not load knowledge base: {exc}"
            )

            documents = []

        st.caption(
            f"{len(documents)} document(s) found in "
            f"`{settings.knowledge_base_path}`"
        )

        categories = sorted(
            {
                document.category
                for document in documents
            }
        )

        filter_col, search_col = st.columns(
            [1, 2]
        )

        with filter_col:

            selected_cats = st.multiselect(
                "Filter by category",
                categories,
                default=categories,
            )

        with search_col:

            search_term = st.text_input(
                "Search documents",
                placeholder=(
                    "e.g. shipping, warranty, laptop…"
                ),
            )

        search_lower = search_term.lower()

        filtered = [
            document
            for document in documents
            if (
                document.category in selected_cats
                and (
                    not search_lower
                    or search_lower in document.content.lower()
                    or search_lower in document.title.lower()
                )
            )
        ]

        st.caption(
            f"Showing {len(filtered)} of "
            f"{len(documents)} document(s)"
        )

        for document in filtered:

            with st.expander(
                f"📄 {document.title} · "
                f"_{document.category}_"
            ):

                st.caption(
                    f"doc_id: `{document.doc_id}` · "
                    f"file: `{document.source_file}`"
                )

                content = document.content

                if len(content) > 1800:

                    content = (
                        content[:1800]
                        + "…"
                    )

                st.markdown(
                    content
                )

    # ======================================================================
    # UPLOAD / REBUILD
    # ======================================================================

    with tab_upload:

        st.markdown(
            "##### Upload a new document"
        )

        st.caption(
            "Upload a Markdown (.md) file with "
            "front-matter (title, category, doc_id). "
            "It will be saved into the FAQ folder. "
            "Rebuild the index afterwards for it to take effect."
        )

        uploaded_file = st.file_uploader(
            "Choose a .md file",
            type=["md"],
        )

        if uploaded_file is not None:

            if not uploaded_file.name.lower().endswith(
                ".md"
            ):

                st.error(
                    "Only .md (Markdown) files are supported."
                )

            elif uploaded_file.size > 2_000_000:

                st.error(
                    "File is too large (max 2 MB)."
                )

            else:

                try:

                    target_dir = os.path.join(
                        settings.knowledge_base_path,
                        "faqs",
                    )

                    os.makedirs(
                        target_dir,
                        exist_ok=True,
                    )

                    target_path = os.path.join(
                        target_dir,
                        uploaded_file.name,
                    )

                    with open(
                        target_path,
                        "wb",
                    ) as file:

                        file.write(
                            uploaded_file.getbuffer()
                        )

                    st.success(
                        f"Saved '{uploaded_file.name}'. "
                        "Click 'Rebuild Index' below "
                        "to include it."
                    )

                except Exception as exc:

                    st.error(
                        f"Failed to save file: {exc}"
                    )

        st.divider()

        st.markdown(
            "##### Rebuild vector index"
        )

        st.caption(
            "Re-reads every document in the knowledge base, "
            "re-chunks, re-embeds, and replaces the vector index."
        )

        if st.button(
            "🔄 Rebuild Index",
            type="primary",
        ):

            progress = st.progress(
                0,
                text="Loading documents…",
            )

            try:

                progress.progress(
                    25,
                    text="Loading embedding model…",
                )

                embedding_model = EmbeddingModel(
                    settings.embedding_model
                )

                progress.progress(
                    60,
                    text="Chunking + embedding knowledge base…",
                )

                result = rebuild_index(
                    embedding_model
                )

                progress.progress(
                    100,
                    text="Done.",
                )

                st.success(
                    f"✅ Index rebuilt: "
                    f"{result['documents_indexed']} documents, "
                    f"{result['chunks_indexed']} chunks."
                )

            except Exception as exc:

                st.error(
                    f"Rebuild failed: {exc}"
                )

    # ======================================================================
    # SETTINGS
    # ======================================================================

    with tab_settings:

        st.caption(
            "Current runtime configuration "
            "(read-only — edit `.env` to change these)."
        )

        cfg_rows = [
            ("App name", settings.app_name),
            ("Company / tenant", settings.company_name),
            ("LLM provider", settings.llm_provider),
            ("LLM model", settings.llm_model),
            ("Embedding model", settings.embedding_model),
            ("Vector store path", settings.vector_store_path),
            ("Retrieval top-K", settings.retrieval_top_k),
            (
                "Similarity threshold",
                settings.similarity_threshold,
            ),
            (
                "Max history turns",
                settings.max_history_turns,
            ),
            ("Debug mode", settings.debug),
        ]

        for label, value in cfg_rows:

            c1, c2 = st.columns(
                [1, 2]
            )

            c1.markdown(
                f"**{label}**"
            )

            c2.code(
                str(value),
                language=None,
            )


# ==========================================================================
# ROUTER
# ==========================================================================

if page.startswith("💬"):

    render_chat_page()

else:

    render_admin_page()
