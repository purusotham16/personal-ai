import streamlit as st
import os
import re
import hashlib
import tempfile
import requests

from dotenv import load_dotenv
from faster_whisper import WhisperModel

from rag import retrieve

from document_handler import (
    create_document,
    get_document_context,
    SUPPORTED_EXTENSIONS
)

from database import (
    create_database,
    create_conversation,
    save_message
)

from chat_history import (
    render_chat_history
)

from memory import (
    create_memory_table,
    save_memory,
    get_memory_context,
    get_memories,
    delete_memory,
    clear_memories,
    count_memories
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

create_database()
create_memory_table()


# ============================================================
# LOAD ENVIRONMENT + GROQ DIAGNOSTICS
# ============================================================

# Always load .env from the same folder as this app.py first.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(APP_DIR, ".env"), override=False)
load_dotenv(override=False)

groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Put GROQ_API_KEY=your_key in the .env file next to app.py.")
    st.stop()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_CHAT_URL = f"{GROQ_BASE_URL}/chat/completions"
GROQ_MODELS_URL = f"{GROQ_BASE_URL}/models"

# We do NOT hard-code one model and assume every Groq project can access it.
# The app asks Groq which models THIS API KEY can actually see, then selects one.
PREFERRED_CHAT_MODELS = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "gemma2-9b-it",
    "llama3-8b-8192",
]
PREFERRED_WEB_MODELS = [
    "groq/compound-mini",
    "groq/compound",
]


def get_groq_models():
    """Return models visible to the current API key/project."""
    try:
        r = requests.get(
            GROQ_MODELS_URL,
            headers={"Authorization": f"Bearer {groq_api_key}"},
            timeout=20,
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "Cannot reach Groq. Check internet/DNS/firewall. Details: " + str(e)
        )

    if r.status_code != 200:
        try:
            detail = r.json().get("error", {}).get("message", r.text)
        except Exception:
            detail = r.text
        if r.status_code == 401:
            raise RuntimeError("Groq API key is invalid or expired (401). Create/check the key in GroqCloud.")
        if r.status_code == 403:
            raise RuntimeError("This Groq project/key is not allowed to list models (403). Check project/model permissions.")
        raise RuntimeError(f"Groq model-list request failed ({r.status_code}): {detail}")

    data = r.json().get("data", [])
    return [m for m in data if isinstance(m, dict) and m.get("id")]


try:
    _groq_models = get_groq_models()
    _groq_model_ids = {m["id"] for m in _groq_models}
except Exception as _startup_error:
    st.error(f"Groq startup check failed: {_startup_error}")
    st.info("The app stopped before sending a chat request so you get one clear error instead of changing errors.")
    st.stop()

GROQ_CHAT_MODEL = next(
    (m for m in PREFERRED_CHAT_MODELS if m in _groq_model_ids),
    None,
)
GROQ_WEB_MODEL = next(
    (m for m in PREFERRED_WEB_MODELS if m in _groq_model_ids),
    None,
)

if GROQ_CHAT_MODEL is None:
    # Pick a normal text model visible to the key, excluding audio/safety/compound systems.
    blocked_prefixes = ("whisper", "distil-whisper", "llama-guard", "groq/compound")
    candidates = [
        m["id"] for m in _groq_models
        if isinstance(m.get("id"), str)
        and not m["id"].startswith(blocked_prefixes)
    ]
    if candidates:
        GROQ_CHAT_MODEL = candidates[0]

if GROQ_CHAT_MODEL is None:
    visible = ", ".join(sorted(_groq_model_ids)) or "none"
    st.error("This API key has no usable Groq chat model access.")
    st.code(visible)
    st.info("Open GroqCloud and check the project's Model Permissions. The app cannot fix a key that has no model access.")
    st.stop()


def generate_ai_response(prompt, use_web=False):
    """Send a small request and fall back only when a model is unavailable.

    We never switch models because of 429 rate limits; doing so can consume
    more quota. A fallback is used only for 403/404 model-access problems.
    """
    if use_web and GROQ_WEB_MODEL:
        candidate_models = [GROQ_WEB_MODEL, GROQ_CHAT_MODEL]
    else:
        candidate_models = [GROQ_CHAT_MODEL]

    # Add other preferred chat models that THIS key actually listed.
    for candidate in PREFERRED_CHAT_MODELS:
        if candidate in _groq_model_ids and candidate not in candidate_models:
            candidate_models.append(candidate)

    last_error = "Unknown Groq error"

    for model in candidate_models:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 400,
            "temperature": 0.2,
        }

        try:
            response = requests.post(
                GROQ_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=90,
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                "Could not connect to Groq. Check internet/DNS/firewall. Details: " + str(e)
            )

        if response.status_code == 200:
            return response.json()

        try:
            error_data = response.json()
            message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            message = response.text

        last_error = message

        if response.status_code == 401:
            raise RuntimeError("Groq authentication failed (401). Check GROQ_API_KEY.")

        if response.status_code in (403, 404):
            # Model-specific access problem. Try the next model that the key
            # reported as available, instead of crashing immediately.
            continue

        if response.status_code == 413:
            raise RuntimeError(
                "Groq rejected the request as too large (413). "
                "The app's context limits need to be reduced further."
            )

        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            extra = (
                f" Try again after about {retry_after} seconds."
                if retry_after else
                " Wait for the free-tier rate-limit window to reset."
            )
            raise RuntimeError(
                f"Groq free-tier rate limit reached (429).{extra} Details: {message}"
            )

        raise RuntimeError(f"Groq API error ({response.status_code}): {message}")

    raise RuntimeError(
        "None of the Groq models available to this API key accepted the request. "
        f"Last error: {last_error}"
    )


def clip_text(text, max_chars):
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[context truncated]"


def should_use_web(question):
    q = question.lower().strip()
    web_words = [
        "latest", "current", "today", "now", "recent", "news",
        "this week", "this month", "price", "stock", "weather",
        "release", "version", "online", "search the web", "look up",
        "what happened", "who won", "score", "schedule", "available"
    ]
    return any(word in q for word in web_words)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Personal AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "current_conversation" not in st.session_state:

    st.session_state.current_conversation = None


if "theme" not in st.session_state:

    st.session_state.theme = "dark"


if "search_query" not in st.session_state:

    st.session_state.search_query = ""


if "quick_question" not in st.session_state:

    st.session_state.quick_question = None


if "last_voice_audio_hash" not in st.session_state:

    st.session_state.last_voice_audio_hash = None


if "uploaded_documents" not in st.session_state:

    st.session_state.uploaded_documents = []


# ============================================================
# THEME
# ============================================================

theme = st.session_state.theme


if theme == "dark":

    bg_color = "#0e1117"
    sidebar_color = "#10141c"
    card_color = "#1b1e26"
    button_color = "#292b36"
    text_color = "#f5f5f5"
    secondary_text = "#9aa4b2"
    border_color = "#343844"
    input_color = "#242631"

else:

    bg_color = "#f5f7fb"
    sidebar_color = "#ffffff"
    card_color = "#ffffff"
    button_color = "#eef1f6"
    text_color = "#171a21"
    secondary_text = "#5f6878"
    border_color = "#d7dce5"
    input_color = "#ffffff"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {sidebar_color};
        border-right: 1px solid {border_color};
    }}

    section[data-testid="stSidebar"] * {{
        color: {text_color};
    }}

    .sidebar-title {{
        font-size: 27px;
        font-weight: 700;
        color: {text_color};
        margin-bottom: 4px;
    }}

    .sidebar-subtitle {{
        color: {secondary_text};
        font-size: 14px;
        margin-bottom: 20px;
    }}

    .main-title {{
        text-align: center;
        font-size: 46px;
        font-weight: 750;
        color: {text_color};
        margin-top: 50px;
        margin-bottom: 10px;
    }}

    .main-subtitle {{
        text-align: center;
        color: {secondary_text};
        font-size: 17px;
        margin-bottom: 35px;
    }}

    .quick-title {{
        color: {secondary_text};
        font-size: 15px;
        margin-bottom: 8px;
    }}

    [data-testid="stChatMessage"] {{
        background-color: {card_color};
        border: 1px solid {border_color};
        border-radius: 15px;
        padding: 12px;
        margin-bottom: 10px;
    }}

    [data-testid="stChatInput"] {{
        background-color: {input_color};
        border: 1px solid {border_color};
        border-radius: 14px;
    }}

    [data-testid="stChatInput"] textarea {{
        color: {text_color} !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: {secondary_text} !important;
    }}

    .stButton > button {{
        background-color: {button_color};
        color: {text_color};
        border: 1px solid {border_color};
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
    }}

    .stButton > button:hover {{
        border-color: #7c83ff;
        color: {text_color};
    }}

    section[data-testid="stSidebar"] .stButton > button {{
        background-color: {button_color};
        color: {text_color};
        border: 1px solid {border_color};
    }}

    [data-testid="stExpander"] {{
        border: 1px solid {border_color};
        border-radius: 10px;
        background-color: {card_color};
    }}

    .document-card {{
        background-color: {card_color};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 8px;
    }}

    .document-name {{
        color: {text_color};
        font-weight: 600;
        font-size: 14px;
    }}

    .document-info {{
        color: {secondary_text};
        font-size: 12px;
    }}

    .memory-card {{
        background-color: {card_color};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 8px;
    }}

    .memory-text {{
        color: {text_color};
        font-size: 13px;
        line-height: 1.4;
    }}

    .memory-category {{
        color: {secondary_text};
        font-size: 11px;
        margin-top: 4px;
    }}

    .source-card {{
        background-color: {card_color};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 10px;
        margin-top: 6px;
        margin-bottom: 6px;
    }}

    .source-title {{
        color: {text_color};
        font-weight: 600;
        font-size: 13px;
    }}

    .source-url {{
        color: #7c83ff;
        font-size: 12px;
        word-break: break-all;
    }}

    .capability {{
        background-color: {card_color};
        border: 1px solid {border_color};
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        color: {text_color};
        font-weight: 600;
    }}

    .footer {{
        text-align: center;
        color: {secondary_text};
        font-size: 13px;
        margin-top: 35px;
        margin-bottom: 80px;
    }}

    hr {{
        border-color: {border_color};
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MEMORY DETECTION
# ============================================================

def detect_memory(user_input):

    text = user_input.strip()

    if not text:

        return None, "general"


    lower_text = text.lower()


    # --------------------------------------------------------
    # EXPLICIT REMEMBER REQUEST
    # --------------------------------------------------------

    remember_patterns = [

        "remember that ",
        "remember this ",
        "remember my ",
        "don't forget that ",
        "dont forget that ",
        "please remember ",
        "save this ",
        "keep this in mind "

    ]


    for pattern in remember_patterns:

        if lower_text.startswith(pattern):

            memory_text = text[
                len(pattern):
            ].strip()


            if memory_text:

                return (
                    memory_text,
                    "personal"
                )


    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name_match = re.search(
        r"\bmy name is\s+(.+)",
        text,
        re.IGNORECASE
    )


    if name_match:

        value = name_match.group(
            1
        ).strip()


        return (
            f"The user's name is {value}.",
            "personal"
        )


    # --------------------------------------------------------
    # FAVORITE
    # --------------------------------------------------------

    favorite_match = re.search(
        r"\bmy favou?rite\s+(.+?)\s+is\s+(.+)",
        text,
        re.IGNORECASE
    )


    if favorite_match:

        thing = favorite_match.group(
            1
        ).strip()


        value = favorite_match.group(
            2
        ).strip()


        return (
            f"The user's favorite {thing} is {value}.",
            "preference"
        )


    # --------------------------------------------------------
    # LIKES
    # --------------------------------------------------------

    like_match = re.search(
        r"\bi (?:really )?like\s+(.+)",
        text,
        re.IGNORECASE
    )


    if like_match:

        value = like_match.group(
            1
        ).strip()


        return (
            f"The user likes {value}.",
            "preference"
        )


    # --------------------------------------------------------
    # PREFERENCES
    # --------------------------------------------------------

    prefer_match = re.search(
        r"\bi prefer\s+(.+)",
        text,
        re.IGNORECASE
    )


    if prefer_match:

        value = prefer_match.group(
            1
        ).strip()


        return (
            f"The user prefers {value}.",
            "preference"
        )


    # --------------------------------------------------------
    # PROGRAMMING LANGUAGE
    # --------------------------------------------------------

    language_match = re.search(
        r"\bmy (?:preferred |main )?programming language\s+is\s+(.+)",
        text,
        re.IGNORECASE
    )


    if language_match:

        value = language_match.group(
            1
        ).strip()


        return (
            f"The user's programming language is {value}.",
            "technical"
        )


    # --------------------------------------------------------
    # PROJECT
    # --------------------------------------------------------

    project_match = re.search(
        r"\bmy (?:main |final year )?project\s+is\s+(.+)",
        text,
        re.IGNORECASE
    )


    if project_match:

        value = project_match.group(
            1
        ).strip()


        return (
            f"The user's project is {value}.",
            "project"
        )


    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    study_match = re.search(
        r"\bi (?:study|am studying)\s+(.+)",
        text,
        re.IGNORECASE
    )


    if study_match:

        value = study_match.group(
            1
        ).strip()


        return (
            f"The user studies {value}.",
            "education"
        )


    # --------------------------------------------------------
    # WORK
    # --------------------------------------------------------

    work_match = re.search(
        r"\bi work (?:at|for)\s+(.+)",
        text,
        re.IGNORECASE
    )


    if work_match:

        value = work_match.group(
            1
        ).strip()


        return (
            f"The user works at {value}.",
            "work"
        )


    return None, "general"


# ============================================================
# GROQ COMPOUND WEB SEARCH
# ============================================================

def _collect_source_items(value, sources):
    """Best-effort extraction of URLs from Groq executed_tools."""
    if isinstance(value, dict):
        url = value.get("url") or value.get("uri")
        title = value.get("title") or value.get("name") or "Web source"

        if isinstance(url, str) and url.startswith(("http://", "https://")):
            sources.append({"title": str(title), "url": url})

        for child in value.values():
            _collect_source_items(child, sources)

    elif isinstance(value, list):
        for child in value:
            _collect_source_items(child, sources)


def extract_web_sources(response_data):
    sources = []

    try:
        choices = response_data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            executed_tools = message.get("executed_tools", [])
            _collect_source_items(executed_tools, sources)
    except Exception:
        return []

    unique_sources = []
    seen_urls = set()

    for source in sources:
        url = source["url"]

        if url in seen_urls:
            continue

        seen_urls.add(url)
        unique_sources.append(source)

    return unique_sources


# ============================================================
# DISPLAY WEB SOURCES
# ============================================================

def display_web_sources(sources):

    if not sources:

        return


    st.markdown(
        "### 🌐 Web Sources"
    )


    for index, source in enumerate(
        sources,
        start=1
    ):

        title = source.get(
            "title",
            "Web source"
        )


        url = source.get(
            "url",
            ""
        )


        st.markdown(
            f"""
            <div class="source-card">

                <div class="source-title">
                    {index}. {title}
                </div>

                <div class="source-url">
                    <a href="{url}"
                       target="_blank">
                       {url}
                    </a>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="sidebar-title">
            🤖 Personal AI
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="sidebar-subtitle">
            Your intelligent personal assistant
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # APPEARANCE
    # --------------------------------------------------------

    st.markdown(
        "### 🎨 Appearance"
    )


    if theme == "dark":

        if st.button(
            "☀️ Switch to Light Mode",
            use_container_width=True
        ):

            st.session_state.theme = "light"

            st.rerun()

    else:

        if st.button(
            "🌙 Switch to Dark Mode",
            use_container_width=True
        ):

            st.session_state.theme = "dark"

            st.rerun()


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.current_conversation = None

        st.rerun()


    st.markdown("---")


    # ========================================================
    # DOCUMENT UPLOAD
    # ========================================================

    st.markdown(
        "### 📎 Documents"
    )


    st.caption(
        "Upload documents and ask questions about them."
    )


    uploaded_files = st.file_uploader(
        "Upload files",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        key="document_uploader"
    )


    # --------------------------------------------------------
    # PROCESS UPLOADED FILES
    # --------------------------------------------------------

    if uploaded_files:

        existing_names = [

            document["name"]

            for document
            in st.session_state.uploaded_documents

        ]


        for uploaded_file in uploaded_files:

            if uploaded_file.name in existing_names:

                continue


            try:

                file_bytes = (
                    uploaded_file.getvalue()
                )


                document = create_document(
                    uploaded_file.name,
                    file_bytes
                )


                if not document["text"]:

                    st.warning(
                        f"Could not extract text "
                        f"from {uploaded_file.name}"
                    )

                    continue


                st.session_state.uploaded_documents.append(
                    document
                )


                st.success(
                    f"Added {uploaded_file.name}"
                )


            except Exception as e:

                st.error(
                    f"Error reading "
                    f"{uploaded_file.name}: {e}"
                )


    # --------------------------------------------------------
    # SHOW UPLOADED DOCUMENTS
    # --------------------------------------------------------

    if st.session_state.uploaded_documents:

        st.markdown(
            "**Uploaded documents**"
        )


        for document in (
            st.session_state.uploaded_documents
        ):

            st.markdown(
                f"""
                <div class="document-card">

                    <div class="document-name">
                        📄 {document["name"]}
                    </div>

                    <div class="document-info">
                        {document["chunk_count"]} chunks
                        •
                        {document["characters"]:,} characters
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        if st.button(
            "🗑️ Clear Uploaded Documents",
            use_container_width=True
        ):

            st.session_state.uploaded_documents = []

            st.rerun()


    st.markdown("---")


    # ========================================================
    # MEMORY
    # ========================================================

    st.markdown(
        "### 🧠 Memory"
    )


    memory_count = count_memories()


    st.caption(
        f"{memory_count} saved memory"
        + (
            "ies"
            if memory_count != 1
            else ""
        )
    )


    memories = get_memories()


    if memories:

        with st.expander(
            "View saved memories"
        ):

            for (
                memory_id,
                memory_text,
                category,
                created_at
            ) in memories:

                st.markdown(
                    f"""
                    <div class="memory-card">

                        <div class="memory-text">
                            🧠 {memory_text}
                        </div>

                        <div class="memory-category">
                            {category}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                if st.button(
                    "Delete",
                    key=f"memory_delete_{memory_id}",
                    use_container_width=True
                ):

                    delete_memory(
                        memory_id
                    )

                    st.rerun()


            if st.button(
                "🗑️ Clear All Memories",
                key="clear_all_memories",
                use_container_width=True
            ):

                clear_memories()

                st.rerun()


    else:

        st.caption(
            "No memories saved yet."
        )


    st.markdown("---")


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    render_chat_history()


    st.markdown("---")


    # ========================================================
    # CAPABILITIES
    # ========================================================

    st.markdown(
        "### ⚡ Capabilities"
    )


    capabilities = [

        "🟢 Groq AI (Free Tier)",

        "🌐 Web Search (when available)",

        "🟢 RAG Documents",

        "🟢 File Upload",

        "🟢 Long-Term Memory",

        "🟢 Conversation Memory",

        "🎤 Voice Input",

        "🟢 LangGraph"

    ]


    for capability in capabilities:

        st.markdown(
            f"""
            <div class="capability">
                {capability}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# GROQ STATUS
# ============================================================
with st.sidebar:
    with st.expander("🔧 Groq connection status"):
        st.success("Groq API key accepted and model list loaded.")
        st.write(f"Chat model: `{GROQ_CHAT_MODEL}`")
        st.write(f"Web model: `{GROQ_WEB_MODEL or 'unavailable'}`")
        st.caption("The app selects only models visible to your current API key/project.")


# ============================================================
# MAIN AREA
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="main-title">
            👋 Hey! How can I help you?
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="main-subtitle">
            Ask me anything — technical questions,
            web questions, documents, projects,
            internship or personal information.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # DOCUMENT STATUS
    # --------------------------------------------------------

    if st.session_state.uploaded_documents:

        count = len(
            st.session_state.uploaded_documents
        )


        st.info(
            f"📎 {count} document(s) ready "
            f"for questions."
        )


    # --------------------------------------------------------
    # MEMORY STATUS
    # --------------------------------------------------------

    if memory_count:

        st.success(
            f"🧠 {memory_count} personal "
            f"memory/memories available."
        )


    # --------------------------------------------------------
    # QUICK QUESTIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="quick-title">Quick questions</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        if st.button(
            "📚 My Skills",
            use_container_width=True
        ):

            st.session_state.quick_question = (
                "What are my technical skills?"
            )

            st.rerun()


    with col2:

        if st.button(
            "🚀 My Project",
            use_container_width=True
        ):

            st.session_state.quick_question = (
                "Tell me about my final year project."
            )

            st.rerun()


    with col3:

        if st.button(
            "💼 My Internship",
            use_container_width=True
        ):

            st.session_state.quick_question = (
                "Tell me about my internship."
            )

            st.rerun()


# ============================================================
# DISPLAY EXISTING CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# VOICE INPUT
# ============================================================

@st.cache_resource
def get_whisper_model():

    return WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"
    )


def transcribe_voice_audio(
    audio_file
):

    if audio_file is None:

        return ""


    try:

        audio_bytes = audio_file.getvalue()


        if not audio_bytes:

            return ""


        # ----------------------------------------------------
        # SAVE TEMP AUDIO
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_audio:

            temp_audio.write(
                audio_bytes
            )

            temp_audio_path = (
                temp_audio.name
            )


        try:

            # ------------------------------------------------
            # LOCAL WHISPER
            # ------------------------------------------------

            model = get_whisper_model()


            segments, info = model.transcribe(
                temp_audio_path,
                beam_size=5,
                vad_filter=True
            )


            transcript = " ".join(

                segment.text.strip()

                for segment in segments

                if segment.text.strip()

            )


            return transcript.strip()


        finally:

            try:

                os.remove(
                    temp_audio_path
                )

            except OSError:

                pass


    except Exception as e:

        st.error(
            f"Voice transcription failed: {e}"
        )

        return ""


voice_input = st.audio_input(
    "🎤 Speak your message",
    key="voice_input"
)


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Message Personal AI..."
)


# ============================================================
# PROCESS VOICE
# ============================================================

if voice_input is not None:

    audio_bytes = voice_input.getvalue()


    audio_hash = hashlib.sha256(
        audio_bytes
    ).hexdigest()


    if (
        audio_hash
        != st.session_state.last_voice_audio_hash
    ):

        st.session_state.last_voice_audio_hash = (
            audio_hash
        )


        with st.spinner(
            "🎤 Converting your voice to text..."
        ):

            voice_text = (
                transcribe_voice_audio(
                    voice_input
                )
            )


        if voice_text:

            user_input = voice_text


            st.info(
                f"🎤 You said: {voice_text}"
            )


# ============================================================
# QUICK QUESTION INPUT
# ============================================================

if (
    "quick_question"
    in st.session_state
    and
    st.session_state.quick_question
):

    user_input = (
        st.session_state.quick_question
    )


    st.session_state.quick_question = None


# ============================================================
# PROCESS MESSAGE
# ============================================================

if user_input:

    # ========================================================
    # CREATE CONVERSATION
    # ========================================================

    if (
        st.session_state.current_conversation
        is None
    ):

        title = user_input.strip()


        if len(title) > 50:

            title = (
                title[:50]
                + "..."
            )


        conversation_id = (
            create_conversation(
                title
            )
        )


        st.session_state.current_conversation = (
            conversation_id
        )


    conversation_id = (
        st.session_state.current_conversation
    )


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    save_message(
        conversation_id,
        "user",
        user_input
    )


    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )


    # ========================================================
    # SAVE EXPLICIT MEMORY
    # ========================================================

    detected_memory, memory_category = (
        detect_memory(
            user_input
        )
    )


    if detected_memory:

        memory_id = save_memory(
            detected_memory,
            memory_category
        )


        if memory_id:

            st.toast(
                "🧠 Memory saved",
                icon="🧠"
            )


    # ========================================================
    # EXISTING PERSONAL RAG
    # ========================================================

    try:

        context = retrieve(
            user_input
        )
        context = clip_text(context, 1800)

    except Exception as e:

        context = ""

        st.warning(
            f"RAG retrieval failed: {e}"
        )


    # ========================================================
    # UPLOADED DOCUMENT CONTEXT
    # ========================================================

    uploaded_context = ""


    if st.session_state.uploaded_documents:

        try:

            uploaded_context = (
                get_document_context(
                    user_input,
                    st.session_state.uploaded_documents,
                    max_chunks=2
                )
            )
            uploaded_context = clip_text(uploaded_context, 1800)

        except Exception as e:

            uploaded_context = ""

            st.warning(
                f"Uploaded document search "
                f"failed: {e}"
            )


    # ========================================================
    # LONG-TERM MEMORY
    # ========================================================

    try:

        memory_context = (
            get_memory_context(
                user_input,
                limit=3
            )
        )
        memory_context = clip_text(memory_context, 900)

    except Exception as e:

        memory_context = ""

        st.warning(
            f"Memory retrieval failed: {e}"
        )


    # ========================================================
    # RECENT CONVERSATION
    # ========================================================

    previous_messages = ""

    for message in st.session_state.messages[-4:]:
        msg_text = clip_text(message.get("content", ""), 600)
        previous_messages += (
            f"{message['role'].upper()}: {msg_text}\n"
        )

    previous_messages = clip_text(previous_messages, 1200)


    # ========================================================
    # WEB-ENABLED PROMPT
    # ========================================================

    use_web = should_use_web(user_input) and GROQ_WEB_MODEL is not None

    prompt = f"""
You are Personal AI, a helpful and beginner-friendly assistant.

Rules:
- Answer the user's question directly and clearly.
- Never invent personal information.
- Use personal/document/memory context only when relevant.
- If web search is enabled, use it for current facts and reliable sources.
- Keep answers concise unless the user asks for detail.
- Do not search the web for private personal information.

PERSONAL RAG:
{context}

UPLOADED DOCUMENTS:
{uploaded_context}

LONG-TERM MEMORY:
{memory_context}

RECENT CHAT:
{previous_messages}

USER QUESTION:
{clip_text(user_input, 800)}
"""


    # ========================================================
    # GEMINI + GOOGLE SEARCH
    # ========================================================

    sources = []


    try:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🌐 Searching the web and thinking with Groq..."
                if use_web else "🤖 Thinking..."
            ):

                response = generate_ai_response(prompt, use_web=use_web)

                answer = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )

                if not answer:
                    raise RuntimeError("Groq returned an empty response.")


                st.markdown(
                    answer
                )


                # ------------------------------------------------
                # EXTRACT GOOGLE SOURCES
                # ------------------------------------------------

                sources = (
                    extract_web_sources(
                        response
                    )
                )


                if sources:

                    display_web_sources(
                        sources
                    )


    except Exception as e:

        answer = (
            "Sorry, something went wrong.\n\n"
            f"Error: {e}"
        )


        with st.chat_message(
            "assistant"
        ):

            st.error(
                answer
            )


    # ========================================================
    # SAVE AI RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    save_message(
        conversation_id,
        "assistant",
        answer
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Personal AI • Groq AI • Optional Web Search •
        RAG • File Upload • Long-Term Memory •
        Voice Input • SQLite
    </div>
    """,
    unsafe_allow_html=True
)