import streamlit as st
from llanggraph_rag_backend import (
    chatbot,
    retrieve_all_threads_with_titles,
    submit_async_task,
)
from llanggraph_rag_backend import (
    ingest_pdf,
    thread_document_metadata,
)
import queue
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# ============================ Page Setup ===========================
st.set_page_config(
    page_title="ThreadAI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* ---------- Global ---------- */
        html, body, [class*="css"] {
            font-family: "Söhne", "Inter", -apple-system, BlinkMacSystemFont,
                "Segoe UI", Helvetica, Arial, sans-serif;
        }

        #MainMenu, footer {visibility: hidden;}

        /* Keep Streamlit's single top-left sidebar toggle available. */
        header[data-testid="stHeader"] {
            background: transparent;
        }

        header[data-testid="stHeader"] button {
            color: #edf5f1;
        }

        .stApp {
                    background: radial-gradient(circle at 12% 0%, #19343a 0, #11191d 36%, #0d1114 100%);
                    color: #edf5f1;
        }

        /* ---------- Center the chat column like ChatGPT ---------- */
        .block-container {
            max-width: 780px;
            padding-top: 2.5rem;
            padding-bottom: 8rem;
            margin: 0 auto;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background-color: #171717;
            border-right: 1px solid #2f2f2f;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }

        section[data-testid="stSidebar"] h1 {
            font-size: 1.15rem;
            font-weight: 600;
            color: #ececec;
            padding-left: 0.2rem;
        }

        section[data-testid="stSidebar"] h2 {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: #8e8e8e;
            margin-top: 1.4rem;
            padding-left: 0.2rem;
        }

        /* New Chat button */
        section[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button {
            background-color: transparent;
            border: 1px solid #3a3a3a;
            color: #ececec;
            border-radius: 10px;
            text-align: left;
            font-weight: 500;
            padding: 0.55rem 0.8rem;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button:hover {
            background-color: #2a2a2a;
            border-color: #4a4a4a;
        }

        /* Conversation list buttons */
        section[data-testid="stSidebar"] button {
            background-color: transparent;
            border: none;
            color: #d9d9d9;
            text-align: left;
            border-radius: 8px;
            font-size: 0.88rem;
            font-weight: 400;
            padding: 0.45rem 0.6rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        section[data-testid="stSidebar"] button:hover {
            background-color: #2a2a2a;
            color: #ffffff;
        }

        /* ---------- Chat title ---------- */
        h1#threadai, .main h1 {
            text-align: center;
            font-size: 1.6rem;
            font-weight: 600;
            color: #ececec;
            margin-bottom: 2rem;
        }

        .threadai-kicker {
            color: #7fd4c5;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            text-align: center;
            margin-bottom: 0.4rem;
        }

        .threadai-subtitle {
            color: #9aada9;
            text-align: center;
            margin: -1.2rem auto 2rem;
            max-width: 34rem;
        }

        /* ---------- Chat bubbles ---------- */
        div[data-testid="stChatMessage"] {
            background: transparent;
            padding: 0.9rem 0;
            border: none;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] span {
            color: #ececec;
            font-size: 1rem;
            line-height: 1.65;
        }

        /* User bubble gets a subtle rounded card, assistant stays flat like ChatGPT */
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
            display: flex;
            justify-content: flex-end;
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) > div:nth-child(2) {
            background-color: #2f2f2f;
            border-radius: 18px;
            padding: 0.6rem 1rem;
            max-width: 75%;
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="chatAvatarIcon-user"] {
            display: none;
        }

        /* ---------- Chat input ---------- */
        div[data-testid="stChatInput"] {
            max-width: 780px;
            margin: 0 auto;
        }

        div[data-testid="stChatInput"] textarea {
            background-color: #2f2f2f !important;
            border: 1px solid #3f3f3f !important;
            border-radius: 24px !important;
            color: #ececec !important;
            padding: 0.75rem 1rem !important;
        }

        div[data-testid="stChatInput"] textarea::placeholder {
            color: #8e8e8e;
        }

        /* ---------- Tool status box ---------- */
        div[data-testid="stStatusWidget"] {
            background-color: #2a2a2a;
            border: 1px solid #3a3a3a;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def truncate_title(text, limit=30):
    text = text.strip()
    if not text:
        return "New Chat"
    return text if len(text) <= limit else text[:limit] + "..."

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id, title="New Chat"):
    # New/most-recently-touched chats always go to the FRONT of the
    # order list so they show up at the top of the sidebar, matching
    # how a normal chat app behaves.
    if thread_id in st.session_state["chat_order"]:
        st.session_state["chat_order"].remove(thread_id)
    st.session_state["chat_order"].insert(0, thread_id)
    st.session_state["chat_titles"].setdefault(thread_id, title)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])

def extract_text(content):
    """Normalize AIMessage content into a plain string.

    `content` can be:
      - a plain string (older / simple models)
      - a list of content blocks, e.g.
        [{"type": "text", "text": "..."}, {"type": "signature", ...}]
        where only "text" blocks should be shown to the user. Tool-call
        deltas, thinking blocks, and signature/citation metadata blocks
        should be skipped entirely.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # anything else (tool_use, signature, thinking, etc.) is skipped
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)

    return ""

def escape_dollars(text):
    """Escape literal '$' so Streamlit's markdown renderer doesn't try to
    interpret price strings like $225.16 ... $0.14 as LaTeX math (which is
    what causes garbled output when two or more $ appear in a message)."""
    return text.replace("$", "\\$")

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_titles" not in st.session_state or "chat_order" not in st.session_state:
    # retrieve_all_threads_with_titles() already returns threads ordered
    # newest-first with a proper heading (the first user message) instead
    # of a raw thread id, so we can use its key order directly.
    titles = retrieve_all_threads_with_titles()
    st.session_state["chat_titles"] = titles
    st.session_state["chat_order"] = list(titles.keys())

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("ThreadAI")

st.sidebar.caption("A focused workspace for thinking with your sources.")

if st.sidebar.button("📝 New Chat", use_container_width=True):
    reset_chat()

st.sidebar.header("Source document")
uploaded_pdf = st.sidebar.file_uploader(
    "Add a PDF to this thread",
    type=["pdf"],
    help="ThreadAI will index this document so you can ask grounded questions about it.",
)
if uploaded_pdf is not None:
    upload_key = f"{st.session_state['thread_id']}:{uploaded_pdf.name}:{uploaded_pdf.size}"
    if st.session_state.get("indexed_upload") != upload_key:
        with st.sidebar.status("Indexing document...", expanded=False) as status:
            try:
                metadata = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    st.session_state["thread_id"],
                    uploaded_pdf.name,
                )
                st.session_state["indexed_upload"] = upload_key
                status.update(label="Document ready", state="complete")
            except Exception as exc:
                status.update(label="Could not index document", state="error")
                st.sidebar.error(str(exc))

document_metadata = thread_document_metadata(st.session_state["thread_id"])
if document_metadata:
    st.sidebar.success(
        f"{document_metadata['filename']} · {document_metadata['chunks']} searchable sections"
    )

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_order"]:
    label = st.session_state["chat_titles"].get(thread_id, "New Chat")
    is_active = thread_id == st.session_state["thread_id"]
    button_label = f" 🔹{label}" if is_active else f"{label}"
    if st.sidebar.button(button_label, key=f"thread-{thread_id}", use_container_width=True):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": extract_text(msg.content)})
        st.session_state["message_history"] = temp_messages

        # Move this thread to the top since it's now the active one.
        add_thread(thread_id, st.session_state["chat_titles"].get(thread_id, "New Chat"))
        st.rerun()

# ============================ Main UI ============================
st.markdown(
    """
    <style>
    .threadai-hero {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
    }

    .threadai-brand {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-bottom: 0.6rem;
    }

    .threadai-brand span {
        font-weight: 400;
    }

    .threadai-subtitle {
        font-size: 1.05rem;
        opacity: 0.7;
        margin: 0 auto;
        max-width: 650px;
        line-height: 1.6;
    }

    .threadai-capabilities {
        text-align: center;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        opacity: 0.5;
        margin-top: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>

    <div class="threadai-hero">
        <div class="threadai-brand">
            THREAD<span>AI</span>
        </div>
    </div>
        <div class="threadai-subtitle">
            Your intelligent workspace for conversations, web search, and documents.
        </div>
    

    <div class="threadai-capabilities">
        CHAT&nbsp;&nbsp;·&nbsp;&nbsp;WEB SEARCH&nbsp;&nbsp;·&nbsp;&nbsp;DOCUMENTS
    </div>
    """,
    unsafe_allow_html=True,
)

if document_metadata:
    st.info(
        f"Grounded mode is active for **{document_metadata['filename']}** · "
        f"{document_metadata['documents']} pages · {document_metadata['chunks']} searchable sections"
    )

# Render history
for message in st.session_state["message_history"]:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(escape_dollars(message["content"]))

user_input = st.chat_input("Type here")

if user_input:
    # If this is the first message in a brand-new thread, give the
    # sidebar entry a real heading right away instead of "New Chat".
    is_new_thread = len(st.session_state["message_history"]) == 0
    if is_new_thread:
        st.session_state["chat_titles"][st.session_state["thread_id"]] = truncate_title(user_input)

    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(escape_dollars(user_input))

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant", avatar="🤖"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}
        # Accumulate the RAW (unescaped) text here so we can save the true
        # content to session state — the generator itself yields an escaped
        # version purely for display, so we don't want to save the escaped
        # ("\$") text into history.
        raw_parts = []

        def ai_only_stream():
            chunk_queue = queue.Queue()
            _DONE = object()

            async def produce():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        chunk_queue.put((message_chunk, metadata))
                except Exception as e:
                    chunk_queue.put(e)
                finally:
                    chunk_queue.put(_DONE)

            # Runs the async generator on the backend's dedicated event loop,
            # while this (sync) generator just blocks on the queue and yields
            # items to Streamlit as they arrive.
            submit_async_task(produce())

            while True:
                item = chunk_queue.get()

                if item is _DONE:
                    break

                if isinstance(item, Exception):
                    raise item

                message_chunk, metadata = item

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    text = extract_text(message_chunk.content)
                    if text:
                        raw_parts.append(text)
                        yield escape_dollars(text)
        st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save the raw (unescaped) assistant message
    ai_message = "".join(raw_parts)
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # This thread just got activity — keep it pinned at the top of the
    # sidebar, matching normal chat-app behavior.
    add_thread(
        st.session_state["thread_id"],
        st.session_state["chat_titles"].get(st.session_state["thread_id"], "New Chat"),
    )

    if is_new_thread:
        st.rerun()