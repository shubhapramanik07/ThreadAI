import streamlit as st
from langgraph_tool_backend import (
    chatbot,
    retrieve_all_threads_with_titles,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# ============================ Page Setup ===========================
st.set_page_config(
    page_title="ThreadAI",
    page_icon="🧵",
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

        #MainMenu, footer, header {visibility: hidden;}

        .stApp {
            background-color: #212121;
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
    return uuid.uuid4()

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

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_order"]:
    label = st.session_state["chat_titles"].get(thread_id, "New Chat")
    is_active = thread_id == st.session_state["thread_id"]
    button_label = f"🟢 {label}" if is_active else f"{label}"
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
st.title("ThreadAI")

# Render history
for message in st.session_state["message_history"]:
    avatar = "🧑" if message["role"] == "user" else "🧵"
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
    with st.chat_message("assistant", avatar="🧵"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}
        # Accumulate the RAW (unescaped) text here so we can save the true
        # content to session state — the generator itself yields an escaped
        # version purely for display, so we don't want to save the escaped
        # ("\$") text into history.
        raw_parts = []

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
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

                # Stream ONLY assistant tokens, and only real text content.
                # message_chunk.content can be a list of content blocks
                # (text / tool_use / signature / thinking) rather than a
                # plain string, so we need to pull just the text out —
                # otherwise raw block dicts (and empty [] chunks) get
                # printed straight into the chat.
                if isinstance(message_chunk, AIMessage):
                    text = extract_text(message_chunk.content)
                    if text:
                        raw_parts.append(text)
                        # Escape $ signs here so Streamlit's live markdown
                        # renderer doesn't treat "$225.16 ... $0.14" as a
                        # LaTeX math block mid-stream.
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