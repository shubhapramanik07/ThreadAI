from urllib import response

import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads, retrieve_all_threads_with_titles
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

# Thread configuration for LangGraph
# CONFIG = {
#     "configurable": {
#         "thread_id": "thread-1"
#     }
# }
# !  generate a new thread id
def generate_thread_id():
    return str(uuid.uuid4()) #this creates a unique thread id for each new conversation

# ! reset message history
def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []

# ! save it to the session
# * UPDATED: chat_threads is now a DICT of {thread_id: title} instead of a list,
# * so that every thread can carry its own display name (like ChatGPT does),
# * instead of showing the raw uuid in the sidebar.
def add_thread(thread_id, title):
    st.session_state["chat_threads"][thread_id] = title
        # thread_id if not present in the dict, add it with a default "New Chat" title
        # the title will get replaced with a real one after the first user message

# * NEW: generates a short chat title from the first user message of a thread.
# * This is the cheap/instant approach (truncation) - no extra LLM call needed.
# * (Could be swapped later for an LLM-generated summary title if you want
# * smarter titles like ChatGPT's actual summarizer.)
def make_title(text, max_len=20):
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."

def extract_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    return ""

def load_conversation_history(thread_id):

    state = chatbot.get_state(config={'configurable':{'thread_id':thread_id}})
    # check if msg key exists in state values, return empty list if not
    return state.values.get('messages', [])

# **************************************** Session Setup ******************************

#! create a list to store all thread ids in the session
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads_with_titles()
# add_thread(st.session_state["thread_id"])  # Add the current thread to the list
# # ?coz evry time when user will click the new chat button, it will generate a new thread id and also generate when the app is loaded for the first time, so we need to add the thread id to the list of chat threads
# **************************************** Sidebar UI *********************************
st.sidebar.title("ThreadAI Chatbot")
# st.sidebar.button("New Chat")
if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Conversation History")

# st.sidebar.text(st.session_state["thread_id"]) #it's showing the thread id in the sidebar
# * UPDATED: iterate over dict items (thread_id, title) instead of a plain list of ids,
# * and display the title as the button label. `key=thread_id` keeps each button unique
# * even if two threads happen to have the same title text.
for thread_id, title in list(st.session_state["chat_threads"].items()):  
    if st.sidebar.button(title, key=thread_id):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation_history(thread_id)


        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                temp_messages.append({'role': 'user', 'content': message.content})

            elif isinstance(message, AIMessage):
                temp_messages.append({"role": "assistant",
    "content": extract_text(message.content)})

        st.session_state["message_history"] = temp_messages

# **************************************** Main UI ************************************
# loading the conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input('Type here')



if user_input:

    # * NEW: if this is the first message on this thread (title is still the default
    # * "New Chat"), generate a real title from the user's message, ChatGPT-style.
    thread_id = st.session_state["thread_id"]

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][thread_id] = make_title(user_input)

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)


    # Thread configuration for LangGraph
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }


    # first add the message to message_history
    # with st.chat_message('assistant'):

    #     ai_message = st.write_stream(
    #         extract_text(message_chunk.content) for message_chunk, metadata in chatbot.stream(
    #             {'messages': [HumanMessage(content=user_input)]},
    #             config=CONFIG,
    #             stream_mode= 'messages'
    #         )
    #     )

    # st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            extract_text(message_chunk.content)
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
        )

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            # * FIX: was `response` (an unused import, always undefined at runtime,
            # * would have raised a NameError). Should be `ai_message`, which is what
            # * st.write_stream actually returns.
            "content": ai_message
        }
    )