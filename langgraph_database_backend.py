from langchain_protocol import Checkpoint
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

import sqlite3

# from streamlit_frontend_threading import CONFIG #we need to import sqlite3 to use the SqliteSaver

load_dotenv()

llm = init_chat_model(
    "google_genai:gemini-3.6-flash",
    max_retries=10,  # Increase for unreliable networks (default: 6)
    timeout=120,  # Seconds; increase for slow connections
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False) #this line creates a connection to the SQLite database named 'chatbot.db'. The check_same_thread=False argument allows the connection to be used across different threads, which is useful in multi-threaded applications.
# Checkpointer
checkpointer = SqliteSaver(conn=conn) #this line creates an instance of the SqliteSaver class, passing the SQLite connection object conn as an argument. This checkpointer will handle saving and loading the state of the chatbot to and from the SQLite database.

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
# this function retrieves all unique thread IDs from the checkpoints stored in the SQLite database. It iterates through all checkpoints using the checkpointer's list method, extracts the thread_id from each checkpoint's configuration, and adds it to a set to ensure uniqueness. Finally, it returns a list of all unique thread IDs.



# testing part:
# ... keep everything above as-is (llm, ChatState, chat_node, conn, checkpointer, graph, chatbot, retrieve_all_threads) ...

# if __name__ == "__main__":
#     # testing part — only runs when you execute this file directly,
#     # e.g. `python langgraph_backend.py`, not when Streamlit imports it
#     CONFIG = {
#         "configurable": {
#             "thread_id": "thread-1"
#         }
#     }

#     response = chatbot.invoke(
#         {"messages": [HumanMessage(content="Hi, shubha this side")]},
#         config=CONFIG,
#     )
#     print("Response from chatbot:", response)


# all_threads = set()
# for Checkpoint in checkpointer.list(None):  # List all checkpoints in the database
#     all_threads.add(Checkpoint.config['configurable']['thread_id'])
# print("All unique thread IDs:", list(all_threads))


def retrieve_all_threads_with_titles():
    """
    Returns a dict of {thread_id: title}, ordered from most recently
    updated thread to least recently updated.
    """
    thread_data = {}  # thread_id -> {"messages": [...], "ts": checkpoint_timestamp}

    for checkpoint_tuple in checkpointer.list(None):
        thread_id = checkpoint_tuple.config['configurable']['thread_id']
        messages = checkpoint_tuple.checkpoint.get('channel_values', {}).get('messages', [])
        ts = checkpoint_tuple.checkpoint.get('ts')  # ISO timestamp string

        existing = thread_data.get(thread_id)
        if existing is None or len(messages) > len(existing["messages"]):
            thread_data[thread_id] = {"messages": messages, "ts": ts}
        # keep the max ts seen for this thread regardless
        if existing is not None and ts and (not existing["ts"] or ts > existing["ts"]):
            thread_data[thread_id]["ts"] = ts

    # sort thread_ids by ts, most recent first
    sorted_thread_ids = sorted(
        thread_data.keys(),
        key=lambda tid: thread_data[tid]["ts"] or "",
        reverse=True,
    )

    thread_titles = {}
    for thread_id in sorted_thread_ids:
        messages = thread_data[thread_id]["messages"]
        title = "New Chat"
        for message in messages:
            if isinstance(message, HumanMessage):
                content = message.content
                title = content if len(content) <= 24 else content[:24] + "..."
                break
        thread_titles[thread_id] = title

    return thread_titles