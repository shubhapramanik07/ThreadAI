import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# Thread configuration for LangGraph
CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}

# Initialize chat history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Display previous messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Type here")

# Only execute when the user sends a message
if user_input is not None and user_input.strip():

    # Debugging
    print("User Input:", repr(user_input))

    # Store user message
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # ================= WRONG POSITION =================
    # This was outside the if block:
    #
    # response = chatbot.invoke(
    #     {"messages": [HumanMessage(content=user_input)]},
    #     config=CONFIG
    # )
    #
    # When Streamlit first loads,
    # user_input = None
    # which becomes
    # HumanMessage(content=None)
    # and raises ValidationError.
    # ================================================

    # Call LangGraph chatbot
    response = chatbot.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=CONFIG
    )

    # Get assistant response
    # ai_message = response["messages"][-1].content
    ai_message = response["messages"][-1].content[0]["text"]

    print("AI Response:", ai_message)
    # Display assistant message
    with st.chat_message("assistant"):
        st.write(ai_message)
        
    # Save assistant message
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )

    