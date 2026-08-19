from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from database import save_conversation, get_conversations
from rag import retrieve


class State(TypedDict):
    message: str
    history: str
    context: str
    response: str


# ============================================
# LOAD CHAT HISTORY
# ============================================

def load_history(state: State):

    conversations = get_conversations()

    history = ""

    for user_message, ai_response in conversations:

        history += f"User: {user_message}\n"
        history += f"AI: {ai_response}\n\n"

    return {
        "history": history
    }


# ============================================
# RAG RETRIEVAL
# ============================================

def retrieve_context(state: State):

    try:

        context = retrieve(
            state["message"]
        )

    except Exception as e:

        print("RAG error:", e)

        context = ""

    return {
        "context": context
    }


# ============================================
# AI RESPONSE
# ============================================

def generate_response(state: State):

    # Temporary response for testing
    response = (
        f"Your question is: {state['message']}\n\n"
        f"RAG context found:\n{state['context']}"
    )

    return {
        "response": response
    }


# ============================================
# SAVE CONVERSATION
# ============================================

def save_message(state: State):

    save_conversation(
        state["message"],
        state["response"]
    )

    return state


# ============================================
# CREATE GRAPH
# ============================================

builder = StateGraph(State)


builder.add_node(
    "load_history",
    load_history
)

builder.add_node(
    "retrieve_context",
    retrieve_context
)

builder.add_node(
    "generate_response",
    generate_response
)

builder.add_node(
    "save_message",
    save_message
)


# ============================================
# GRAPH FLOW
# ============================================

builder.add_edge(
    START,
    "load_history"
)

builder.add_edge(
    "load_history",
    "retrieve_context"
)

builder.add_edge(
    "retrieve_context",
    "generate_response"
)

builder.add_edge(
    "generate_response",
    "save_message"
)

builder.add_edge(
    "save_message",
    END
)


# ============================================
# COMPILE
# ============================================

graph = builder.compile()


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    result = graph.invoke({

        "message": "What is my internship?",

        "history": "",

        "context": "",

        "response": ""

    })

    print("\n==============================")
    print("LANGGRAPH RESULT")
    print("==============================")

    print(result)