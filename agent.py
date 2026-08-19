import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import tools
from rag import retrieve


# ============================================
# LOAD ENVIRONMENT
# ============================================

load_dotenv()


# ============================================
# GET API KEY
# ============================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# ============================================
# CONNECT TO GEMINI
# ============================================

client = genai.Client(
    api_key=api_key
)


# ============================================
# LOAD MEMORY
# ============================================

try:

    with open(
        "memory.json",
        "r",
        encoding="utf-8"
    ) as file:

        memory = json.load(file)

    if not isinstance(memory, list):
        memory = []

except (FileNotFoundError, json.JSONDecodeError):

    memory = []


# ============================================
# CREATE CHAT
# ============================================

chat = client.chats.create(

    model="gemini-3.6-flash",

    config=types.GenerateContentConfig(

        tools=tools
    )
)


# ============================================
# AI AGENT FUNCTION
# ============================================

def ask_agent(user_input):

    global memory


    # ========================================
    # ADD USER MESSAGE TO MEMORY
    # ========================================

    memory.append({

        "role": "user",

        "content": user_input

    })


    # ========================================
    # GET RELEVANT DOCUMENT INFORMATION
    # ========================================

    try:

        relevant_context = retrieve(
            user_input
        )

    except Exception as e:

        relevant_context = ""

        print("RAG error:", e)


    # ========================================
    # GET PREVIOUS MEMORY
    # ========================================

    conversation = ""


    for message in memory:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )

        conversation += (
            f"{role}: {content}\n"
        )


    # ========================================
    # CREATE PROMPT
    # ========================================

    prompt = f"""

You are my personal AI assistant.

You have three important capabilities:

1. Conversation memory
2. Calculator tool
3. RAG document retrieval


========================================
PREVIOUS CONVERSATION
========================================

{conversation}


========================================
RELEVANT INFORMATION FROM MY DOCUMENT
========================================

{relevant_context}


========================================
INSTRUCTIONS
========================================

- Use conversation memory when relevant.
- Use the document information when relevant.
- If the answer comes from the document, use that information.
- Do not invent personal information.
- If the document does not contain the answer, say that you don't
  have that information in the document.
- Use the calculator tool for mathematical calculations.
- Answer naturally and clearly.


========================================
LATEST USER QUESTION
========================================

{user_input}

"""


    # ========================================
    # SEND TO GEMINI
    # ========================================

    try:

        response = chat.send_message(
            prompt
        )


        # ====================================
        # GET RESPONSE
        # ====================================

        answer = response.text


        # ====================================
        # SAVE AI RESPONSE TO MEMORY
        # ====================================

        memory.append({

            "role": "assistant",

            "content": answer

        })


        # ====================================
        # SAVE MEMORY FILE
        # ====================================

        with open(
            "memory.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                memory,
                file,
                indent=4,
                ensure_ascii=False
            )


        return answer


    except Exception as e:

        # Remove failed user message

        if (
            memory
            and memory[-1]["role"] == "user"
        ):

            memory.pop()


        return f"ERROR: {e}"


# ============================================
# TERMINAL CHAT MODE
# ============================================

if __name__ == "__main__":

    print("\n===================================")
    print("        PERSONAL AI AGENT")
    print("===================================")

    print("Gemini      : ON")
    print("Memory      : ON")
    print("Calculator  : ON")
    print("RAG         : ON")

    print("\nType 'exit' to stop.\n")


    while True:

        user_input = input("You: ")


        if user_input.lower() == "exit":

            print("\nGoodbye bro! 👋")

            break


        answer = ask_agent(
            user_input
        )

        print("\nAI:", answer)

        print("\nMemory saved successfully.")