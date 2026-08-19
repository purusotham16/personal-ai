import os
import json

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Load saved memory
with open("memory.json", "r") as file:
    memory = json.load(file)

# System instruction
system_prompt = f"""
You are a helpful personal AI assistant.

The user's saved memory is:

{memory}

Use this information when it is relevant.
Do not invent memories that are not provided.
"""

# Chat history
chat_history = []

print("🤖 Personal AI started!")
print("Type 'exit' to stop.")
print("Type 'remember <information>' to save something.\n")

while True:

    # Get user's question
    user_input = input("You: ")

    # Exit
    if user_input.lower() == "exit":
        print("AI: Chat ended.")
        break

    # Save memory
    if user_input.lower().startswith("remember "):

        information = user_input[9:]

        memory["personal_info"].append(information)

        with open("memory.json", "w") as file:
            json.dump(memory, file, indent=4)

        print("AI: I'll remember that.")
        continue

    # Add user message to chat history
    chat_history.append({
        "role": "user",
        "parts": [{"text": user_input}]
    })

    # Send conversation to Gemini
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=chat_history,
        config={
            "system_instruction": system_prompt
        }
    )

    # Display response
    print("AI:", response.text)

    # Add AI response to history
    chat_history.append({
        "role": "model",
        "parts": [{"text": response.text}]
    })