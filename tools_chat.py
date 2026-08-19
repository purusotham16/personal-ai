import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import calculator


# Load environment variables
load_dotenv()


# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Define calculator tool
calculator_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="calculator",
            description="Calculate a mathematical expression.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "expression": types.Schema(
                        type="STRING",
                        description="A mathematical expression such as 25 * 40"
                    )
                },
                required=["expression"]
            )
        )
    ]
)


# Tool configuration
config = types.GenerateContentConfig(
    tools=[calculator_tool]
)


# Ask user
user_input = input("You: ")


# Send request to Gemini
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=user_input,
    config=config
)


# Check whether Gemini requested a tool
if response.function_calls:

    function_call = response.function_calls[0]

    print("🔧 Tool requested:", function_call.name)

    # Get arguments
    expression = function_call.args["expression"]

    # Execute calculator
    result = calculator(expression)

    print("🔧 Calculator result:", result)

else:

    print("AI:", response.text)