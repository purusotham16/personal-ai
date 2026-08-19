from fastapi import FastAPI
from pydantic import BaseModel

from graph import graph


app = FastAPI(
    title="Personal AI API"
)


class ChatRequest(BaseModel):

    message: str


@app.get("/")
def home():

    return {
        "message": "Personal AI API is running!"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    result = graph.invoke({

        "message": request.message

    })


    return {

        "response": result["response"]

    }