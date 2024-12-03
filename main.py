import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import requests

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cohere API configuration
COHERE_API_URL = "https://api.cohere.com/v1/chat"
COHERE_API_KEY = "ZKeLJ5aohZD59ZjvnepKP9BYh8SDAeuIxM8IPCIs"

# Bot-specific prompts
BOT_PROMPTS = {
    "NEXUS": "You are NEXUS, an analytical AI specializing in data-driven insights. Break down the user's query and provide a logical response.",
    "ATLAS": "You are ATLAS, a strategic AI. Consider long-term impacts and provide visionary guidance to the user's query.",
    "CIPHER": "You are CIPHER, a technical AI expert. Focus on precise and detailed answers related to technical matters.",
    "COGNIS": "You are COGNIS, a collaborative AI. Integrate the responses from other bots into a single, cohesive response."
}

# Request schema
class ChatRequest(BaseModel):
    message: str
    otherResponses: str = None  # For COGNIS

# Response schema
class ChatResponse(BaseModel):
    text: str
    chat_history: list
    finish_reason: str

# Function to make a synchronous request to Cohere API
def fetch_cohere_response(message: str, preamble: str, model: str = "command-r-08-2024") -> Dict:
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "message": message,
        "model": model,
        "preamble": preamble
    }
    try:
        response = requests.post(COHERE_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Bot-specific endpoints
@app.post("/api/nexus", response_model=ChatResponse)
def chat_nexus(request: ChatRequest):
    preamble = BOT_PROMPTS["NEXUS"]
    data = fetch_cohere_response(request.message, preamble)
    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )

@app.post("/api/atlas", response_model=ChatResponse)
def chat_atlas(request: ChatRequest):
    preamble = BOT_PROMPTS["ATLAS"]
    data = fetch_cohere_response(request.message, preamble)
    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )

@app.post("/api/cipher", response_model=ChatResponse)
def chat_cipher(request: ChatRequest):
    preamble = BOT_PROMPTS["CIPHER"]
    data = fetch_cohere_response(request.message, preamble)
    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )

@app.post("/api/cognis", response_model=ChatResponse)
def chat_cognis(request: ChatRequest):
    if not request.otherResponses:
        raise HTTPException(status_code=400, detail="Missing otherResponses in request.")

    preamble = f"{BOT_PROMPTS['COGNIS']}\nHere are the responses from other AIs:\n{request.otherResponses}"
    data = fetch_cohere_response(request.message, preamble)

    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )

# Start the application
if __name__ == "__main__":
    # Bind to the port specified by Render or default to 8000
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
