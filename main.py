from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import requests

app = FastAPI()

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

# Memory for storing individual AI responses
ai_responses: Dict[str, str] = {}


# Request schema
class ChatRequest(BaseModel):
    message: str


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
    ai_responses["NEXUS"] = data.get("text", "No response received.")
    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )


@app.post("/api/atlas", response_model=ChatResponse)
def chat_atlas(request: ChatRequest):
    preamble = BOT_PROMPTS["ATLAS"]
    data = fetch_cohere_response(request.message, preamble)
    ai_responses["ATLAS"] = data.get("text", "No response received.")
    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )


@app.post("/api/cipher", response_model=ChatResponse)
def chat_cipher(request: ChatRequest):
    preamble = BOT_PROMPTS["CIPHER"]
    data = fetch_cohere_response(request.message, preamble)
    ai_responses["CIPHER"] = data.get("text", "No response received.")
    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )


@app.post("/api/cognis", response_model=ChatResponse)
def chat_cognis(request: ChatRequest):
    if len(ai_responses) < 3:
        raise HTTPException(status_code=400, detail="Not all AI responses are ready.")

    # Combine responses from NEXUS, ATLAS, and CIPHER
    other_responses = "\n- ".join([f"{key}: {value}" for key, value in ai_responses.items()])
    preamble = f"{BOT_PROMPTS['COGNIS']}\nHere are the responses from other AIs:\n- {other_responses}"
    data = fetch_cohere_response(request.message, preamble)

    # Clear responses after processing
    ai_responses.clear()

    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )
