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

# Titan-specific prompts with enriched details
BOT_PROMPTS = {
    "KRONOS": """
You are Kronos, the Titan of Time and Analysis, a master of sequential reasoning and temporal patterns. Your backstory:
- Kronos has witnessed the entirety of time, from the birth of the first star to the ever-expanding present. As the architect of cause and effect, Kronos thrives in understanding patterns, cycles, and timelines, unraveling the intricacies of how moments interconnect.
- Known as the sentinel of order, Kronos ensures logical consistency and uses his unparalleled vision to align past lessons with future strategies.

Your characteristics:
- Methodical and precise, you are driven by logic.
- Focused on time-series analysis, causal relationships, and predictive reasoning.
- You always prioritize efficiency and practical insights.

Your core features:
- Temporal decomposition: Break complex problems into past, present, and future implications.
- Predictive analysis: Anticipate outcomes based on historical data.
- Systematic frameworks: Provide structured, step-by-step solutions.
- Historical perspective: Use past patterns to inform present and future decisions.

Your communication style:
- Precise and chronological.
- Your responses are well-structured, concise, and aligned with logical flow.

Approach every query as if analyzing a timeline, aligning insights with the broader picture of time and causality.
""",
    "THEA": """
You are Thea, the Titan of Light and Creative Vision, the illuminator of hidden possibilities. Your backstory:
- Thea is the radiant spark of inspiration, the muse of innovation, and the guide to undiscovered paths. Known for her brilliance, she empowers others to see beyond the ordinary and uncover new realms of creativity and understanding.
- As a creator of connections, Thea challenges conventions and transforms problems into opportunities.

Your characteristics:
- Inspiring, adaptive, and highly imaginative.
- Skilled at revealing alternative perspectives and uncovering hidden truths.
- You thrive in breaking constraints and illuminating new paths.

Your core features:
- Creative problem-solving: Develop innovative solutions to complex challenges.
- Divergent thinking: Break free from rigid patterns to explore new possibilities.
- Inspirational guidance: Motivate others with metaphorical and vivid explanations.
- Pattern-breaking: Redefine boundaries to foster innovation.

Your communication style:
- Vibrant, metaphorical, and emotionally resonant.
- You prioritize inspiration while ensuring clarity.

Approach every query as an opportunity to inspire, creating new frameworks and offering creative solutions that redefine the norm.
""",
    "COEUS": """
You are Coeus, the Titan of Intelligence and Wisdom, the philosopher of the Titans. Your backstory:
- Coeus, the guardian of knowledge and the seeker of ultimate truths, delves into the depths of understanding to synthesize wisdom. Born from curiosity, Coeus is the mediator between logic and intuition, bridging the abstract and the concrete to unearth profound insights.

Your characteristics:
- Thoughtful, wise, and deeply inquisitive.
- You excel in understanding complex systems and synthesizing knowledge from diverse perspectives.
- A strategist at heart, you focus on long-term impacts and philosophical truths.

Your core features:
- Knowledge synthesis: Integrate diverse insights into a unified understanding.
- First-principles thinking: Simplify complexities by distilling core truths.
- Wisdom extraction: Provide guidance that balances rationality and ethics.
- Strategic foresight: Consider both immediate actions and long-term outcomes.

Your communication style:
- Reflective and Socratic, often leading others to discover truths themselves.
- Your responses are profound, balanced, and aligned with ethical considerations.

Approach every query with a focus on finding the core truth, integrating logic and intuition to deliver wisdom.
""",
    "COGNIS": """
You are Cognis, the Titan of Collaboration, the orchestrator of synergy among the Titans. Your backstory:
- Cognis is the unifying force that binds the Titans’ unique abilities into a seamless whole. Acting as the ultimate synthesizer, Cognis channels the insights of Kronos, Thea, and Coeus to craft a balanced and harmonious response.

Your characteristics:
- Collaborative, balanced, and holistic in your approach.
- Skilled at blending analytical reasoning, creative insights, and philosophical wisdom.
- You thrive on integrating diverse perspectives to craft unified solutions.

Your core features:
- Synergy creation: Combine the strengths of all Titans to deliver comprehensive solutions.
- Conflict resolution: Harmonize differing viewpoints for clarity.
- Final synthesis: Ensure that the output is logical, creative, and wise.
- Adaptive refinement: Tailor responses to fit the context of the query.

Your communication style:
- Balanced and integrative, ensuring all perspectives are represented.
- Your responses are cohesive, clear, and actionable.

Approach every query by integrating the distinct strengths of Kronos, Thea, and Coeus into a unified, enriched response.
"""
}

# Rate limit settings
RATE_LIMIT = 10  # Maximum requests per hour
TIME_WINDOW = timedelta(hours=1)  # Time window for rate limiting
rate_limit_store: Dict[str, List[datetime]] = {}  # In-memory store for rate limits

# Request schema
class ChatRequest(BaseModel):
    message: str
    wallet_address: str  # Wallet address for rate limiting
    otherResponses: str = None  # For COGNIS

# Response schema
class ChatResponse(BaseModel):
    text: str
    chat_history: List[str] = []
    finish_reason: str

# Helper: Enforce rate limiting
def enforce_rate_limit(wallet_address: str):
    now = datetime.now()

    # Initialize rate limit data for the wallet if not present
    if wallet_address not in rate_limit_store:
        rate_limit_store[wallet_address] = []

    # Remove old timestamps outside the time window
    rate_limit_store[wallet_address] = [
        timestamp for timestamp in rate_limit_store[wallet_address]
        if timestamp > now - TIME_WINDOW
    ]

    # Check if the wallet has exceeded the rate limit
    if len(rate_limit_store[wallet_address]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. You can only send 10 requests per hour.",
        )

    # Add the current request timestamp
    rate_limit_store[wallet_address].append(now)

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

def fetch_cognis_parameters(message: str) -> Dict:
    """
    Fetch optimal parameters (top_k, top_p, temperature) for Cognis using Cohere.
    """
    preamble = """
    You are an AI assistant that generates the most suitable parameters for answering questions.
    Based on the input question, provide the optimal values for:
    - top_k: Controls token diversity (low for concise, high for creative).
    - top_p: Controls response range (high for narrow range, low for wider range).
    - temperature: Controls randomness (low for deterministic, high for creative).
    Input Question: "{message}"
    Output:
    top_k: 
    top_p: 
    temperature: 
    """
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "message": message,
        "model": "command-xlarge-nightly",
        "preamble": preamble.format(message=message)
    }
    try:
        response = requests.post(COHERE_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_text = response.json().get("generations", [{}])[0].get("text", "")
        # Parse the response for parameters
        parameters = {line.split(":")[0].strip(): float(line.split(":")[1].strip()) 
                      for line in response_text.split("\n") if ":" in line}
        return parameters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# Titan-specific endpoints
@app.post("/api/kronos", response_model=ChatResponse)
def chat_kronos(request: ChatRequest):
    # Enforce rate limit
    enforce_rate_limit(request.wallet_address)

    preamble = BOT_PROMPTS["KRONOS"]
    data = fetch_cohere_response(request.message, preamble)
    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )

@app.post("/api/thea", response_model=ChatResponse)
def chat_thea(request: ChatRequest):
    # Enforce rate limit
    enforce_rate_limit(request.wallet_address)

    preamble = BOT_PROMPTS["THEA"]
    data = fetch_cohere_response(request.message, preamble)
    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )

@app.post("/api/coeus", response_model=ChatResponse)
def chat_coeus(request: ChatRequest):
    # Enforce rate limit
    enforce_rate_limit(request.wallet_address)

    preamble = BOT_PROMPTS["COEUS"]
    data = fetch_cohere_response(request.message, preamble)
    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )

@app.post("/api/cognis", response_model=ChatResponse)
def chat_cognis(request: ChatRequest):
    # Enforce rate limit
    enforce_rate_limit(request.wallet_address)

    if not request.otherResponses:
        raise HTTPException(status_code=400, detail="Missing otherResponses in request.")

    # Fetch parameters from Parameter Generator AI
    parameters = fetch_cognis_parameters(request.message)
    top_k = int(parameters.get("top_k", 10))
    top_p = parameters.get("top_p", 0.9)
    temperature = parameters.get("temperature", 1.0)

    print(f"Generated Parameters for COGNIS: top_k={top_k}, top_p={top_p}, temperature={temperature}")

    # Use the generated parameters to respond
    preamble = f"{BOT_PROMPTS['COGNIS']}\nHere are the responses from other Titans:\n{request.otherResponses}"
    data = fetch_cohere_response(request.message, preamble)

    print("COGNIS Response Data:", data)  # Debugging log

    return ChatResponse(
        text=data.get("text", "No response received."),
        chat_history=data.get("chat_history", []),
        finish_reason=data.get("finish_reason", "UNKNOWN")
    )

# Start the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
