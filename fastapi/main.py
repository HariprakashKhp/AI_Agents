import os
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from ai.gemini import Gemini
from dotenv import load_dotenv
from auth.throttling import apply_rate_limit
from auth.dependencies import get_user_identifier

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

def load_system_prompt():
    with open("./prompts/system_prompt.md", "r") as f:
        return f.read()
    
system_prompt = load_system_prompt()
gemini_api_key = os.getenv("GEMINI_API_KEY")

ai_platform = Gemini(api_key=gemini_api_key, system_prompts=system_prompt)

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_id: str = Depends(get_user_identifier)):
    apply_rate_limit("global_unauthenticated_user")
    response_text = ai_platform.chat(request.prompt)
    return ChatResponse(response=response_text)
 