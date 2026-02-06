import os
import datetime
from typing import List
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from cryptography.fernet import Fernet
from google import genai

# --- SECURITY & ENCRYPTION SETUP ---
# Use a key from env for encryption/decryption
ENCR_KEY = os.environ.get("SECRET_ENCRYPTION_KEY", Fernet.generate_key().decode())
fernet = Fernet(ENCR_KEY.encode())

def encrypt_data(data: str) -> str:
    return fernet.encrypt(data.encode()).decode()

def decrypt_data(data: str) -> str:
    return fernet.decrypt(data.encode()).decode()

# --- DATABASE SETUP (POSTGRESQL) ---
DB_URL = os.environ.get("DATABASE_URL").replace("postgres://", "postgresql://", 1)
engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "oxo_memory"
    id = Column(Integer, primary_key=True, index=True)
    conv_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text) # Stored encrypted
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- FASTAPI & GEMINI SDK SETUP ---
app = FastAPI(title="OXO AI Professional")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["POST"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    conv_id: str
    prompt: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- THE FALLCHAIN SYSTEM ---
# Tries 3 -> 2.5 Lite -> 1.5
FALLCHAIN = ["gemini-3-flash-preview","gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-flash"]

async def execute_fallchain(history_context: List[dict]):
    for model_id in FALLCHAIN:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=history_context
            )
            return response.text, model_id
        except Exception as e:
            print(f"Fallchain Warning: {model_id} failed. Error: {e}")
            continue
    return None, None

# --- API ROUTES ---
@app.post("/oxo/generate")
async def oxo_generate(req: ChatRequest, db: Session = Depends(get_db)):
    # 1. Fetch and Decrypt Memory
    past_messages = db.query(ChatMessage).filter(ChatMessage.conv_id == req.conv_id).order_by(ChatMessage.timestamp).all()
    
    # Reconstruct History for Gemini
    history_context = []
    for msg in past_messages:
        history_context.append({
            "role": msg.role,
            "parts": [{"text": decrypt_data(msg.content)}]
        })
    
    # Add current user prompt
    history_context.append({"role": "user", "parts": [{"text": req.prompt}]})

    # 2. Execute Fallchain logic
    ai_response, model_used = await execute_fallchain(history_context)
    
    if not ai_response:
        raise HTTPException(status_code=503, detail="All AI Fallchain tiers failed.")

    # 3. Save to Permanent Encrypted Memory
    # Save User Message
    db.add(ChatMessage(conv_id=req.conv_id, role="user", content=encrypt_data(req.prompt)))
    # Save AI Message
    db.add(ChatMessage(conv_id=req.conv_id, role="model", content=encrypt_data(ai_response)))
    db.commit()

    return {
        "status": "success",
        "model": model_used,
        "conversation_id": req.conv_id,
        "output": ai_response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))