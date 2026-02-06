import os
import datetime
import base64
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from Crypto.Cipher import AES # From pycryptodome

# --- SECURITY: AES-256 GCM ENCRYPTION ---
# This key must be exactly 32 bytes for AES-256
RAW_KEY = os.environ.get("SECRET_ENCRYPTION_KEY", "12345678901234567890123456789012").encode()

def encrypt_data(plain_text: str) -> str:
    cipher = AES.new(RAW_KEY, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(plain_text.encode())
    # Store: Nonce + Tag + Ciphertext
    return base64.b64encode(cipher.nonce + tag + ciphertext).decode()

def decrypt_data(encrypted_text: str) -> str:
    data = base64.b64decode(encrypted_text)
    nonce, tag, ciphertext = data[:16], data[16:32], data[32:]
    cipher = AES.new(RAW_KEY, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag).decode()

# --- DATABASE SETUP ---
DB_URL = os.environ.get("DATABASE_URL", "sqlite:///oxo_local.db") # Local sqlite for Pydroid testing
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "oxo_memory"
    id = Column(Integer, primary_key=True, index=True)
    conv_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text) 
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)
