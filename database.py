import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# 1. FIX: Render provides "postgres://", but SQLAlchemy 1.4+ needs "postgresql://"
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 2. HIGH SECURITY: Create engine with pessimistic disconnect handling
# pool_pre_ping=True checks if the connection is alive before every query
engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True, 
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 3. ENCRYPTED TABLE SCHEMA
class ChatMessage(Base):
    __tablename__ = "oxo_memory"
    id = Column(Integer, primary_key=True, index=True)
    conv_id = Column(String, index=True) # Permanent session ID
    role = Column(String)                # 'user' or 'model'
    content = Column(Text)               # THIS WILL HOLD ENCRYPTED DATA
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# 4. AUTO-CREATE TABLES ON STARTUP
Base.metadata.create_all(bind=engine)