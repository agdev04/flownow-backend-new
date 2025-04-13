from dotenv import load_dotenv
load_dotenv()
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_recycle=280, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String(128), unique=True, index=True)
    email = Column(String(255), unique=True, index=True)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    created_at = Column(String(64), default=lambda: datetime.datetime.utcnow().isoformat())
    title = Column(String(255), default="Chat Session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    user_id = Column(Integer, index=True)
    role = Column(String(32))
    message = Column(String(4096))
    created_at = Column(String(64), default=lambda: datetime.datetime.utcnow().isoformat())
    fault = Column(Integer, default=0)  # store as 0/1 boolean

Base.metadata.create_all(bind=engine)