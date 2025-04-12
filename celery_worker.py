import os
from celery import Celery
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import User, ChatSession, ChatMessage, Base, SessionLocal  # reuse models

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

celery_app = Celery(
    "worker",
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_RESULT_BACKEND")
)

@celery_app.task
def save_chat_session_task(data):
    db = SessionLocal()
    try:
        session = ChatSession(**data)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session.id
    finally:
        db.close()

@celery_app.task
def save_chat_message_task(data):
    db = SessionLocal()
    try:
        msg = ChatMessage(**data)
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg.id
    finally:
        db.close()