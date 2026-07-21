import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, Integer, Text, DateTime,
    ForeignKey, JSON, Enum as SAEnum, Boolean,
)

from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


def _now() -> datetime:
    return datetime.now(timezone.utc)

def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_now)
    email      = Column(String(255), unique=True, nullable=True)
    name       = Column(String(255), nullable=True)
    plan       = Column(String(50), default="free", nullable=False)
    api_key    = Column(String(64), unique=True, nullable=True)  # Option B auth key

    runs          = relationship("Run", back_populates="user", lazy="dynamic")
    chat_sessions = relationship("ChatSession", back_populates="user", lazy="dynamic")



class Run(Base):
    __tablename__ = "runs"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    question = Column(Text, nullable=True)
    status = Column(
        SAEnum("running", "done", "failed", name="run_status"),
        default="running", nullable=False,
    )
    created_at = Column(DateTime(timezone=True), default=_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    #Path on disk
    uploaded_ref = Column(String(255), nullable=False)
    cleaned_ref = Column(String(255), nullable=True)

    #Final output
    report = Column(Text, nullable=True)

    user = relationship("User", back_populates="runs")
    events = relationship("RunEvent", back_populates="run",order_by="RunEvent.id", cascade="all, delete-orphan")
    claims = relationship("Claim", back_populates="run", order_by="Claim.id", cascade="all, delete-orphan")
    cleaning_log = relationship("CleaningLogEntry", back_populates="run", order_by="CleaningLogEntry.id", cascade="all, delete-orphan")
    chat_session = relationship("ChatSession", back_populates="run", uselist=False, cascade="all, delete-orphan")
    charts = relationship("Chart", back_populates="run", cascade="all, delete-orphan")


class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False)
    text = Column(Text, nullable=False)
    metric = Column(String(255), nullable=True)
    value = Column(Float, nullable=True)
    source_query = Column(Text, nullable=True)
    source_columns = Column(JSON, nullable=True)
    verification_Status = Column(
         SAEnum("unverified", "confirmed", "contradicted", "unverifiable",
               name="verification_status"),
        default="unverified", nullable=False,
    )
    confidence = Column(Float, nullable=True)
    web_sources = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)

    run = relationship("Run", back_populates="claims")



class RunEvent(Base):
    __tablename__ = "run_events"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    run_id     = Column(String(36), ForeignKey("runs.id"), nullable=False, index=True)
    event_type = Column(String(64), nullable=False)
    data       = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)

    run = relationship("Run", back_populates="events")



class Chart(Base):
    __tablename__ = "charts"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    run_id     = Column(String(36), ForeignKey("runs.id"), nullable=False, index=True)
    ref        = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)

    run = relationship("Run", back_populates="charts")


#----Cleaning Log Entry

class CleaningLogEntry(Base):
    __tablename__ = "cleaning_log_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False)
    operation = Column(String(255), nullable=False)
    column = Column(String(255), nullable=False)
    rows_affected = Column(Integer, nullable=False)
    rationale = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)

    run = relationship("Run", back_populates="cleaning_log")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=True)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)

    user = relationship("User", back_populates="chat_sessions")
    run = relationship("Run", back_populates="chat_session")
    messages = relationship("ChatMessage", back_populates="session", order_by="ChatMessage.created_at", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(SAEnum("system", "user", "assistant", name="chat_role"), nullable=False)
    content = Column(Text, nullable=False)
    chart_ref = Column(JSON, nullable=True)
    citations = Column(JSON, nullable=True)
    follow_up_suggestions = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)

    session = relationship("ChatSession", back_populates="messages")