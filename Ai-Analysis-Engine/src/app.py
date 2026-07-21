"""
Verum FastAPI backend.

Endpoints:
  GET  /                      → SPA HTML dashboard
  POST /api/start             → save file, start run → returns {run_id}
  GET  /api/stream/{run_id}   → SSE stream of named pipeline events
  GET  /charts/{ref}          → serve saved chart JSON files
  GET  /health                → health check

Run with:
  uvicorn analysis_engine.app:app --reload --port 8000
"""
from sqlalchemy.orm import Session
from contextlib import contextmanager
from analysis_engine.db import SessionLocal
import json
import uuid
import asyncio
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from analysis_engine.chat_state import ChatSession, ChatMessage
from analysis_engine.agent.chat_agent import run_chat_turn

from analysis_engine.graph import build_graph
from analysis_engine.state import PipelineState, FileMeta
from analysis_engine.registry import RUN_CALLBACKS
from analysis_engine.db import init_db

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent          # project root
UPLOAD_DIR = Path(os.environ.get("VERUM_UPLOAD_DIR", str(BASE_DIR / "uploads")))
CHARTS_DIR = Path(os.environ.get("VERUM_CHARTS_DIR", str(BASE_DIR / "charts")))
TEMPLATE   = Path(__file__).parent / "templates" / "index.html"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# set env so analysis.py lazy-resolves to the same dir
os.environ["VERUM_CHARTS_DIR"] = str(CHARTS_DIR)
os.environ["VERUM_UPLOAD_DIR"] = str(UPLOAD_DIR)

# in-memory run registry: run_id → asyncio.Queue
# ponytail: In production, migrate to PostgreSQL to eliminate file-level locking
_RUN_QUEUES: dict[str, asyncio.Queue] = {}
_RUN_FILES:     dict[str, tuple[str, str | None]] = {}  # run_id → (file_path, question)
_COMPLETED_RUNS: dict[str, dict] = {}                   # run_id → final serialised PipelineState
_CHAT_SESSIONS:  dict[str, ChatSession] = {}

app = FastAPI(title="Verum Analysis Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    init_db()

# ── helpers ───────────────────────────────────────────────────────────────────

def _named_sse(event_type: str, data: dict) -> str:
    """
    Proper named SSE format so EventSource.addEventListener(name, ...) works.
      event: step_started
      data: {...}
      (blank line)
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ── root ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    if TEMPLATE.exists():
        return HTMLResponse(TEMPLATE.read_text())
    return HTMLResponse("<h2>frontend/index.html not found — check TEMPLATE path</h2>")


# ── charts ────────────────────────────────────────────────────────────────────
@app.get("/charts/{ref}")
async def get_chart(ref: str):
    # security: prevent path traversal
    safe_ref = Path(ref).name
    path = CHARTS_DIR / safe_ref
    if not path.exists():
        raise HTTPException(404, f"Chart {ref} not found")
    return JSONResponse(json.loads(path.read_text()))


# ── step 1: upload file, start background run, return run_id ─────────────────
@app.post("/api/start")
async def start_run(
    file: UploadFile = File(...),
    question: str = Form(default=""),
):
    run_id   = str(uuid.uuid4())
    csv_path = UPLOAD_DIR / f"{run_id}_{file.filename}"

    with csv_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _RUN_QUEUES[run_id] = queue
    _RUN_FILES[run_id]  = (str(csv_path), question or None)

    # ── Database Insert ── (use existing pattern from db/engine.py)
    from analysis_engine.db import SessionLocal
    from analysis_engine.db.models import Run, ChatSession
    try:
        db = SessionLocal()
        db_run = Run(
            id=run_id,
            user_id="default-user",
            filename=file.filename,
            question=question or None,
            status="running",
            uploaded_ref=str(csv_path),
        )
        db.add(db_run)
        db.commit()
        db.refresh(db_run)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    # Create run record in memory with timestamp
    _COMPLETED_RUNS[run_id] = {
        "run_id":       run_id,
        "filename":     file.filename,
        "question":     question or None,
        "status":       "running",
        "report":       None,
        "claims":       [],
        "chart_refs":   [],
        "cleaning_log": [],
        "cleaned_refs": {},
        "created_at":   datetime.utcnow().isoformat() + "Z",
        "started_at":   datetime.utcnow().isoformat() + "Z",
    }

    def on_event(event_type: str, data: dict):
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

        # Save event to database (use existing pattern)
        from analysis_engine.db import SessionLocal
        from analysis_engine.db.models import RunEvent
        try:
            db = SessionLocal()
            event = RunEvent(run_id=run_id, event_type=event_type, data=data)
            db.add(event)
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

    RUN_CALLBACKS[run_id] = on_event

    def run_graph():
        try:
            compiled = build_graph()
            initial_state = PipelineState(
                run_id=run_id,
                question=question or None,
                files=[FileMeta(file_id="input-file-1", ref=str(csv_path))],
            )
            result = compiled.invoke(initial_state)
            final  = PipelineState(**result)

            _COMPLETED_RUNS[run_id].update({
                "status":       final.status,
                "report":       final.report or "",
                "claims":       [c.model_dump() for c in final.claims],
                "chart_refs":   final.chart_refs,
                "cleaning_log": [e.model_dump() for e in final.cleaning_log],
                "cleaned_refs": final.cleaned_refs,
            })

            # Save final results to database (use existing pattern)
            from analysis_engine.db import SessionLocal
            from analysis_engine.db.models import Run, Claim, CleaningLogEntry, Chart
            try:
                db = SessionLocal()
                db_run = db.query(Run).filter(Run.id == run_id).first()
                if db_run:
                    db_run.status = final.status
                    db_run.report = final.report or ""
                    db_run.cleaned_ref = next(iter(final.cleaned_refs.values())) if final.cleaned_refs else None
                    db_run.completed_at = datetime.utcnow()

                    # Save claims
                    for c in final.claims:
                        db_claim = Claim(
                            run_id=run_id,
                            text=c.text,
                            metric=c.metric,
                            value=c.value,
                            source_query=c.source_query,
                            source_columns=c.source_columns,
                            verification_Status=c.verification_status,
                            confidence=c.confidence,
                            web_sources=c.web_sources,
                        )
                        db.add(db_claim)

                    # Save cleaning log entries
                    for e in final.cleaning_log:
                        db_entry = CleaningLogEntry(
                            run_id=run_id,
                            operation=e.operation,
                            column=e.column,
                            rows_affected=e.rows_affected,
                            rationale=e.rationale,
                        )
                        db.add(db_entry)

                    # Save charts
                    for ref in final.chart_refs:
                        db_chart = Chart(
                            run_id=run_id,
                            ref=ref,
                        )
                        db.add(db_chart)

                    db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

            loop.call_soon_threadsafe(queue.put_nowait, ("run_complete", {
                "run_id":       run_id,
                "status":       final.status,
                "report":       final.report or "",
                "claims":       [c.model_dump() for c in final.claims],
                "chart_refs":   final.chart_refs,
                "cleaning_log": [e.model_dump() for e in final.cleaning_log],
            }))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _COMPLETED_RUNS[run_id]["status"] = "failed"

            # Update database status to failed (use existing pattern)
            from analysis_engine.db import SessionLocal
            from analysis_engine.db.models import Run
            try:
                db = SessionLocal()
                db_run = db.query(Run).filter(Run.id == run_id).first()
                if db_run:
                    db_run.status = "failed"
                    db_run.completed_at = datetime.utcnow()
                    db.commit()
            except Exception:
                db.rollback()
            finally:
                db.close()

            loop.call_soon_threadsafe(queue.put_nowait, ("run_error", {"message": str(exc)}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("__done__", {}))
            
            # Delayed cleanup of registries to avoid disconnect errors if client refreshes
            def delayed_cleanup():
                import time
                time.sleep(10)
                RUN_CALLBACKS.pop(run_id, None)
                _RUN_QUEUES.pop(run_id, None)
                _RUN_FILES.pop(run_id, None)
            
            threading.Thread(target=delayed_cleanup, daemon=True).start()

    threading.Thread(target=run_graph, daemon=True).start()
    return {"run_id": run_id}


# ── step 2: SSE stream for this run ──────────────────────────────────────────
@app.get("/api/stream/{run_id}")
async def stream_run(run_id: str):
    queue = _RUN_QUEUES.get(run_id)
    if not queue:
        raise HTTPException(404, f"No run with id {run_id}")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                event_type, data = await queue.get()
                if event_type == "__done__":
                    break
                yield _named_sse(event_type, data)
        finally:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/runs/{run_id}/chat_history")
async def get_run_chat_history(run_id: str):
    """Get chat history for a run from the database."""
    from analysis_engine.db import SessionLocal
    from analysis_engine.db.models import ChatSession, ChatMessage
    
    db = SessionLocal()
    try:
        # Find the chat session for this run
        session = db.query(ChatSession).filter(ChatSession.run_id == run_id).order_by(ChatSession.created_at.desc()).first()
        
        if not session:
            return {"session_id": None, "messages": []}
        
        # Get all messages for this session
        messages = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at).all()
        
        # Sync to in-memory _CHAT_SESSIONS in case of server reload
        session_id_str = str(session.id)
        if session_id_str not in _CHAT_SESSIONS:
            from analysis_engine.db.models import Run
            run = db.query(Run).filter(Run.id == run_id).first()
            cleaned_ref = run.cleaned_ref if run else ""
            
            _CHAT_SESSIONS[session_id_str] = ChatSession(
                session_id = session_id_str,
                run_id = run_id,
                cleaned_ref = cleaned_ref,
                messages = [
                    ChatMessage(
                        role=m.role,
                        content=m.content,
                        chart_ref=m.chart_ref or [],
                        citations=m.citations or [],
                        follow_up_suggestions=m.follow_up_suggestions or [],
                    )
                    for m in messages
                ]
            )
        
        return {
            "session_id": str(session.id),  # Convert integer ID to string
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "chart_refs": m.chart_ref or [],  # map singular in DB to plural for frontend
                    "citations": m.citations or [],
                    "follow_up_suggestions": m.follow_up_suggestions or [],
                }
                for m in messages
            ],
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get chat history: {str(e)}")
    finally:
        db.close()


@app.get("/api/runs")
async def get_runs():
    """Get list of runs with their status and metadata."""
    from analysis_engine.db import SessionLocal
    from analysis_engine.db.models import Run
    
    db = SessionLocal()
    try:
        db_runs = db.query(Run).order_by(Run.created_at.desc()).all()
        runs = []
        for r in db_runs:
            runs.append({
                "run_id": r.id,
                "filename": r.filename,
                "file": r.filename,
                "status": r.status,
                "created_at": r.created_at.isoformat() + "Z" if r.created_at else None,
                "started_at": r.created_at.isoformat() + "Z" if r.created_at else None,
            })
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(500, f"Failed to get runs: {str(e)}")
    finally:
        db.close()


@app.get("/api/runs/{run_id}/events")
async def get_run_events(run_id: str):
    """Get all events for a specific run."""
    from analysis_engine.db import SessionLocal
    from analysis_engine.db.models import RunEvent
    
    db = SessionLocal()
    try:
        events = db.query(RunEvent).filter(RunEvent.run_id == run_id).order_by(RunEvent.created_at).all()
        return [
            {
                "event_type": ev.event_type,
                "data": ev.data,
                "created_at": ev.created_at.isoformat() if ev.created_at else None,
            }
            for ev in events
        ]
    except Exception as e:
        raise HTTPException(500, f"Failed to get events: {str(e)}")
    finally:
        db.close()


@app.get("/api/runs/{run_id}/report")
async def get_run_report(run_id: str):
    """Get report data (claims, cleaning log, filename) for a run."""
    from analysis_engine.db import SessionLocal
    from analysis_engine.db.models import Run, Claim, CleaningLogEntry, Chart
    
    db = SessionLocal()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            raise HTTPException(404, f"Run {run_id} not found")
        
        claims = db.query(Claim).filter(Claim.run_id == run_id).all()
        cleaning_log = db.query(CleaningLogEntry).filter(CleaningLogEntry.run_id == run_id).all()
        charts = db.query(Chart).filter(Chart.run_id == run_id).all()
        
        return {
            "status": run.status,
            "filename": run.filename,
            "file": run.filename,
            "report": run.report or "",
            "claims": [
                {
                    "text": c.text,
                    "metric": c.metric,
                    "value": c.value,
                    "source_query": c.source_query,
                    "source_columns": c.source_columns,
                    "verification_status": c.verification_Status,
                    "confidence": c.confidence,
                    "web_sources": c.web_sources,
                }
                for c in claims
            ],
            "cleaning_log": [
                {
                    "operation": e.operation,
                    "column": e.column,
                    "rows_affected": e.rows_affected,
                    "rationale": e.rationale,
                }
                for e in cleaning_log
            ],
            "chart_refs": [ch.ref for ch in charts],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to get report: {str(e)}")
    finally:
        db.close()


@app.post("/api/chat/start")
async def start_chat_session(body: dict):
    run_id = body.get("run_id")
    if not run_id:
        raise HTTPException(400, "run_id is required")

    from analysis_engine.db import SessionLocal
    from analysis_engine.db.models import Run, ChatSession as DBChatSession
    
    db = SessionLocal()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            raise HTTPException(404, f"Run {run_id} not found.")
        
        if not run.cleaned_ref:
            raise HTTPException(400, f"Run {run_id} has not completed its cleaning stage yet. Please wait for completion before starting a chat.")
        cleaned_ref = run.cleaned_ref

        # Create ChatSession in database
        db_session = DBChatSession(
            user_id="default-user",
            run_id=run_id,
            title=f"Chat on {run.filename or 'Run'}",
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        session_id = str(db_session.id)
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Failed to start chat session: {e}")
    finally:
        db.close()

    _CHAT_SESSIONS[session_id] = ChatSession(
        session_id = session_id,
        run_id = run_id,
        cleaned_ref = cleaned_ref,
    )
    return {"session_id": session_id, "run_id": run_id}


@app.post("/api/chat/message")
async def send_chat_message(body: dict):
    session_id = body.get("session_id")
    user_message = body.get("message","").strip()
    if not session_id:
        raise HTTPException(400, "session_id is required")
    if not user_message:
        raise HTTPException(400, "message is required")

    session = _CHAT_SESSIONS.get(session_id)
    # If not in cache, load from DB
    if not session:
        from analysis_engine.db import SessionLocal
        from analysis_engine.db.models import ChatSession as DBChatSession, ChatMessage as DBChatMessage, Run
        db = SessionLocal()
        try:
            db_session = db.query(DBChatSession).filter(DBChatSession.id == int(session_id)).first()
            if not db_session:
                raise HTTPException(404, f"Chat session {session_id} not found")
            
            run = db.query(Run).filter(Run.id == db_session.run_id).first()
            cleaned_ref = run.cleaned_ref if run else ""
            
            db_messages = db.query(DBChatMessage).filter(DBChatMessage.session_id == db_session.id).order_by(DBChatMessage.created_at).all()
            
            session = ChatSession(
                session_id = session_id,
                run_id = db_session.run_id,
                cleaned_ref = cleaned_ref,
                messages = [
                    ChatMessage(
                        role=m.role,
                        content=m.content,
                        chart_ref=m.chart_ref or [],
                        citations=m.citations or [],
                        follow_up_suggestions=m.follow_up_suggestions or [],
                    )
                    for m in db_messages
                ]
            )
            _CHAT_SESSIONS[session_id] = session
        except Exception as e:
            raise HTTPException(500, f"Failed to restore chat session: {e}")
        finally:
            db.close()

    # Reconstruct final_state from database
    from analysis_engine.db import SessionLocal
    from analysis_engine.db.models import Run, Claim
    db = SessionLocal()
    try:
        db_run = db.query(Run).filter(Run.id == session.run_id).first()
        if not db_run:
            raise HTTPException(404, f"Run {session.run_id} not found")
        
        # If cleaned_ref wasn't set in memory cache yet, update it from DB
        if not session.cleaned_ref and db_run.cleaned_ref:
            session.cleaned_ref = db_run.cleaned_ref
            
        # Verify the pipeline has at least cleaned the dataset before allowing messages
        if not session.cleaned_ref:
            raise HTTPException(400, "The data analysis pipeline is still in progress. Please wait for the cleaning step to complete.")

        claims = db.query(Claim).filter(Claim.run_id == session.run_id).all()
        run_data = {
            "report": db_run.report or "",
            "claims": [
                {
                    "text": c.text,
                    "metric": c.metric,
                    "value": c.value,
                    "source_query": c.source_query,
                    "source_columns": c.source_columns,
                    "verification_status": c.verification_Status,
                    "confidence": c.confidence,
                    "web_sources": c.web_sources,
                }
                for c in claims
            ],
            "cleaned_refs": {"cleaned-file-1": db_run.cleaned_ref} if db_run.cleaned_ref else {},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to load run details for chat context: {e}")
    finally:
        db.close()

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(event_type: str, data: dict):
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

    # Append user message in memory
    session.messages.append(ChatMessage(role="user", content=user_message))

    # Save user message to database
    try:
        from analysis_engine.db import SessionLocal
        from analysis_engine.db.models import ChatMessage as DBChatMessage
        db = SessionLocal()
        db_msg = DBChatMessage(
            session_id=int(session_id),
            role="user",
            content=user_message,
        )
        db.add(db_msg)
        db.commit()
    except Exception as e:
        print(f"[db error] Failed to save user chat message: {e}")
    finally:
        db.close()

    def run_chat():
        try:
            assistant_msg = run_chat_turn(
                session = session,
                user_message = user_message,
                final_state = run_data,
                event_callback = on_event,
            )
            session.messages.append(assistant_msg)
            
            # Save assistant message to database
            try:
                from analysis_engine.db import SessionLocal
                from analysis_engine.db.models import ChatMessage as DBChatMessage
                db = SessionLocal()
                db_msg = DBChatMessage(
                    session_id=int(session_id),
                    role="assistant",
                    content=assistant_msg.content,
                    chart_ref=assistant_msg.chart_ref,
                    citations=assistant_msg.citations,
                    follow_up_suggestions=assistant_msg.follow_up_suggestions,
                )
                db.add(db_msg)
                db.commit()
            except Exception as e:
                print(f"[db error] Failed to save assistant chat message: {e}")
            finally:
                db.close()
        except Exception as e:
            import traceback
            traceback.print_exc()
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("chat_error", {"message": str(e)})
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("__done__", {}))

    threading.Thread(target=run_chat, daemon=True).start()

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            event_type, data = await queue.get()
            if event_type == "__done__":
                break
            yield _named_sse(event_type, data)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """Full conversation history for a session."""
    session = _CHAT_SESSIONS.get(session_id)
    # If not in cache, load from DB
    if not session:
        from analysis_engine.db import SessionLocal
        from analysis_engine.db.models import ChatSession as DBChatSession, ChatMessage as DBChatMessage
        db = SessionLocal()
        try:
            db_session = db.query(DBChatSession).filter(DBChatSession.id == int(session_id)).first()
            if not db_session:
                raise HTTPException(404, f"Session {session_id} not found")
            
            db_messages = db.query(DBChatMessage).filter(DBChatMessage.session_id == db_session.id).order_by(DBChatMessage.created_at).all()
            
            # Reconstruct to cache
            from analysis_engine.db.models import Run
            run = db.query(Run).filter(Run.id == db_session.run_id).first()
            cleaned_ref = run.cleaned_ref if run else ""
            
            session = ChatSession(
                session_id = session_id,
                run_id = db_session.run_id,
                cleaned_ref = cleaned_ref,
                messages = [
                    ChatMessage(
                        role=m.role,
                        content=m.content,
                        chart_ref=m.chart_ref or [],
                        citations=m.citations or [],
                        follow_up_suggestions=m.follow_up_suggestions or [],
                    )
                    for m in db_messages
                ]
            )
            _CHAT_SESSIONS[session_id] = session
        except Exception as e:
            raise HTTPException(500, f"Failed to get chat session history: {e}")
        finally:
            db.close()
            
    return {
        "session_id": session_id,
        "run_id": session.run_id,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "chart_refs": m.chart_ref or [],
                "citations": m.citations or [],
                "follow_up_suggestions": m.follow_up_suggestions or [],
            }
            for m in session.messages
        ],
    }


@app.get("/api/chat/sessions")
async def get_all_chat_sessions():
    """Get list of all chat sessions."""
    from analysis_engine.db import SessionLocal
    from analysis_engine.db.models import ChatSession, ChatMessage
    
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
        
        result = []
        for session in sessions:
            # Get message count
            msg_count = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).count()
            
            # Get last message if exists
            last_msg = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at.desc()).first()
            
            result.append({
                "session_id": session.id,
                "run_id": session.run_id,
                "title": session.title or f"Chat session #{session.id}",
                "created_at": session.created_at.isoformat() + "Z" if session.created_at else None,
                "message_count": msg_count,
                "last_message": last_msg.content[:100] + "..." if last_msg and len(last_msg.content) > 100 else (last_msg.content if last_msg else None),
                "last_message_at": last_msg.created_at.isoformat() + "Z" if last_msg and last_msg.created_at else None,
            })
        
        return result
    except Exception as e:
        raise HTTPException(500, f"Failed to get sessions: {str(e)}")
    finally:
        db.close()


@app.get("/api/chat/sessions/{run_id}")
async def get_chat_sessions(run_id: str):
    """Get all chat sessions for a run."""
    from analysis_engine.db import SessionLocal
    from analysis_engine.db.models import ChatSession, ChatMessage
    
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).filter(ChatSession.run_id == run_id).order_by(ChatSession.created_at.desc()).all()
        
        result = []
        for session in sessions:
            msg_count = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).count()
            last_msg = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at.desc()).first()
            
            result.append({
                "session_id": session.id,
                "run_id": run_id,
                "title": session.title or f"Chat session #{session.id}",
                "created_at": session.created_at.isoformat() + "Z" if session.created_at else None,
                "message_count": msg_count,
                "last_message": last_msg.content[:100] + "..." if last_msg and len(last_msg.content) > 100 else (last_msg.content if last_msg else None),
                "last_message_at": last_msg.created_at.isoformat() + "Z" if last_msg and last_msg.created_at else None,
            })
        
        return {"sessions": result}
    except Exception as e:
        raise HTTPException(500, f"Failed to get sessions: {str(e)}")
    finally:
        db.close()


# ── Database Session Helper ──
# All DB operations in this file should use get_session() instead of SessionLocal() directly
@contextmanager
def get_db_session() -> Session:
    """use this instead of SessionLocal() directly for proper transaction handling."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

