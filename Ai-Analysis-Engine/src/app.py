"""
Verum FastAPI backend — Phase DB-2.

Every SSE event is now persisted to run_events table immediately.
Every completed run is persisted to runs/claims/charts/cleaning_log.
In-memory dicts remain as a fast cache on top of the DB.

New endpoints added in this phase:
  GET /api/runs/{run_id}/events  → full event log for replay on reconnect
"""
import json
import uuid
import asyncio
import os
import shutil
import threading
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from analysis_engine.graph import build_graph
from analysis_engine.state import PipelineState, FileMeta
from analysis_engine.registry import RUN_CALLBACKS
from analysis_engine.chat_state import ChatSession, ChatMessage
from analysis_engine.agent.chat_agent import run_chat_turn
from analysis_engine.db import init_db
from analysis_engine.db.repository import (
    persist_run_started,
    persist_run_completed,
    persist_run_failed,
    persist_event,
    get_events_for_run,
    get_run,
    list_runs as db_list_runs,
    persist_chat_session,
    persist_chat_message,
    get_chat_history as db_get_chat_history,
)

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
UPLOAD_DIR = Path(os.environ.get("VERUM_UPLOAD_DIR", str(BASE_DIR / "uploads")))
CHARTS_DIR = Path(os.environ.get("VERUM_CHARTS_DIR", str(BASE_DIR / "charts")))
TEMPLATE   = BASE_DIR / "frontend" / "index.html"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["VERUM_CHARTS_DIR"] = str(CHARTS_DIR)
os.environ["VERUM_UPLOAD_DIR"] = str(UPLOAD_DIR)

# in-memory cache (fast path) — DB is source of truth
_RUN_QUEUES:     dict[str, asyncio.Queue] = {}
_COMPLETED_RUNS: dict[str, dict]          = {}
_CHAT_SESSIONS:  dict[str, ChatSession]   = {}

app = FastAPI(title="Verum Analysis Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    init_db()


# ── helpers ───────────────────────────────────────────────────────────────────

def _named_sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _emit(run_id: str, loop, queue: asyncio.Queue, event_type: str, data: dict):
    """
    Emit one SSE event:
      1. persist to DB immediately (replay buffer)
      2. push to in-memory queue (live stream)
    """
    persist_event(run_id, event_type, data)
    loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))


# ── root ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    if TEMPLATE.exists():
        return HTMLResponse(TEMPLATE.read_text())
    return HTMLResponse("<h2>frontend/index.html not found</h2>")


# ── charts ────────────────────────────────────────────────────────────────────
@app.get("/charts/{ref}")
async def get_chart(ref: str):
    safe_ref = Path(ref).name
    path = CHARTS_DIR / safe_ref
    if not path.exists():
        raise HTTPException(404, f"Chart {ref} not found")
    return JSONResponse(json.loads(path.read_text()))


# ── start run ─────────────────────────────────────────────────────────────────
@app.post("/api/start")
async def start_run(
    file: UploadFile = File(...),
    question: str = Form(default=""),
):
    run_id   = str(uuid.uuid4())
    csv_path = UPLOAD_DIR / f"{run_id}_{file.filename}"

    with csv_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    # persist run row immediately so events can reference it
    persist_run_started(
        run_id=run_id,
        filename=file.filename,
        question=question or None,
        upload_ref=str(csv_path),
    )

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _RUN_QUEUES[run_id] = queue

    def on_event(event_type: str, data: dict):
        _emit(run_id, loop, queue, event_type, data)

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

            claims_data       = [c.model_dump() for c in final.claims]
            cleaning_log_data = [e.model_dump() for e in final.cleaning_log]
            cleaned_ref       = next(iter(final.cleaned_refs.values()), None) if final.cleaned_refs else None

            # persist everything to DB
            persist_run_completed(
                run_id=run_id,
                cleaned_ref=cleaned_ref or "",
                report=final.report or "",
                claims=claims_data,
                chart_refs=final.chart_refs,
                cleaning_logs=cleaning_log_data,
            )

            # update in-memory cache
            completed = {
                "run_id":       run_id,
                "filename":     file.filename,
                "question":     question or None,
                "status":       final.status,
                "report":       final.report or "",
                "claims":       claims_data,
                "chart_refs":   final.chart_refs,
                "cleaning_log": cleaning_log_data,
                "cleaned_refs": final.cleaned_refs,
            }
            _COMPLETED_RUNS[run_id] = completed

            run_complete_data = {
                "run_id":       run_id,
                "status":       final.status,
                "report":       final.report or "",
                "claims":       claims_data,
                "chart_refs":   final.chart_refs,
                "cleaning_log": cleaning_log_data,
            }
            _emit(run_id, loop, queue, "run_complete", run_complete_data)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            error_msg = str(exc)
            persist_run_failed(run_id, error_msg)
            _emit(run_id, loop, queue, "run_error", {"message": error_msg})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("__done__", {}))

    threading.Thread(target=run_graph, daemon=True).start()
    return {"run_id": run_id}


# ── live SSE stream ───────────────────────────────────────────────────────────
@app.get("/api/stream/{run_id}")
async def stream_run(run_id: str):
    queue = _RUN_QUEUES.get(run_id)
    if not queue:
        raise HTTPException(404, f"No active stream for run {run_id}. "
                                "Use /api/runs/{run_id}/events to replay a completed run.")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                event_type, data = await queue.get()
                if event_type == "__done__":
                    break
                yield _named_sse(event_type, data)
        finally:
            RUN_CALLBACKS.pop(run_id, None)
            _RUN_QUEUES.pop(run_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── event replay (Phase DB-3 endpoint, available from DB-2 onwards) ───────────
@app.get("/api/runs/{run_id}/events")
async def get_run_events(run_id: str):
    """
    Return all events for a run in chronological order.
    Frontend uses this on reconnect/page-load to replay pipeline state.
    """
    events = get_events_for_run(run_id)
    if not events:
        # check if run exists at all
        run = _COMPLETED_RUNS.get(run_id) or get_run(run_id)
        if not run:
            raise HTTPException(404, f"Run {run_id} not found")
    return events


# ── run history ───────────────────────────────────────────────────────────────
@app.get("/api/runs")
async def list_runs():
    """All runs — reads from DB so it survives restarts."""
    return db_list_runs()


@app.get("/api/runs/{run_id}/report")
async def get_run_report(run_id: str):
    """Full result for one run — checks cache first, falls back to DB."""
    # fast path: in-memory cache
    run = _COMPLETED_RUNS.get(run_id)
    if run:
        return run
    # slow path: DB (e.g. after server restart)
    run = get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    # warm the cache
    _COMPLETED_RUNS[run_id] = run
    return run


# ── chat ──────────────────────────────────────────────────────────────────────
@app.post("/api/chat/start")
async def start_chat_session(body: dict):
    """
    Create a chat session bound to a completed run.
    Body: { "run_id": "..." }
    Returns: { "session_id": "..." }
    """
    run_id = body.get("run_id")
    if not run_id:
        raise HTTPException(400, "run_id is required")

    # check cache first, then DB
    run = _COMPLETED_RUNS.get(run_id) or get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found or not yet complete.")

    cleaned_refs = run.get("cleaned_refs", {})
    if not cleaned_refs:
        raise HTTPException(400, f"Run {run_id} has no cleaned data.")
    cleaned_ref = next(iter(cleaned_refs.values()))

    session_id = str(uuid.uuid4())

    # persist to DB
    persist_chat_session(session_id=session_id, run_id=run_id)

    # keep in memory for fast access
    _CHAT_SESSIONS[session_id] = ChatSession(
        session_id=session_id,
        run_id=run_id,
        cleaned_ref=cleaned_ref,
    )
    return {"session_id": session_id, "run_id": run_id}


@app.post("/api/chat/message")
async def send_chat_message(body: dict):
    """
    Body: { "session_id": "...", "message": "..." }
    Returns SSE stream ending with chat_answer.
    """
    session_id   = body.get("session_id")
    user_message = body.get("message", "").strip()

    if not session_id:
        raise HTTPException(400, "session_id is required")
    if not user_message:
        raise HTTPException(400, "message cannot be empty")

    session = _CHAT_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    run = _COMPLETED_RUNS.get(session.run_id) or get_run(session.run_id)
    if not run:
        raise HTTPException(404, f"Pipeline run {session.run_id} not found")

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(event_type: str, data: dict):
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

    # persist user message
    persist_chat_message(
        session_id=session_id,
        role="user",
        content=user_message,
        chart_refs=[],
        citations=[],
    )
    session.messages.append(ChatMessage(role="user", content=user_message))

    def run_chat():
        try:
            assistant_msg = run_chat_turn(
                session=session,
                user_message=user_message,
                final_state=run,
                event_callback=on_event,
            )
            # persist assistant message
            persist_chat_message(
                session_id=session_id,
                role="assistant",
                content=assistant_msg.content,
                chart_refs=assistant_msg.chart_ref,
                citations=assistant_msg.citations,
            )
            session.messages.append(assistant_msg)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            loop.call_soon_threadsafe(
                queue.put_nowait, ("chat_error", {"message": str(exc)})
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
    """Full conversation history — reads from DB."""
    session = _CHAT_SESSIONS.get(session_id)
    run_id  = session.run_id if session else None
    msgs    = db_get_chat_history(session_id)
    return {
        "session_id": session_id,
        "run_id":     run_id,
        "messages":   msgs,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}