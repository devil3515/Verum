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

    def on_event(event_type: str, data: dict):
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

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

            _COMPLETED_RUNS[run_id] = {
                "run_id":       run_id,
                "filename":     file.filename,
                "question":     question or None,
                "status":       final.status,
                "report":       final.report or "",
                "claims":       [c.model_dump() for c in final.claims],
                "chart_refs":   final.chart_refs,
                "cleaning_log": [e.model_dump() for e in final.cleaning_log],
                "cleaned_refs": final.cleaned_refs,
            }

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
            loop.call_soon_threadsafe(queue.put_nowait, ("run_error", {"message": str(exc)}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("__done__", {}))

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
            RUN_CALLBACKS.pop(run_id, None)
            _RUN_QUEUES.pop(run_id, None)
            _RUN_FILES.pop(run_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}



@app.post("/api/chat/start")
async def start_chat_session(body: dict):
    run_id = body.get("run_id")
    if not run_id:
        raise HTTPException(400, "run_id is required")

    run = _COMPLETED_RUNS.get(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found or not yet complete. "
                                 "Wait for run_complete event before starting a chat.")

    cleaned_refs = run.get("cleaned_refs", {})
    if not cleaned_refs:
        raise HTTPException(400, f"Run {run_id} has no cleaned data to chat against.")
    cleaned_ref = next(iter(cleaned_refs.values()))

    session_id = str(uuid.uuid4())
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
    if not session:
        raise HTTPException(404, f"Chat session {session_id} not found")

    run = _COMPLETED_RUNS.get(session.run_id)
    if not run:
        raise HTTPException(404, f"Run {session.run_id} not found or not yet complete. "
                                 "Wait for run_complete event before sending chat messages.")

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(event_type: str, data: dict):
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

    session.messages.append(ChatMessage(role="user", content=user_message))

    def run_chat():
        try:
            assistant_msg = run_chat_turn(
                session = session,
                user_message = user_message,
                final_state = run,
                event_callback = on_event,
            )
            session.messages.append(assistant_msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("chat_error", {"message": str(exc)})
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
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    return {
        "session_id": session_id,
        "run_id": session.run_id,
        "messages": [m.model_dump() for m in session.messages],
    }

