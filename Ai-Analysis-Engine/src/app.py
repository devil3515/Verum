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

from analysis_engine.graph import build_graph
from analysis_engine.state import PipelineState, FileMeta
from analysis_engine.registry import RUN_CALLBACKS

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
_RUN_FILES:  dict[str, tuple[str, str | None]] = {}  # run_id → (file_path, question)

app = FastAPI(title="Verum Analysis Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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