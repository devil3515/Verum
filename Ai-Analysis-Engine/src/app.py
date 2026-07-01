"""
FastAPI web server for the Verum AI Analysis Engine.

Exposes:
  GET  /            → SPA HTML dashboard
  POST /api/run     → SSE stream of pipeline events (upload CSV, stream events)
  GET  /charts/{ref} → Serve saved chart JSON files

Run with:
  uvicorn analysis_engine.web:app --reload --port 8000
"""
import json
import uuid
import asyncio
import os
import shutil
import threading
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from analysis_engine.graph import build_graph
from analysis_engine.state import PipelineState, FileMeta

# ─── config ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent           # src/
UPLOAD_DIR = Path(os.environ.get("VERUM_UPLOAD_DIR",  str(BASE_DIR / "uploads")))
CHARTS_DIR = Path(os.environ.get("VERUM_CHARTS_DIR",  str(BASE_DIR / "charts")))
TEMPLATE   = BASE_DIR / "templates" / "index.html"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Verum Analysis Engine")

# ─── chart static files ───────────────────────────────────────────────────────
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")


# ─── root: serve SPA ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(TEMPLATE.read_text())


# ─── SSE event helper ─────────────────────────────────────────────────────────
def _sse(event_type: str, data: dict) -> str:
    payload = json.dumps({"type": event_type, **data})
    return f"data: {payload}\n\n"


# ─── run endpoint ─────────────────────────────────────────────────────────────
@app.post("/api/run")
async def run_analysis(
    file: UploadFile = File(...),
    question: str = Form(default=""),
):
    """
    Accept a CSV upload; stream back Server-Sent Events as the pipeline runs.
    """
    run_id  = str(uuid.uuid4())
    csv_path = UPLOAD_DIR / f"{run_id}_{file.filename}"

    # save uploaded file
    with csv_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        # callback fired from the analysis node (different thread)
        def on_event(event_type: str, data: dict):
            loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

        yield _sse("run_started", {"run_id": run_id, "file": file.filename})

        # register callback so analysis_node can push events
        from analysis_engine.nodes.analysis import RUN_CALLBACKS
        RUN_CALLBACKS[run_id] = on_event

        # build graph initial state
        initial_state = PipelineState(
            run_id=run_id,
            question=question or None,
            files=[FileMeta(file_id="input-file-1", ref=str(csv_path))],
        )
        compiled = build_graph()

        # run graph in a background thread so we don't block the event loop
        final_state_container = {}
        error_container = {}

        def run_graph():
            try:
                result = compiled.invoke(initial_state)
                final_state_container["state"] = PipelineState(**result)
            except Exception as exc:
                error_container["error"] = str(exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("__done__", {}))

        thread = threading.Thread(target=run_graph, daemon=True)
        thread.start()

        # drain the event queue until done
        while True:
            event_type, data = await queue.get()
            if event_type == "__done__":
                break
            yield _sse(event_type, data)

        # cleanup callback
        RUN_CALLBACKS.pop(run_id, None)

        if "error" in error_container:
            yield _sse("run_error", {"message": error_container["error"]})
            return

        final: PipelineState = final_state_container["state"]

        # emit completed state
        yield _sse("run_complete", {
            "run_id":   run_id,
            "status":   final.status,
            "report":   final.report or "",
            "claims":   [c.model_dump() for c in final.claims],
            "chart_refs": final.chart_refs,
            "cleaning_log": [e.model_dump() for e in final.cleaning_log],
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
