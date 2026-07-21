# Verum Agent Swarm - Complete Implementation Plan

## Overview

Build a **7-agent parallel swarm research engine** on top of existing analysis engine with:

- **Parallel fan-out** using LangGraph Send API
- **Self-correction loops** via reflector
- **Source tracking** via dedicated ResearchSource table
- **Recursive depth limits** (max 50)
- **7 distinct agents** (orchestrator, web_scourer, deep_reader, specialist, reflector, fact_checker, synthesizer)

**Architecture**: Orchestrator → Publish sub-tasks to ALL agents → Parallel execution → Reflector → [gaps?]→ Orchestrator again → Synthesizer → Fact Checker → END

---

## Phase 1: Database Schema (2 hours)

### 1.1 Create Migration Script

**File**: `Ai-Analysis-Engine/migrations/add_research_tables.sql`

```sql
-- Table 1: Research Tasks
CREATE TABLE research_tasks (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    run_id VARCHAR(36),
    objective TEXT NOT NULL,
    task_type VARCHAR(50) NOT NULL,  -- general, company, academic, medical
    depth INTEGER DEFAULT 3,          -- 1=quick, 3=standard, 5=deep
    status VARCHAR(50) DEFAULT 'pending',
    sub_tasks JSON DEFAULT '[]',
    findings JSON DEFAULT '[]',
    synthesis TEXT,
    error TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Table 2: Research Sources (deduplicated URLs for citations)
CREATE TABLE research_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id VARCHAR(36) NOT NULL,
    url TEXT NOT NULL,
    url_hash VARCHAR(64) NOT NULL,   -- SHA256 for O(1) dedup lookups
    title TEXT,
    source_type VARCHAR(50),         -- web, news, paper, document
    content_snippet TEXT,            -- first 500 chars
    credibility_score REAL,          -- 0.0-1.0
    retrieved_by VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES research_tasks(id)
);

-- Table 3: Task Events (granular event log)
CREATE TABLE task_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id VARCHAR(36) NOT NULL,
    event_type VARCHAR(100) NOT NULL,  -- plan, search, scrape, analyze, reflect, synthesize, error
    event_level VARCHAR(20) NOT NULL,  -- info, warning, error, debug
    agent_role VARCHAR(50),
    agent_id VARCHAR(50),
    parent_event_id INTEGER,
    data JSON NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES research_tasks(id)
);

-- PERFORMANCE INDEXES
CREATE INDEX idx_task_id ON research_tasks(id);
CREATE INDEX idx_task_status ON research_tasks(status);
CREATE INDEX idx_sources_url_hash ON research_sources(url_hash);
CREATE INDEX idx_sources_task_id ON research_sources(task_id);
CREATE INDEX idx_events_task_id ON task_events(task_id);
CREATE INDEX idx_events_created_at ON task_events(created_at);
```

**Test**:
```bash
cd Ai-Analysis-Engine
sqlite3 verum.db < migrations/add_research_tables.sql

# Verify
sqlite3 verum.db ".tables"              # Should show: research_tasks, research_sources, task_events
sqlite3 verum.db "SELECT COUNT(*) FROM research_sources;"  # Should show 0 rows
```

---

### 1.2 Add Models

**File**: `Ai-Analysis-Engine/src/analysis_engine/db/models.py`

**Action**: Add these models AFTER existing models (after Chart class at line ~110).

```python
class ResearchTask(Base):
    """Stored research objectives and state tracking."""
    __tablename__ = "research_tasks"

    id = Column(String(36), primary_key=True, default=str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    run_id = Column(String(36), ForeignKey("runs.id"), default=None)
    objective = Column(Text, nullable=False)
    task_type = Column(String(50), nullable=False)
    depth = Column(Integer, default=3)
    status = Column(String(50), default="pending")
    sub_tasks = Column(JSON, default=list)
    findings = Column(JSON, default=list)
    synthesis = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class ResearchSource(Base):
    """Deduplicated source tracking for citations."""
    __tablename__ = "research_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), ForeignKey("research_tasks.id"), nullable=False)
    url = Column(Text, nullable=False)
    url_hash = Column(String(64), nullable=False, index=True)  # SHA256 hash
    title = Column(Text, nullable=True)
    source_type = Column(String(50), nullable=True)
    content_snippet = Column(Text, nullable=True)
    credibility_score = Column(Float, default=0.5)
    retrieved_by = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class TaskEvent(Base):
    """Granular event log for each agent operation."""
    __tablename__ = "task_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), ForeignKey("research_tasks.id"), nullable=False, index=True)
    event_type = Column(String(100), nullable=False)
    event_level = Column(String(20), nullable=False)
    agent_role = Column(String(50), nullable=True)
    agent_id = Column(String(50), nullable=True)
    parent_event_id = Column(Integer, nullable=True)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
```

**Test**:
```python
# In Python shell
from database import SessionLocal
from models import ResearchTask

db = SessionLocal()

# Insert test task
task = ResearchTask(
    objective="Test objective",
    task_type="general",
    depth=3
)
db.add(task)
db.commit()
print("ResearchTask model working")
db.close()
```

---

## Phase 2: Backend Core (8 hours)

### 2.1 Create Research State (TypedDict)

**File**: `Ai-Analysis-Engine/src/research/state.py`

```python
from typing import TypedDict, Literal, Annotated
from operator import add


class SubTask(TypedDict):
    """Individual distributed work item."""
    id: str
    question: str
    assigned_agent: str  # Which agent handles this
    status: str  # pending, in_progress, complete, failed


class Finding(TypedDict):
    """Evidence surface gathered by an agent."""
    agent: str
    sub_task_id: str
    content: str
    sources: list[str]  # URLs
    confidence: float
    metadata: dict[str, str]


class ResearchState(TypedDict):
    """Core orchestration state for parallel agent swarm."""
    # Identity
    task_id: str
    user_id: str
    run_id: str | None

    # Objective
    objective: str
    task_type: str
    depth: int

    # Plan
    sub_tasks: list[SubTask]

    # Accumulated results (parallel merge with Annotated[add])
    findings: Annotated[list[Finding], add]
    sources: Annotated[list[str], add]  # URLs collected
    searched_urls: Annotated[list[str], add]  # Dedup tracking

    # Control flow
    iteration: int
    max_iterations: int
    gaps: list[str]  # Identified by reflector
    status: str

    # Output
    synthesis: str | None
    error: str | None
```

---

### 2.2 Create Agent Taxonomy

**File**: `Ai-Analysis-Engine/src/research/agents.py`

```python
"""Agent configuration registry for the 7-agent swarm."""

AGENT_CONFIG = {
    "orchestrator": {
        "depth": 1,
        "role": "Manager",
        "model": "claude-sonnet",
        "capabilities": ["decompose_task", "assign_team", "evaluate_completeness"],
        max_iterations: int = 5,
    },
    "web_searcher": {
        "depth": 2,
        "role": "Web Researcher",
        "model": "gpt-4o-mini",
        "capabilities": ["search_web", "search_news"],
        max_iterations: int = 10,
        token_budget: int = 8000,
    },
    "deep_reader": {
        "depth": 2,
        "role": "Content Reader",
        "model": "gpt-4o-mini",
        "capabilities": ["scrape_url", "read_pdf", "extract_key_points"],
        max_iterations: int = 8,
        token_budget: int = 15000,
    },
    "topic_specialist": {
        "depth": 3,
        "role": "Domain Analyst",
        "model": "claude-sonnet",
        "capabilities": ["analyze_findings", "identify_gaps", "assess_credibility"],
        max_iterations: int = 5,
        token_budget: int = 12000,
    },
    "reflector": {
        "depth": 1,
        "role": "Critic",
        "model": "claude-sonnet",
        "capabilities": ["evaluate_completeness", "identify_gaps", "suggest_next_steps"],
        max_iterations: int = 3,
    },
    "fact_checker": {
        "depth": 3,
        "role": "Verifier",
        "model": "gpt-4o-mini",
        "capabilities": ["verify_claims", "cross_reference", "flag_contradictions"],
        max_iterations: int = 5,
        token_budget: int = 8000,
    },
    "synthesizer": {
        "depth": 1,
        "role": "Writer",
        "model": "claude-sonnet",
        "capabilities": ["combine_findings", "format_report", "generate_citations"],
        max_iterations: int = 2,
    },
}
```

---

### 2.3 Create Research Registry (SEPARATE)

**File**: `Ai-Analysis-Engine/src/research/registry.py`

```python
"""Separate registry for research tasks. Don't modify existing RUN_CALLBACKS."""
import asyncio
from collections import defaultdict
from typing import Callable

# Research task callbacks
RESEARCH_CALLBACKS: dict[str, Callable] = {}
# Task ID → SSE queue
RESEARCH_QUEUES: dict[str, asyncio.Queue] = {}


def register_research_callback(task_id: str, callback: Callable):
    """Register event callback for research task."""
    RESEARCH_CALLBACKS[task_id] = callback


def get_research_queue(task_id: str) -> asyncio.Queue:
    """Get or create SSE queue for task."""
    if task_id not in RESEARCH_QUEUES:
        RESEARCH_QUEUES[task_id] = asyncio.Queue()
    return RESEARCH_QUEUES[task_id]


def cleanup_research_task(task_id: str):
    """Remove task from registries when completed/cancelled."""
    RESEARCH_CALLBACKS.pop(task_id, None)
    RESEARCH_QUEUES.pop(task_id, None)
```

---

### 2.4 Create Orchestrator Graph

**File**: `Ai-Analysis-Engine/src/research/graph.py`

```python
from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from src.research.state import ResearchState

# Import nodes (create later in Phase 2.5)

graph = StateGraph(ResearchState)

# Add nodes
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("web_searcher", web_search_node)
graph.add_node("deep_reader", deep_reader_node)
graph.add_node("topic_specialist", specialist_node)
graph.add_node("reflector", reflector_node)
graph.add_node("fact_checker", fact_check_node)
graph.add_node("synthesizer", synthesizer_node)

# Entry point
graph.set_entry_point("orchestrator")

# Fan-out: Orchestrator → parallel agents (critical: Send API)
def route_from_orchestrator(state: ResearchState) -> list[Send]:
    """Send sub-tasks to agents in PARALLEL."""
    sends = []
    for sub_task in state.get("sub_tasks", []):
        if sub_task.get("status") != "pending":
            continue
        agent = sub_task.get("assigned_agent")
        if agent:
            sends.append(Send(agent, {"**state", "current_sub_task": sub_task}))
    return sends

graph.add_conditional_edges(
    "orchestrator",
    route_from_orchestrator,
    ["web_searcher", "deep_reader", "topic_specialist"]
)

# All agents → Reflector (parallel execution)
graph.add_edge("web_searcher", "reflector")
graph.add_edge("deep_reader", "reflector")
graph.add_edge("topic_specialist", "reflector")

# Reflector: loop back on gaps, or finish to synthesizer
def route_from_reflector(state: ResearchState) -> str:
    iteration = state.get("iteration", 0)
    gaps = state.get("gaps", [])
    status = state.get("status")

    if iteration >= state.get("max_iterations", 5):
        return "synthesizer"
    if gaps and status != "complete":
        return "orchestrator"  # Re-decompose with gaps
    return "synthesizer"

graph.add_conditional_edges(
    "reflector",
    route_from_reflector,
    {"orchestrator": "orchestrator", "synthesizer": "synthesizer"}
)

# Synthesizer → Fact Checker → END
graph.add_edge("synthesizer", "fact_checker")
graph.add_edge("fact_checker", END)

# Compile with RECURSION LIMIT = 50
app = graph.compile(recursion_limit=50)
```

**Critical additions from v2**:
- `Send` API for parallel fan-out (3x speed improvement)
- `recursion_limit=50` prevents infinite loops
- `route_from_reflector` handles self-correction loop
- `fact_checker` runs AFTER synthesis for verification

---

### 2.5 Create 7 Agent Nodes

**Directory**: `Ai-Analysis-Engine/src/research/nodes/`

#### Node 1: `coordinator.py` (Orchestrator)

```python
def orchestrator_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Read objective from state
    2. If iteration == 0: decompose into sub_tasks
    3. If iteration > 0: use gaps to create new sub_tasks
    4. Assign each sub_task to appropriate agent
    5. Return updated state
    """
    # Dynamic agent assignment based on question keywords:
    # - "search web" / "find" → web_searcher
    # - "read" / "analyze" → deep_reader
    # - "financial" / "annual report" → specialist
    #
    # Return state with sub_tasks populated
```

#### Node 2: `web_agent.py` (Web Researcher)

```python
def web_search_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Read assigned sub-tasks from state["sub_tasks"]
    2. Use tavily API to search web
    3. Filter out URLs already in state["searched_urls"]
    4. Save new sources to database
    5. Append findings to state["findings"]
    """
    # CRITICAL: Check searched_urls before querying API
```

#### Node 3: `reader.py` (Deep Content Reader)

```python
def deep_reader_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Read URLs from findings["sources"]
    2. Scrape full page content (Playwright for JS sites)
    3. Extract key points using LLM
    4. Handle failures gracefully
    """
```

#### Node 4: `specialist.py` (Domain Specialist)

```python
def specialist_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Read task_type (company, medical, academic)
    2. Select domain-specific prompt template
    3. Analyze findings through lens
    4. Identify gaps and contradictions
    """
```

#### Node 5: `reflector.py` (Self-Correction)

```python
def reflector_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Review all findings collected so far
    2. Check: Does this answer the original objective?
    3. Identify gaps: "We have X but missing Y"
    4. If gaps found: add to state["gaps"]
    5. Increment iteration counter
    6. Return updated state
    """
```

#### Node 6: `fact_checker.py` (Post-Synthesis Verification)

```python
def fact_check_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Extract claims from synthesis
    2. Cross-reference each claim against source findings
    3. Flag unsupported claims
    4. Add verification notes to synthesis
    5. Return verified synthesis
    """
```

#### Node 7: `synthesizer.py` (Final Report Writer)

```python
def synthesizer_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Read all findings from state
    2. Generate structured report:
       - Executive summary
       - Key findings (with inline citations)
       - Sources section
       - Confidence assessment
    3. Return synthesis text
    """
```

**Error handling pattern** (apply to ALL nodes):
```python
def web_search_node(state: ResearchState, callback) -> ResearchState:
    try:
        # Agent logic
        return {**state, "findings": new_findings}
    except Exception as e:
        # Log error, don't crash graph
        callback("error", {"message": str(e)})
        return state
```

---

### 2.6 Add API Endpoints

**File**: `Ai-Analysis-Engine/src/app.py` (MODIFY)

**Add imports**:
```python
from src.research.registry import (
    register_research_callback,
    get_research_queue,
    cleanup_research_task,
)
from src.research.state import ResearchState
from src.research.graph import build_research_orchestrator
from src.research.nodes import *
```

**Add endpoints** (after line 420 or existing research endpoints):

```python
# POST /api/research/start — Create research task
@app.post("/api/research/start", status_code=201)
async def start_research_task(request: ResearchStartRequest):
    task_id = str(uuid.uuid4())

    # Create DB record with status_code=201
    with get_db_session() as db:
        db_task = ResearchTask(
            task_id=task_id,
            user_id=request.user_id,
            objective=request.objective,
            task_type=request.task_type,
            depth=request.depth or 3,
            status="pending"
        )
        db.add(db_task)
        db.commit()

    # Initialize SSE queue
    queue = get_research_queue(task_id)

    # Register callback
    def on_event(event_type: str, data: dict):
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

        # Log to task_events table
        with get_db_session() as db:
            event = TaskEvent(
                task_id=task_id,
                event_type=event_type,
                data=data,
                created_at=datetime.utcnow(timezone.utc)
            )
            db.add(event)
            db.commit()

    register_research_callback(task_id, on_event)

    # Determine max_iterations based on depth
    max_iters = {1: 2, 3: 5, 5: 10}[request.depth or 3]

    # Start background thread
    def run_task():
        try:
            compiled = build_research_orchestrator()
            initial_state = ResearchState(
                task_id=task_id,
                user_id=request.user_id,
                objective=request.objective,
                task_type=request.task_type,
                depth=request.depth or 3,
                max_iterations=max_iters
            )
            result = compiled.invoke(initial_state)
            # Update status to complete
            with get_db_session() as db:
                db_task = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
                if db_task:
                    db_task.status = "complete"
                    db_task.synthesis = result.get("synthesis")
                    db_task.completed_at = datetime.utcnow(timezone.utc)
                    db.commit()
        except Exception:
            with get_db_session() as db:
                db_task = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
                if db_task:
                    db_task.status = "failed"
                    db_task.completed_at = datetime.utcnow(timezone.utc)
                    db.commit()
        finally:
            cleanup_research_task(task_id)

    threading.Thread(target=run_task, daemon=True).start()

    return {"task_id": task_id}

# GET /api/research/sessions
@app.get("/api/research/sessions")
async def list_research_tasks(limit: int = 20, offset: int = 0):
    with get_db_session() as db:
        tasks = db.query(ResearchTask).order_by(ResearchTask.created_at.desc()).offset(offset).limit(limit).all()
        return {
            "tasks": [{"task_id": t.id, "objective": t.objective, "status": t.status, "created_at": t.created_at} for t in tasks],
            "total": db.query(func.count()).filter(ResearchTask.task_id # would need func import)
        }

# GET /api/research/{task_id}/stream
@app.get("/api/research/{task_id}/stream")
async def stream_research_events(task_id: str):
    queue = get_research_queue(task_id)

    async def event_generator():
        try:
            while True:
                event_type, data = await asyncio.wait_for(queue.get(), timeout=60.0)
                if event_type == "__done__":
                    break
                yield _named_sse(event_type, data)
        except asyncio.TimeoutError:
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# DELETE /api/research/{task_id} (cancellation)
@app.delete("/api/research/{task_id}")
async def cancel_research_task(task_id: str):
    cleanup_research_task(task_id)
    return {"success": True}

# GET /api/research/{task_id}/sources
@app.get("/api/research/{task_id}/sources")
async def get_research_sources(task_id: str):
    with get_db_session() as db:
        sources = db.query(ResearchSource).filter(ResearchSource.task_id == task_id).all()
        return {"sources": [{"url": s.url, "title": s.title, "credibility_score": s.credibility_score} for s in sources]}
```

---

## Phase 3: Frontend (6 hours)

### 3.1 Components to Create

**File**: `src/components/ResearchModeCard.tsx`
- Display 3 options: Data Analysis vs AI Research vs Documents
- Use existing Card / lucide-react

**File**: `src/components/TaskList.tsx`
- Fetch from `/api/research/sessions?limit=20`
- Show objective, depth, status with badges
- Pagination and cancel button
- Navigate to detail page

**File**: `src/components/ResearchStage.tsx`
- Fetch events from `/api/research/{task_id}`
- Group by agent_role
- Border colors:
  - orchestrator (amber)
  - web_searcher (blue)
  - deep_reader (cyan)
  - topic_specialist (purple)
  - reflector (orange)
  - synthesizer (green)
  - fact_checker (red)
- Collapse 3+ consecutive events from same agent
- **SSE reconnection** (max 3 retries with exponential backoff)

**File**: `src/components/SourcesPanel.tsx` (NEW)
- Fetch from `/api/research/{task_id}/sources`
- Show title, URL, credibility score sorted descending
- Dedup display

**File**: `src/routes/research/$id.tsx`
- Dynamic route
- Layout: header + sidebar (actions + sources) + main (events)
- EventSource with reconnection
- Cancel button

**File**: `src/routes/research/index.tsx`
- Research hub: Task list ✗ Task detail
- "New Research" button → modal input

**File**: `src/routes/index.tsx` (MODIFY)
- Add "AI Research" card
- Navigate to /research

---

## Phase 4: Testing (4 hours)

### 4.1 Backend Tests

```bash
# Database migration
sqlite3 verum.db ".tables"

# Create task
curl -X POST http://localhost:8000/api/research/start \
  -H "Content-Type: application/json" \
  -d '{"objective":"Test objective", "task_type":"general", "depth":3}'

# Check events stream
curl http://localhost:8000/api/research/<task_id>/stream

# Cancel task
curl -X DELETE http://localhost:8000/api/research/<task_id>
```

### 4.2 Expectations

✅ Database: 3 tables created with indexes
✅ Graph: Send API spawns parallel agents
✅ recursion_limit=50 enforced
✅ Source dedup via url_hash working
✅ Task cancellation stops swarm
✅ Frontend: SSE reconnection logic functional
✅ Existing features: Analysis chat unchanged

---

## What NOT to Change

- `src/analysis_engine/graph.py` (existing)
- `src/analysis_engine/nodes/*.py` (existing)
- `src/analysis_engine/registry.py` (don't modify RUN_CALLBACKS)
- `src/analysis_engine/llm/client.py`
- Existing frontend pages

---

## Total Time Estimate

**Backend**: 10 hours
**Frontend**: 6 hours
**Testing**: 4 hours
**Total**: 20 hours

**Implementation order**: Phase 1 → Phase 2 → Phase 3 → Phase 4
