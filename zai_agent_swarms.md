Verum Agent Swarm - Implementation Instructions (Revised)
Overview
Add research capability to Verum. It sits on top of existing analysis engine. Don't rewrite anything.

Key principle: Reuse existing RUN_CALLBACKS system with a separate registry for research tasks. Extend events to support research tasks. Keep everything the same.

Architecture: 7-agent swarm with parallel fan-out, self-correction loop, and explicit recursion limits.

Backend Implementation (Python - LangGraph)
1. Add Database Tables for Research Tasks
Location: src/analysis_engine/db/models.py

Action: Add these THREE new models AFTER existing models:

class ResearchTask(Base):    __tablename__ = "research_tasks"    id = Column(String(36), primary_key=True)    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)    run_id = Column(String(36), ForeignKey("runs.id"), nullable=True)    objective = Column(Text, nullable=False)    task_type = Column(String(50), nullable=False)  # general, company, academic, medical    depth = Column(Integer, default=3)  # 1=quick, 3=standard, 5=deep    status = Column(String(50), default="pending")  # pending, planning, researching, synthesizing, complete, failed, cancelled    sub_tasks = Column(JSON, default=list)  # decomposed task list with status    findings = Column(JSON, default=list)  # [{agent, finding, confidence}]    synthesis = Column(Text, nullable=True)    error = Column(Text, nullable=True)    created_at = Column(DateTime, default=datetime.utcnow)    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)    completed_at = Column(DateTime, nullable=True)class ResearchSource(Base):    """Separate table for tracking sources — enables citation and dedup"""    __tablename__ = "research_sources"    id = Column(Integer, primary_key=True, autoincrement=True)    task_id = Column(String(36), ForeignKey("research_tasks.id"), nullable=False)    url = Column(Text, nullable=False)    url_hash = Column(String(64), nullable=False, index=True)  # SHA256 for dedup    title = Column(Text, nullable=True)    source_type = Column(String(50))  # web, news, paper, document    content_snippet = Column(Text)  # first 500 chars    credibility_score = Column(Float, nullable=True)  # 0.0-1.0    retrieved_by = Column(String(50))  # which agent found it    created_at = Column(DateTime, default=datetime.utcnow)class TaskEvent(Base):    __tablename__ = "task_events"    id = Column(Integer, primary_key=True, autoincrement=True)    task_id = Column(String(36), ForeignKey("research_tasks.id"), nullable=False)    event_type = Column(String(100), nullable=False)  # plan, search, scrape, analyze, reflect, synthesize, error    event_level = Column(String(20), nullable=False)  # info, warning, error, debug    agent_role = Column(String(50), nullable=True)    agent_id = Column(String(50), nullable=True)  # for parallel agent instances    parent_event_id = Column(Integer, nullable=True)  # hierarchical events    data = Column(JSON, nullable=False)    created_at = Column(DateTime, default=datetime.utcnow)
Why 3 tables not 2: ResearchSource enables citation tracking, URL deduplication (via url_hash), and credibility scoring. Without it, you can't produce trustworthy research.

2. Create Research State File
File: src/research/state.py (NEW)

Action: Create LangGraph state using TypedDict (NOT Pydantic — LangGraph convention):

python

from typing import TypedDict, Literal, Annotated
from operator import add

class SubTask(TypedDict):
    id: str
    question: str
    assigned_agent: str
    status: str  # pending, in_progress, complete, failed
    result: str | None

class Finding(TypedDict):
    agent: str
    sub_task_id: str
    content: str
    sources: list[str]  # URLs
    confidence: float

class ResearchState(TypedDict):
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

    # Accumulated results (use `add` reducer for parallel merge)
    findings: Annotated[list[Finding], add]
    sources: Annotated[list[str], add]  # URLs collected
    searched_urls: Annotated[list[str], add]  # dedup tracking

    # Control flow
    iteration: int
    max_iterations: int
    gaps: list[str]  # identified by reflector
    status: str

    # Output
    synthesis: str | None
    error: str | None
Connection: This state flows through orchestrator graph (like PipelineState flows through analysis graph). The Annotated[list, add] reducer is critical — it allows parallel agents to merge results without overwriting.

3. Create Agent Taxonomy
File: src/research/agents.py (NEW)

Action: Define 7 agents with clear separation of concerns:

python

AGENT_CONFIG = {
    "orchestrator": {
        "depth": 1,
        "role": "Manager",
        "model": "claude-sonnet",  # strong reasoning for planning
        "capabilities": ["decompose_task", "assign_team", "evaluate_completeness"],
        "max_iterations": 5,
    },
    "web_searcher": {
        "depth": 2,
        "role": "Web Researcher",
        "model": "gpt-4o-mini",  # fast, cheap for search queries
        "capabilities": ["search_web", "search_news"],
        "max_iterations": 10,
        "token_budget": 8000,
    },
    "deep_reader": {
        "depth": 2,
        "role": "Content Reader",
        "model": "gpt-4o-mini",
        "capabilities": ["scrape_url", "read_pdf", "extract_key_points"],
        "max_iterations": 8,
        "token_budget": 15000,
    },
    "topic_specialist": {
        "depth": 3,
        "role": "Domain Analyst",
        "model": "claude-sonnet",  # needs reasoning for analysis
        "capabilities": ["analyze_findings", "identify_gaps", "assess_credibility"],
        "max_iterations": 5,
        "token_budget": 12000,
    },
    "reflector": {
        "depth": 1,
        "role": "Critic",
        "model": "claude-sonnet",
        "capabilities": ["evaluate_completeness", "identify_gaps", "suggest_next_steps"],
        "max_iterations": 3,
    },
    "fact_checker": {
        "depth": 3,
        "role": "Verifier",
        "model": "gpt-4o-mini",
        "capabilities": ["verify_claims", "cross_reference", "flag_contradictions"],
        "max_iterations": 5,
        "token_budget": 8000,
    },
    "synthesizer": {
        "depth": 1,
        "role": "Writer",
        "model": "claude-sonnet",  # best writing quality
        "capabilities": ["combine_findings", "format_report", "generate_citations"],
        "max_iterations": 2,
    },
}
Why 7 agents not 4:

Reflector: Without it, the swarm has no self-correction. It will produce incomplete research and stop.
Deep Reader: Search snippets are ~200 chars. Real research needs full page content.
Fact Checker: Cross-references claims across sources. Critical for trust.
Synthesizer separate from Orchestrator: Orchestrator plans; Synthesizer writes. Different prompts, different models potentially.
4. Create Orchestrator Graph
File: src/research/graph.py (NEW)

Action: Write coordination graph with parallel fan-out and reflection loop:

python

from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from src.research.state import ResearchState

graph = StateGraph(ResearchState)

# --- Nodes ---
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("web_searcher", web_search_node)
graph.add_node("deep_reader", deep_reader_node)
graph.add_node("topic_specialist", specialist_node)
graph.add_node("reflector", reflector_node)
graph.add_node("fact_checker", fact_check_node)
graph.add_node("synthesizer", synthesizer_node)

# --- Entry ---
graph.set_entry_point("orchestrator")

# --- Fan-out: Orchestrator → parallel agents ---
def route_from_orchestrator(state: ResearchState) -> list[Send]:
    """Send sub-tasks to agents in PARALLEL using Send API"""
    sends = []
    for sub_task in state["sub_tasks"]:
        if sub_task["status"] != "pending":
            continue
        agent = sub_task["assigned_agent"]
        sends.append(Send(agent, {**state, "current_sub_task": sub_task}))
    return sends

graph.add_conditional_edges(
    "orchestrator",
    route_from_orchestrator,
    ["web_searcher", "deep_reader", "topic_specialist"]
)

# --- All agents → Reflector ---
graph.add_edge("web_searcher", "reflector")
graph.add_edge("deep_reader", "reflector")
graph.add_edge("topic_specialist", "reflector")

# --- Reflector: loop back or finish ---
def route_from_reflector(state: ResearchState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return "synthesizer"
    if len(state["gaps"]) > 0 and state["status"] != "complete":
        return "orchestrator"  # re-plan with gaps
    return "synthesizer"

graph.add_conditional_edges(
    "reflector",
    route_from_reflector,
    {"orchestrator": "orchestrator", "synthesizer": "synthesizer"}
)

# --- Synthesizer → Fact Checker → END ---
graph.add_edge("synthesizer", "fact_checker")
graph.add_edge("fact_checker", END)

# --- Compile with RECURSION LIMIT ---
app = graph.compile(recursion_limit=50)
Critical additions vs original plan:

Send API for parallel fan-out (3x speed improvement)
recursion_limit=50 prevents infinite loops
Reflector → Orchestrator loop for self-correction
Fact checker runs AFTER synthesis (verifies final output)
Annotated[list, add] reducer merges parallel results automatically
5. Create Agent Nodes
Directory: src/research/nodes/ (NEW)

python

# coordinator.py — Task decomposition + dynamic agent assignment
def orchestrator_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Read objective from state
    2. If iteration == 0: decompose into sub_tasks
    3. If iteration > 0: use reflector gaps to create new sub_tasks
    4. Assign each sub_task to appropriate agent based on capability
    5. Return updated state with sub_tasks
    """
    # Dynamic agent selection logic:
    # - "search the web for X" → web_searcher
    # - "read this URL in depth" → deep_reader
    # - "analyze financial implications" → topic_specialist
    # - "find recent news about X" → web_searcher (news mode)


# web_agent.py — Search coordination with DEDUPLICATION
def web_search_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Use Tavily/Serper API for search
    2. Filter out URLs already in state['searched_urls']
    3. For each new URL: save to ResearchSource table
    4. Return findings + sources
    """
    # CRITICAL: Check searched_urls before querying API
    # to avoid duplicate searches across parallel agents


# reader.py — Deep content extraction
def deep_reader_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Take URLs from findings
    2. Scrape full page content (Playwright for JS-heavy sites)
    3. Extract key points using LLM
    4. Handle failures gracefully (404, timeout, paywall)
    5. Return structured findings
    """


# specialist.py — Domain analysis (dynamically specialized)
def specialist_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Read task_type from state (company, medical, academic, general)
    2. Select domain-specific prompt template
    3. Analyze findings through domain lens
    4. Identify gaps and contradictions
    5. Return analysis + gaps
    """


# reflector.py — Self-correction loop
def reflector_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Review all findings collected so far
    2. Check: Does this answer the original objective?
    3. Identify gaps: "We have X but missing Y"
    4. If gaps found: add to state['gaps']
    5. Increment iteration counter
    6. Return updated state
    """


# fact_checker.py — Post-synthesis verification
def fact_check_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Extract claims from synthesis
    2. Cross-reference each claim against source findings
    3. Flag unsupported claims
    4. Add verification notes to synthesis
    5. Return verified synthesis
    """


# synthesizer.py — Final report generation
def synthesizer_node(state: ResearchState, callback) -> ResearchState:
    """
    1. Read all findings from state
    2. Resolve contradictions (prefer higher credibility sources)
    3. Generate structured report with:
       - Executive summary
       - Key findings (with inline citations)
       - Sources section
       - Confidence assessment
    4. Return synthesis text
    """
Connection: Each node uses existing LLM client with explore_loop pattern. Each node must handle its own errors and return state (never raise — log error to state and continue).

6. Create Research Callback Registry (SEPARATE from RUN_CALLBACKS)
File: src/research/registry.py (NEW)

Action: Create a separate registry — don't hack existing RUN_CALLBACKS:

python

import asyncio
from collections import defaultdict
from typing import Callable

# Separate registry for research tasks
RESEARCH_CALLBACKS: dict[str, Callable] = {}
RESEARCH_QUEUES: dict[str, asyncio.Queue] = {}

def register_research_callback(task_id: str, callback: Callable):
    RESEARCH_CALLBACKS[task_id] = callback

def get_research_queue(task_id: str) -> asyncio.Queue:
    if task_id not in RESEARCH_QUEUES:
        RESEARCH_QUEUES[task_id] = asyncio.Queue()
    return RESEARCH_QUEUES[task_id]

def cleanup_research_task(task_id: str):
    """Call when task completes or is cancelled"""
    RESEARCH_CALLBACKS.pop(task_id, None)
    RESEARCH_QUEUES.pop(task_id, None)
Why separate: The original plan's RUN_CALLBACKS["research_"] = None is a hack that could break the existing analysis engine. A separate dict with the same pattern is cleaner and safer.

7. Create API Endpoints
File: src/app.py (MODIFY — add new endpoints)

python

from src.research.registry import (
    register_research_callback,
    get_research_queue,
    cleanup_research_task,
)

# POST /api/research/start — Create research task
@app.post("/api/research/start", status_code=201)
async def start_research_task(request: ResearchStartRequest):
    """
    Body: { objective: str, task_type: str, depth: int }
    Returns: { task_id: str, stream_url: str }
    """
    task_id = str(uuid4())

    # 1. Create ResearchTask in DB
    # 2. Initialize SSE queue
    # 3. Register callback
    # 4. Start background thread running orchestrator graph
    # 5. Return task_id + stream_url

    # CRITICAL: Set max_iterations based on depth:
    # depth=1 → max_iterations=2, depth=3 → max_iterations=5, depth=5 → max_iterations=10


# GET /api/research/sessions — List all tasks (with pagination)
@app.get("/api/research/sessions")
async def list_research_tasks(limit: int = 20, offset: int = 0):
    # Query research_tasks ORDER BY created_at DESC
    # Return { tasks: [...], total: int }


# GET /api/research/{task_id} — Get single task with events
@app.get("/api/research/{task_id}")
async def get_research_task(task_id: str):
    # Query research_tasks + task_events + research_sources
    # Return { task, events: [...], sources: [...] }


# GET /api/research/{task_id}/stream — SSE stream
@app.get("/api/research/{task_id}/stream")
async def stream_research_events(task_id: str):
    # Get queue from RESEARCH_QUEUES
    # Yield events same as existing /api/stream/{run_id}
    # Handle client disconnect: cleanup_research_task(task_id)


# DELETE /api/research/{task_id} — Cancel running task  ← NEW
@app.delete("/api/research/{task_id}")
async def cancel_research_task(task_id: str):
    # Set status = "cancelled" in DB
    # Send cancellation event to queue
    # cleanup_research_task(task_id)
    # Return { success: true }


# GET /api/research/{task_id}/sources — Get all sources  ← NEW
@app.get("/api/research/{task_id}/sources")
async def get_research_sources(task_id: str):
    # Query research_sources where task_id
    # Return { sources: [...] }
Critical additions:

DELETE endpoint for cancellation (long-running tasks MUST be cancellable)
GET /sources endpoint for citation display
Pagination on list endpoint
status_code=201 on create (REST convention)
Client disconnect handling on SSE
8. Run Database Migration
Location: migrations/add_research_tables.sql

sql

CREATE TABLE research_tasks (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    run_id VARCHAR(36),
    objective TEXT NOT NULL,
    task_type VARCHAR(50) NOT NULL,
    depth INTEGER DEFAULT 3,
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

CREATE TABLE research_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id VARCHAR(36) NOT NULL,
    url TEXT NOT NULL,
    url_hash VARCHAR(64) NOT NULL,
    title TEXT,
    source_type VARCHAR(50),
    content_snippet TEXT,
    credibility_score REAL,
    retrieved_by VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES research_tasks(id)
);

CREATE INDEX idx_sources_url_hash ON research_sources(url_hash);
CREATE INDEX idx_sources_task_id ON research_sources(task_id);

CREATE TABLE task_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id VARCHAR(36) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_level VARCHAR(20) NOT NULL,
    agent_role VARCHAR(50),
    agent_id VARCHAR(50),
    parent_event_id INTEGER,
    data JSON NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES research_tasks(id)
);

CREATE INDEX idx_events_task_id ON task_events(task_id);
Changes from original:

DATETIME instead of TEXT for timestamps (proper sorting)
Added research_sources table (3rd table)
Added indexes for query performance
url_hash index enables O(1) dedup lookups
Frontend Implementation (TypeScript - Lovable)
1. Create Mode Selection Component
File: src/components/ResearchModeCard.tsx (NEW)

Display 3 options: Data Analysis vs AI Research vs Documents
Use existing Card component
Use lucide-react icons
On click, navigate to separate view
2. Create Task List Component
File: src/components/TaskList.tsx (NEW)

Fetch from /api/research/sessions?limit=20&offset=0
Display objective, depth, status with color badges
Add pagination (load more button)
Add cancel button for running tasks (calls DELETE endpoint)
Navigation to task detail page
3. Create Event Timeline Component
File: src/components/ResearchStage.tsx (NEW)

Fetch events from /api/research/{task_id}
Group events by agent_role
Border colors: orchestrator=amber, searchers=blue, specialist=purple, reflector=orange, synthesizer=green, fact_checker=red
Collapse 3+ consecutive events from same agent
Add SSE reconnection logic (retry on disconnect, max 3 retries)
Add error state display (red banner on event_level: "error")
Append synthesis at end (green box when complete)
typescript

// SSE reconnection pattern
const connectStream = () => {
  const source = new EventSource(`/api/research/${taskId}/stream`);
  source.onmessage = (e) => { /* handle event */ };
  source.onerror = () => {
    source.close();
    if (retries < 3) {
      retries++;
      setTimeout(connectStream, 2000 * retries); // exponential backoff
    }
  };
};
4. Create Research Detail Page
File: src/routes/research/$taskId.tsx (NEW — dynamic route)

Fetch task from /api/research/{task_id}
Setup EventSource with reconnection (see above)
Layout: header + sidebar (actions + sources) + main (events)
Sources sidebar: list all sources from /api/research/{task_id}/sources
Cancel button: calls DELETE, shows "cancelled" state
Copy structure from existing analysis page
5. Update Homepage
File: src/routes/index.tsx (MODIFY)

Add "AI Research" card using ResearchModeCard
On click → navigate to /research
6. Create Research Hub Route
File: src/routes/research/index.tsx (NEW)

Task list on left (with pagination)
Selected task detail on right
"New Research" button → modal with objective input, type selector, depth slider
7. Create Sources Panel Component ← NEW
File: src/components/SourcesPanel.tsx (NEW)

Fetch from /api/research/{task_id}/sources
Display: title, URL (clickable), type badge, credibility score
Sort by credibility (descending)
Dedup display by URL
Error Handling Patterns (NEW SECTION)
Each agent node must follow this pattern:

python

async def web_search_node(state: ResearchState, callback) -> ResearchState:
    try:
        # ... agent logic ...
        return {**state, "findings": new_findings, "status": "searching"}
    except RateLimitError:
        # Log warning, return partial results
        await emit_event(callback, "warning", "Search rate limited, retrying...")
        await asyncio.sleep(5)
        return state
    except Exception as e:
        # Log error to state, don't crash the graph
        await emit_event(callback, "error", f"Web search failed: {str(e)}")
        return {**state, "error": str(e)}
Error categories to handle:

Error Type
Where
Strategy
LLM API timeout	All agents	Retry 2x, then skip sub-task
Search rate limit	web_searcher	Exponential backoff (5s, 10s, 20s)
Scraping 404/timeout	deep_reader	Log, skip URL, continue
Paywall/block	deep_reader	Log, mark source as "limited"
Graph recursion limit	graph.py	Catch, force synthesizer
DB write failure	callback	Log, continue (events are best-effort)

Implementation Priority (Revised)
Phase 1: Database (2 hours)
Create migration SQL (3 tables + indexes)
Add models to models.py
Run migration and verify
Test: sqlite3 verum.db ".tables" shows all 3
Phase 2: Backend Core (8 hours)
Create state.py with TypedDict + reducers
Create agents.py with 7 agent configs
Create registry.py (separate from RUN_CALLBACKS)
Create nodes/coordinator.py (orchestrator)
Create nodes/web_agent.py (search + dedup)
Create nodes/reader.py (scraper)
Create nodes/specialist.py (domain analysis)
Create nodes/reflector.py (self-correction)
Create nodes/fact_checker.py (verification)
Create nodes/synthesizer.py (report writer)
Create graph.py with Send API + recursion limit
Add API endpoints to app.py (6 endpoints)
Phase 3: Frontend Core (6 hours)
ResearchModeCard.tsx
TaskList.tsx (with pagination + cancel)
ResearchStage.tsx (with SSE reconnection)
SourcesPanel.tsx (NEW)
research/$taskId.tsx route
research/index.tsx route
Update index.tsx homepage
Phase 4: Testing & Polish (6 hours)
Unit test each agent with mocked LLM
Integration test: full graph with real search
Test cancellation mid-research
Test SSE reconnection
Test recursion limit enforcement
Error handling edge cases
Documentation
Total time estimate: ~22 hours (backend 10h + frontend 6h + testing 6h)

What NOT to Change
src/analysis_engine/graph.py
src/analysis_engine/nodes/*.py
src/analysis_engine/tools/sandbox/*.py
src/analysis_engine/registry.py — don't touch (separate research registry)
src/analysis_engine/llm/client.py
Existing frontend pages
Test Commands
bash

# Test database
sqlite3 verum.db ".tables"  # Should show: research_tasks, research_sources, task_events
sqlite3 verum.db "SELECT COUNT(*) FROM research_sources;"

# Test API - create task
curl -X POST http://localhost:8000/api/research/start \
  -H "Content-Type: application/json" \
  -d '{"objective":"Research OpenAI's business model", "task_type":"company", "depth":3}'
# Expected: {"task_id": "...", "stream_url": "/api/research/.../stream"}

# Test stream
curl http://localhost:8000/api/research/<task_id>/stream

# Test cancellation
curl -X DELETE http://localhost:8000/api/research/<task_id>

# Test sources
curl http://localhost:8000/api/research/<task_id>/sources

# Test pagination
curl "http://localhost:8000/api/research/sessions?limit=10&offset=0"
Success Criteria
✅ Database: 3 tables created with indexes
✅ Backend: 7 agents deployed via LangGraph with Send API parallelization
✅ Backend: recursion_limit=50 enforced
✅ Backend: Source deduplication via url_hash working
✅ Backend: 6 API endpoints respond correctly
✅ Backend: Task cancellation works mid-research
✅ Frontend: SSE stream with reconnection logic
✅ Frontend: Sources panel shows all citations
✅ Frontend: Cancel button stops running task
✅ Error handling: Agent failures don't crash graph
✅ Existing features: Data analysis still works (nothing broken)
✅ Cost: Total token usage stays within budget per depth level