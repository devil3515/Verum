# Verum Agent Swarm - Implementation Instructions

## Overview (Lazy Version)

Add research capability to Verum. It sits on top of existing analysis engine. Don't rewrite anything.

**Key principle**: Reuse existing `RUN_CALLBACKS` system. Extend events to support research tasks. Keep everything the same.

---

## Backend Implementation (Python - LangGraph)

### 1. Add Database Tables for Research Tasks

**Location**: `src/analysis_engine/db/models.py`

**Action**: Add these two new models AFTER existing models:

```python
class ResearchTask(Base):
    __tablename__ = "research_tasks"
    # Add tables: task_id, user_id, objective, task_type, depth, status, findings, evidence, synthesis
    # Link to existing `users.id` and `runs.id`

class TaskEvent(Base):
    __tablename__ = "task_events"
    # Add tables: task_id, event_type, event_level, agent_role, data, success, error
    # Link to `research_tasks.id`
```

**Why**: Stores research state in database. Enables SSE streams to persist.

---

### 2. Create New Research State File

**File**: `src/research/state.py` (NEW)

**Action**: Create lightweight state matching existing `PipelineState` pattern:

```python
class ResearchTask(BaseModel):
    # Same pattern as PipelineState:
    # - task_id, user_id, run_id
    # - objective, task_type, depth
    # - status, findings, evidence, synthesis
    # - events list (extend RunEvent pattern from existing app.py)
```

**Connection**: This state flows through orchestrator graph (like PipelineState flows through analysis graph).

---

### 3. Create New Agent Taxonomy

**File**: `src/research/agents.py` (NEW)

**Action**: Define agent types extending existing patterns (don't rewrite client):

```python
AGENTS = {
    "orchestrator": {
        "depth": 1,
        "role": "Manager",
        "capabilities": ["decompose_task", "assign_team", "synthesize_outcomes"]
    },
    "research_web": {
        "depth": 2,
        "role": "Web Researcher",
        "capabilities": ["search", "read_urls", "verify_claims"]
    },
    "research_doc": {
        "depth": 2,
        "role": "Document Reader",
        "capabilities": ["read_files", "summarize", "extract_keywords"]
    },
    "research_paper": {
        "depth": 3,
        "role": "Paper Analyst",
        "capabilities": ["search_papers", "cite_analysis", "track_trends"]
    }
}
```

**Why**: LLM decisions based on required capability (not hardcoded paths).

---

### 4. Create Orchestrator Graph

**File**: `src/research/graph.py` (NEW)

**Action**: Write coordination graph matching existing `analysis_engine/graph.py` pattern:

- Entry point: `orchestrator_agent`
- Branches to specialized agents based on task_type
- Each agent can recurse back to orchestrator for next sub-task
- Final synthesis node with LLM call to produce summary
- End point: END

**Reuse**: Copy structure from `graph.py` - replace node names.

---

### 5. Create Agent Nodes

**Directory**: `src/research/nodes/` (NEW)

**Files to create**:

```python
# coordinator.py - Task decomposition logic (matches planner_node.py pattern)
def coordinator_node(state, callback):
    # Read objective from state
    # Decide what agents to deploy
    # Route to specialized agents
    # Return sub-task assignments

# web_agent.py - Search coordination (matches analysis_node.py pattern)
def web_research_node(state, callback):
    # Use tavily search integration in tools/
    # Call agent to read URLs
    # Extract findings
    # Return results

# doc_agent.py - Document processing (matches cleaning_node.py pattern)
def document_agent(state, callback):
    # Read files (PDFs, docs)
    # Summarize content
    # Extract key points

# paper_agent.py - Academic analysis (matches verification_node.py pattern)
def paper_agent(state, callback):
    # Search arXiv/PubMed
    # Analyze citations
    # Track trends
```

**Connection**: Each node uses existing LLM client with `explore_loop` pattern.

---

### 6. Create API Endpoints

**File**: `src/app.py` (MODIFY - add new endpoints)

**Actions**:

```python
# Add these endpoints TO existing app.py (after line 420 or before endpoints)

# POST /api/research/start - Create research task
@app.post("/api/research/start")
async def start_research_task(request):
    # Generate task_id
    # Create ResearchTask in database (mart stores task_id)
    # Initialize queue for SSE stream
    # Register callback in RUN_CALLBACKS (reuse existing registry)
    # Start background thread running orchestrator graph
    # Return {task_id, stream_url}

# GET /api/research/sessions - List all tasks
@app.get("/api/research/sessions")
async def list_research_tasks():
    # Query research_tasks table
    # Return list of {task_id, objective, status, created_at}

# GET /api/research/{task_id} - Get single task
@app.get("/api/research/{task_id}")
async def get_research_task(task_id: str):
    # Query research_tasks where task_id = XXX
    # Query task_events where task_id = XXX (ordered by timestamp)
    # Return {task, events: [{type, data, timestamp}]}

# GET /api/research/{task_id}/stream - SSE stream
@app.get("/api/research/{task_id}/stream")
async def stream_research_events(task_id: str):
    # Get queue from _RESEARCH_QUEUES dict
    # Yield event source same as existing /api/stream/{run_id}
    # Use existing event_generator() pattern
```

**Key**: Reuse existing `_RUN_QUEUES`, `RUN_CALLBACKS` registry. Don't create new state management.

---

### 7. Run Database Migration

**Location**: Create SQL file `migrations/add_research_tables.sql`

**Action**: Run this SQL to create new tables:

```sql
CREATE TABLE research_tasks (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36),
    objective TEXT,
    task_type VARCHAR(50),
    depth INTEGER,
    status VARCHAR(50),
    findings JSON,
    evidence JSON,
    synthesis TEXT,
    created_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE task_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id VARCHAR(36) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_level VARCHAR(20) NOT NULL,
    agent_role VARCHAR(50),
    data JSON NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES research_tasks(id)
);
```

**Commands to run**:
```bash
cd Ai-Analysis-Engine
sqlite3 verum.db < migrations/add_research_tables.sql
```

Verify with:
```bash
sqlite3 verum.db ".tables"  # Should show research_tasks and task_events
sqlite3 verum.db ".schema research_tasks"
```

---

### 8. Register Events

**File**: `src/analysis_engine/registry.py` (MODIFY - extend existing)

**Action**: Extend `RUN_CALLBACKS` dict to include research keys:

```python
# ADD THESE LINES (after line 16):

# research task callbacks - run_id -> task_id mapping
# Don't create new dict - reuse RUN_CALLBACKS with prefix
RUN_CALLBACKS["research_"] = None  # Name placeholder
```

**Connection**: Use same callback mechanism as analysis engine.

---

## Frontend Implementation (TypeScript - Lovable)

### 1. Create Mode Selection Component

**File**: `src/components/ResearchModeCard.tsx` (NEW)

**Action**: Create card component matching existing analysis page layout:

- Display 3 options: Data Analysis vs AI Research vs Documents
- Use existing `Card` component reference from `src/components/ui/card.tsx`
- Use existing icon set from lucide-react
- On click, navigate to separate view
- Copy styling from existing dashboard cards

---

### 2. Create Task List Component

**File**: `src/components/TaskList.tsx` (NEW)

**Action**: List all research tasks with status indicators:

- Fetch tasks from `/api/research/sessions`
- Display up to 10 tasks (pagination not needed for v1)
- Show objective, depth, status
- Navigation to task detail page
- Copy layout from existing task lists

---

### 3. Create Event Timeline Component

**File**: `src/components/ResearchStage.tsx` (NEW)

**Action**: Display live agent events:

- Fetch events from `/api/research/{task_id}`
- Group events by agent role
- Different border colors based on agent level (coordinator = amber, agent = blue, tool = slate)
- Collapse 3+ consecutive events from same agent (save screen space)
- Append synthesis at end (green box when complete)
- Copy pattern from existing analysis event display

---

### 4. Create Research Detail Page

**File**: `src/routes/research.tsx` (NEW - dynamic route)

**Action**: Task detail page for watching progress:

- Fetch task from `/api/research/{task_id}`
- Setup EventSource to `/api/research/{task_id}/stream`
- Same pattern as existing analysis page `/api/stream/{run_id}`
- Layout: header + sidebar (actions) + main (events)
- Copy structure from existing analysis page
- Use AppShell wrapper (already exists in root layout)

---

### 5. Update Homepage with Research Toggle

**File**: `src/routes/index.tsx` (MODIFY)

**Actions**:

```tsx
// Add two new cards inside the grid for "AI Research" and "Documents"
// Import and use ResearchModeCard component
// On click, navigate to /research route
// Keep existing "Data Analysis" card (don't change)
```

---

### 6. Create Update Dashboard Route

**File**: `src/routes/research/index.tsx` (NEW)

**Action**: Research hub page listing all tasks:

- Fetch list of tasks from API
- Render TaskList component on left side
- Render TaskDetail card on right (show single selected task)
- Clicking task updates detail view
- Copy pattern from existing `/history` or `/chat` page

---

## Integration Points (Lazy Summary)

**Reuse Existing System**:

1. **Database**: Extend `RunEvent` pattern → add `TaskEvent` and `ResearchTask`
2. **Events**: Reuse `RUN_CALLBACKS` registry (don't recreate)
3. **Streams**: Reuse `event_generator()` and EventSource pattern
4. **API Pattern**: Same endpoint structure as analysis (POST → stream → poll)
5. **LLM Client**: Reuse `get_llm()` and `explore_loop()` from existing
6. **UI Components**: All new components use shadcn/ui components already installed

**New Only**:

1. **Database tables**: `research_tasks`, `task_events` (2 tables)
2. **Backend files**: 3 new files (`src/research/state.py`, `src/research/graph.py`, `src/research/agents.py`)
3. **Backend nodes**: 4 directory files (`src/research/nodes/coordinator.py`, `src/research/nodes/web_agent.py`, `src/research/nodes/doc_agent.py`, `src/research/nodes/paper_agent.py`)
4. **API endpoints**: 4 new endpoints in `app.py`
5. **Frontend files**: 5 new components + 1 route

---

## Implementation Priority (Order Matters)

### Phase 1: Database Only
1. Add database tables (2 tables, 40 minutes of work)
2. Add models to models.py (20 minutes)
3. Run migration and verify (10 minutes)

### Phase 2: Backend Core  
4. Create `src/research/state.py` (30 minutes)
5. Create `src/research/agents.py` (20 minutes)
6. Create orchestrator graph (40 minutes)
7. Create synchronization node (similar to planner_node) (30 minutes)

### Phase 3: Frontend Core
8. ResearchModeCard component (30 minutes)
9. TaskList component (20 minutes)
10. ResearchStage component (30 minutes)
11. research.tsx route (40 minutes)
12. index.tsx update (10 minutes)

### Phase 4: Polish
13. Testing agents against real tasks (1 hour)
14. Error handling (solidity, bounded loops) (30 minutes)
15. Documentation (what each agent does) (30 minutes)

**Total time estimate**: ~8 hours (backend 4h + frontend 4h)

---

## What NOT to Change

**Don't touch these existing files**:

- `src/analysis_engine/graph.py` - Keep exactly as is
- `src/analysis_engine/nodes/*.py` - Don't modify existing nodes
- `src/analysis_engine/tools/sandbox/*.py` - Keep sandbox as is
- `src/analysis_engine/registry.py` - Don't delete RUN_CALLBACKS
- `src/analysis_engine/llm/client.py` - Keep LLM client
- `src/app.py` - Only ADD new endpoints, don't delete
- Existing frontend pages - Keep analysis/chat/history intact

**Why**: These are working. Only extend. Don't rewrite.

---

## Test Commands

After implementing:

```bash
# Test database
sqlite3 verum.db "SELECT COUNT(*) FROM research_tasks;"

# Test API
curl -X POST http://localhost:8000/api/research/start \
  -H "Content-Type: application/json" \
  -d '{"objective":"test", "task_type":"general", "depth":3}'

# Test stream
curl http://localhost:8000/api/research/<task_id>/stream
```

Start browser, navigate to `/research` route, should see new research task in list.

---

## Success Criteria

✅ Database: `add_research_tables.sql` runs without error

✅ Backend: 5 new API endpoints respond with correct data

✅ Frontend: Can create research task from homepage, see events stream

✅ Events: EventSource shows real-time progress same pattern as analysis stream

✅ Existing features: Data analysis still works (nothing broken)
