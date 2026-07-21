# Backend Implementation Phases - Agent Swarm

## Week 1: Core Infrastructure

### Phase 1.1: Database Setup (2 hours)

**Tasks**:
1. Open `Ai-Analysis-Engine/`
2. Create new directory: `migrations/`
3. Create file `migrations/add_research_tables.sql`
4. Add content to create two tables:
   - `research_tasks` - stores task metadata
   - `task_events` - stores agent event logs
5. Add foreign key constraints to `research_tasks.user_id` linking to existing `users.id`
6. Add foreign key constraint to `task_events.task_id` linking to `research_tasks.id`
7. Run migration command: `sqlite3 verum.db < migrations/add_research_tables.sql`
8. Verify migration success by running: `sqlite3 verum.db ".tables"` (should show research_tasks, task_events)
9. Test insertion of dummy record: `INSERT INTO research_tasks (task_id, user_id, objective, task_type, depth) VALUES (...)`
10. Check schema with: `sqlite3 verum.db ".schema research_tasks"`

**Success Criteria**:
✅ No SQL errors when running migration
✅ Both tables exist in database
✅ Foreign keys set up correctly (schema shows CONSTRAINT clauses)
✅ Can insert test data and query back

---

### Phase 1.2: State Model (3 hours)

**Tasks**:
1. Create new directory: `Ai-Analysis-Engine/src/research/`
2. Create file `src/research/state.py`
3. Copy pattern from `src/analysis_engine/state.py` (PipelineState)
4. Rename classes to research equivalents:
   - `PipelineState` → `ResearchTask`
   - `Claim` → `Finding` (tasks produce findings, not claims)
   - `WebSource` → `SourceEvidence` (literal URLs from web/search)
   - `CleaningLogEntry` → `TaskEventLog` (this stores per-agent events)
5. Add fields to ResearchTask:
   - `task_id` (UUID, primary key)
   - `user_id` (string, links to users table)
   - `objective` (text, user's question)
   - `task_type` (enum: general/web/document/paper/tool)
   - `depth` (int: number of agent layers, 1-5)
   - `status` (enum: queued/active/failed/completed)
   - `findings` (list of Finding objects)
   - `evidence` (list of SourceEvidence objects)
   - `synthesis` (string: final LLM summary)
6. Add ContextManager method (same as existing `get_session()`)
7. Update imports in `src/analysis_engine/state.py` to include research if needed

**Success Criteria**:
✅ State model imports without errors
✅ Pydantic validation works for suggested fields
✅ Can serialize and deserialize state objects
✅ All agents can read/write this state via LangGraph

---

### Phase 1.3: Agent Registry (2 hours)

**Tasks**:
1. Open file `Ai-Analysis-Engine/src/research/agents.py`
2. Create dictionary constant `AGENTS`
3. Define 5 agent entries with these keys:
   - `orchestrator` (depth=1): manager that creates task breakdown, routes to specialized agents
   - `research_web` (depth=2): searches web, reads URLs, verifies claims
   - `research_doc` (depth=2): reads documents, extracts summaries
   - `research_paper` (depth=3): analyzes academic papers, tracks trends
   - `research_tool` (depth=2): runs Python tools for calculations
4. Each agent needs: name, depth, role name, list of capabilities
5. Map capabilities to existing tool functions from `tools/`
6. Add `get_agent_for_task(task_type)` helper: return agent name for given objective
7. Add `get_max_depth(task_type)` helper: return max recursive layers needed

**Success Criteria**:
✅ agents.py imports without errors
✅ Defining list of capabilities matches existing tool names
✅ Can query registry for agent by task_type (e.g., "all" → list of all agents)
✅ Capabilities section in documentation reflects reality

---

### Phase 1.4: Orchestrator Graph Skeleton (3 hours)

**Tasks**:
1. Open file `Ai-Analysis-Engine/src/research/graph.py`
2. Create `build_research_orchestrator()` function (mirrors `build_graph()` from analysis)
3. Create StateGraph with ResearchTask as state argument
4. Add nodes:
   - `"orchestrator"` with `coordinator_node` (skeleton)
   - `"web"` with `web_research_node` (skeleton)
   - `"doc"` with `document_research_node` (skeleton)
   - `"paper"` with `paper_agent_node` (skeleton)
   - `"tool"` with `tool_operator_node` (skeleton)
   - `"synthesize"` with `synthesis_node` (skeleton)
5. Add entry point: `"orchestrator"`
6. Add conditional edge from orchestrator to specialized agents based on task_type field
7. Add conditional edges from specialized agents back to orchestrator if more research needed
8. Add final edge to `synthesize` node
9. Add final edge from synthesis to END
10. Return compiled graph with `graph.compile()`
11. Add `main()` test function (similar to analysis graph test)

**Success Criteria**:
✅ Graph compiles without errors
✅ Can create state object from ResearchTask model
✅ Graph.invoke() runs without TypeError
✅ No circular import errors between research module and existing analysis module

---

## Week 2: Agent Nodes Implementation

### Phase 2.1: Coordinator Node (3 hours)

**Tasks**:
1. Create directory: `Ai-Analysis-Engine/src/research/nodes/`
2. Create file `Ai-Analysis-Engine/src/research/nodes/coordinator.py`
3. Import dependencies from existing patterns:
   - `from analysis_engine.state import ResearchTask`
   - `from analysis_engine.llm.client import get_llm`
   - `from research.agents import AGENTS`
4. Create `coordinator_node(state, callback)` function
5. Implement task decomposition logic:
   - Read `state.objective` and `state.task_type`
   - Use LLM to break down into sub-tasks (like planner_node)
   - Decide which specialized agents to deploy
   - Update `state.plan` with list of agent names
   - Call `callback(event_type, data)` to log orchestrator decisions
6. Add routing logic based on task_type:
   - general → route to multiple agents
   - web → route only to web_research_agent
   - document → route only to document_agent
7. Create function `route_to_specialized_agents(state)` returns agent names
8. Add tool dispatcher for coordinator (optional if using explore_loop)
9. Return updated state dict (not new ResearchTask object)

**Success Criteria**:
✅ Node runs with valid ResearchTask state
✅ callback is called with event_type="orchestrator_decision"
✅ LLM returns structured plan (even if fake LLM)
✅ state.plan field updated with agent names
✅ No errors when trying to call LLM with empty objective

---

### Phase 2.2: Web Research Node (4 hours)

**Tasks**:
1. Create file `Ai-Analysis-Engine/src/research/nodes/web_agent.py`
2. Import dependencies:
   - `from analysis_engine.state import ResearchTask`
   - `from research.agents import AGENTS`
   - `from analysis_engine.llm.client import get_llm`
3. Create `web_research_node(state, callback)` function
4. Read `state.task_type` and constraints from agent config
5. Implement web search coordination:
   - For each sub-task in state.plan:
     - Call LLM to generate web search query
     - Use tavily API in `tools/web_search.py`
     - Log event_type="web_search" with callback
     - Read top N URLs from search results
     - For each URL:
       - Log event_type="read_url"
       - Call LLM to extract key points from URL content
6. Implement claim verification (reuse from verification_node):
   - Store findings in `state.findings`
   - Flag which sources verify each finding
7. Add iterative loop:
   - Continue until state.depth reaches 0 or plan complete
   - Recurse back to coordinator for more sub-tasks
8. Store findings in `state.findings` list
9. Call callback to log event_type="web_agent_complete" when done
10. Return updated state

**Success Criteria**:
✅ Node reads from state and writes to state
✅ Callback logs events for search and URL reads
✅ Findings stored in correct format
✅ Handles empty tavily API key gracefully (return error message)
✅ Tool dispatcher optional (can use explore_loop pattern for demo)

---

### Phase 2.3: Document Research Node (3 hours)

**Tasks**:
1. Create file `Ai-Analysis-Engine/src/research/nodes/doc_agent.py`
2. Import same dependencies as web_agent
3. Create `document_research_node(state, callback)` function
4. Implement document reading logic:
   - Service 2 options:
     - Option A: Read uploaded files (reuse from analysis)
       - Parse CSV/JSON from uploads/
       - Extract columns/fields as "documents"
     - Option B: Accept new file upload in this node
   - For each document:
     - Extract metadata (file name, size, type)
     - Use LLM to summarize content
     - Extract keywords/t tags
5. Store document metadata in state.evidence
6. Generate "key insights" from documents:
   - Use LLM to synthesize across all docs
   - Create findings list
7. Add callback events for document_read, summary_complete
8. Handle case where no documents available (log warning)
9. Return state updates

**Success Criteria**:
✅ Reads uploaded files (uploads/ directory)
✅ Callback logs document metadata
✅ Summaries generated via LLM
✅ Works even if uploads/ is empty (graceful failure)
✅ Findings stored in state

---

### Phase 2.4: Paper Research Node (3 hours)

**Tasks**:
1. Create file `Ai-Analysis-Engine/src/research/nodes/paper_agent.py`
2. Import dependencies
3. Create `paper_agent_node(state, callback)` function
4. Implement academic search logic:
   - Search arXiv/PubMed APIs (reuse from existing tools if any)
   - Or mock search results if no API key
   - For each paper:
     - Extract metadata (title, authors, citation count)
     - Use LLM to summarize paper
     - Identify trends/citations
5. Store evidence in state.evidence (research paper findings)
6. Generate citations network:
   - Build mapping of {paper_id → [cited_by_ids]}
   - Store in state.evidence network attribute
7. Add callback events for paper_search, paper_summary
8. Handle missing API keys (return error, mark status="failed")
9. Return state with findings

**Success Criteria**:
✅ Searches academic sources (mock or real)
✅ Callback logs paper metadata
✅ Citations network built correctly
✅ Graceful failure if no API key
✅ Findings stored in state

---

### Phase 2.5: Tool Research Node (3 hours)

**Tasks**:
1. Create file `Ai-Analysis-Engine/src/research/nodes/tool_agent.py`
2. Import dependencies (reuse sandbox tools)
3. Create `tool_agent_node(state, callback)` function
4. Implement Python tool execution:
   - Add new tool to `analysis_tools.py` or create `tools/research_tools.py`:
     - `run_python_research`: Execute research code
     - `compute_statistic`: Run custom calculations
5. Read state.findings to identify which computations needed
6. Generate code for each computation:
   - Use LLM to write Python code based on findings
   - Validate code using sandbox (reuse `sandbox/executor.py`)
7. Execute code in sandbox (timeout 30s, memory 512MB limit)
8. Capture results in state.findings
9. Log tool events for code execution
10. Return updated state with improved findings

**Success Criteria**:
✅ Sandboxed code executes without crashing system
✅ Tool limits enforced (no infinite loops)
✅ Error messages captured when code fails
✅ Findings updated with computed results
✅ Callback logs tool execution events

---

### Phase 2.6: Synthesis Node (3 hours)

**Tasks**:
1. Create file `Ai-Analysis-Engine/src/research/nodes/synthesis.py` (NEW)
2. Import dependencies
3. Create `synthesis_node(state, callback)` function
4. Read all findings from state:
   - `state.findings` list
   - `state.evidence` list
   - `state.synthesis` (if already built)
5. Build synthesis prompt:
   - Include objective, findings, evidence, evidence on limitations
   - Ask LLM to produce 2-3 sentence conclusion
6. Call LLM to synthesize final answer:
   - Use `call_structured()` with AnswerOutput schema
7. Parse synthesis output:
   - Extract `final_answer` string
   - Extract `follow_up_questions` list
8. Update state.synthesis field
9. Call callback to log event_type="synthesis_complete"
10. Mark task completed (status="completed")

**Success Criteria**:
✅ Integrates all findings into synthesis
✅ LLM returns structured synthesis
✅ Callback logs completion event
✅ Task status updated to "completed"
✅ Follow-up questions suggested (optional)

---

## Week 3: API Integration

### Phase 3.1: API Endpoints Implementation (4 hours)

**Tasks**:
1. Open file `Ai-Analysis-Engine/src/app.py`
2. Add these imports (bottom of file or existing imports):
   ```python
   from research.state import ResearchTask
   from research.agents import AGENTS
   from research.graph import build_research_orchestrator
   from research.nodes.coordinator import coordinator_node
   from research.nodes.web_agent import web_research_node
   from research.nodes.doc_agent import document_research_node
   from research.nodes.paper_agent import paper_agent_node
   from research.nodes.tool_agent import tool_agent_node
   from research.nodes.synthesis import synthesis_node
   ```
3. Add dictionary registries at top of file (after line 50, before app definition):
   ```python
   # research task registries (reuse RUN_CALLBACKS pattern)
   _RESEARCH_QUEUES: dict[str, asyncio.Queue] = {}
   _RESEARCH_TASKS: dict[str, ResearchTask] = {}
   ```
4. Add endpoint 1: `POST /api/research/start`:
   - Generate UUID as task_id
   - Create ResearchTask object
   - Insert into research_tasks table using `get_db_session()`
   - Create asyncio.Queue
   - Register callback in RUN_CALLBACKS (reuse existing pattern)
   - Start background thread with `run_research_orchestrator(task_id, callback)`
   - Return `{task_id: str}`
5. Add endpoint 2: `GET /api/research/sessions`:
   - Query research_tasks table
   - Return list of `{task_id, objective, status, created_at}`
   - Reuse existing pagenation pattern if needed
6. Add endpoint 3: `GET /api/research/{task_id}`:
   - Query research_tasks where task_id = XXX
   - Query task_events where task_id = XXX ordered by created_at
   - Return `{task: {...}, events: [...]}`
7. Add endpoint 4: `GET /api/research/{task_id}/stream`:
   - Get queue from _RESEARCH_QUEUES
   - Create event_generator() similar to `/api/stream/{run_id}`
   - Return StreamingResponse with SSE format
8. Add exception handling to all endpoints:
   - Always close database session
   - Return HTTPException on errors

**Success Criteria**:
✅ All 4 endpoints exist without breaking existing routes
✅ POST route creates record in research_tasks table
✅ GET sessions returns list from DB
✅ GET detail returns task + events from DB
✅ SSE stream emits events when agents run

---

### Phase 3.2: Registry Integration (2 hours)

**Tasks**:
1. Open file `Ai-Analysis-Engine/src/analysis_engine/registry.py`
2. Add comment explaining extension line 16:
   ```python
   # Register research task callbacks with prefix
   # Research tasks will use keys like "research_<task_id>"
   ```
3. Add note about extending callback pattern:
   - Research callbacks use same signature as analysis callbacks
   - Events logged to task_events table via callback
   - Don't change RUN_CALLBACKS dict structure

**Success Criteria**:
✅ Registry file unchanged
✅ Only documentation added explaining extension
✅ No code changes needed in registry.py

---

### Phase 3.3: Error Handling (3 hours)

**Tasks**:
1. Add exception handling to coordinator_node:
   - Catch LLM errors/k timeouts
   - Mark state.status = "failed"
   - Log error to callback
   - Return early with error message
2. Add exception handling to specialized nodes:
   - Catch tool execution timeouts
   - Catch database write errors
   - Log to callback
   - Propagate to next node or synthesis
3. Add exception handling to synthesis_node:
   - Catch LLM generation failures
   - Mark status = "failed"
   - Log error
4. Add thread safety in agent execution wrapper:
   ```python
   def run_research_orchestrator(task_id, callback):
       try:
           compiled = build_research_orchestrator()
           initial = ResearchTask(...)  # load from DB
           result = compiled.invoke(initial)
           # save to DB
       except Exception as e:
           log error
           callback("run_failed", {"error": str(e)})
       finally:
           callback("__done__", {})
   ```
5. Add proper cleanup in finally blocks (ensure queue cleared)

**Success Criteria**:
✅ Errors caught and logged via callback
✅ Task status marked "failed" on exception
✅ Thread exits gracefully (no orphaned threads)
✅ Database changes rolled back on error (using get_db_session())

---

### Phase 3.4: Testing (2 hours)

**Tasks**:
1. Test Phase 1 (database):
   - Run migration script
   - Verify both tables exist
   - Insert test record manually
   - Query back to confirm
2. Test Phase 2 (node execution):
   - Run `python src/research/graph.py` (integration test)
   - Start mock server: `uvicorn app:app`
   - POST /api/research/start with dummy objective
   - Check research_tasks table for new record
3. Test Phase 3 (API endpoints):
   - curl POST /api/research/start → get task_id
   - curl GET /api/research/sessions → verify task appears
   - curl GET /api/research/<task_id> → verify structure
   - Browser: open SSE URL → watch events stream → should see aggregations
4. Verify existing features still work (analysis, chat)
5. Reset database for clean slate

**Success Criteria**:
✅ Migration runs cleanly
✅ All 4 endpoints return correct data
✅ SSE stream shows real-time events matching agent activity
✅ Existing analysis pipeline still functions
✅ No crashes in log file

---

## Week 4: Documentation & Polish

### Phase 4.1: README Updates (2 hours)

**Tasks**:
1. Open `Ai-Analysis-Engine/README.md`
2. Add new section after existing "Projects Overview":
   - Title: "Research Mode"
   - Description: "Multi-agent research engine for general-purpose questions"
   - Explain architecture (coordinator + specialized agents)
   - List endpoints: `/api/research/start`, `/api/research/sessions`, etc.
   - Add architecture diagram (text-based from plan)
3. Update build documentation:
   - Add `python src/research/graph.py` to CLI examples
4. Document agent roles:
   - Explain what each agent does
   - Link to agent definitions in src/research/agents.py

**Success Criteria**:
✅ README shows research section clearly
✅ Links all mentioned endpoints
✅ Documentation matches actual implementation
✅ Architecture diagram reflects real graph structure

---

### Phase 4.2: Code Comments (1 hour)

**Tasks**:
1. Add comments to `src/research/graph.py`:
   - Explain entry point and branching logic
   - Document why agents recurse to coordinator
   - Add "Ponytail note" about recursive depth limits
2. Add comments to `src/research/nodes/coordinator.py`:
   - Explain task decomposition strategy
   - Add docstring to routing function
3. Add comments to `src/research/nodes/synthesis.py`:
   - Explain synthesis prompt building
   - Mark LLM call as one-shot (not multi-turn)
4. Add comments to `src/app.py` new endpoints:
   - Explain queue pattern
   - Explain callback registration

**Success Criteria**:
✅ All changes match documentation in README
✅ Code comments explain "why", not "what"
✅ Ponytail ceiling notes added where appropriate
✅ Code self-documenting

---

### Phase 4.3: Performance Review (2 hours)

**Tasks**:
1. Check database query performance:
   ```sql
   EXPLAIN QUERY PLAN
   SELECT * FROM research_tasks WHERE user_id = 'test';
   ```
2. Check table index utilization:
   ```sql
   PRAGMA index_list(research_tasks)
   ```
3. Identify slow queries during integration tests
4. Add indexes if needed (task_id, user_id)
5. Test concurrent runs:
   - Start 5 research tasks simultaneously
   - Watch for database locks
   - Verify all finish without timeout
6. Review memory usage during long runs (10+ find paper iterations)
7. Document performance ceilings:
   - SQLite concurrency limit (1 writer at a time)
   - Event log size limit (suggest cleanup)
   - Max recursive depth (5 layers)

**Success Criteria**:
✅ Indexes created for task_id and user_id
✅ No blocking during concurrent operations
✅ Event log grows predictably
✅ Performance constraints documented

---

## Summary Timeline

**Phase 1** - Week 1 (all 4 sub-phases): Core infrastructure, database, models, graph skeleton

**Phase 2** - Week 2 (all 6 sub-phases): Agent node implementations (coordinator, web, doc, paper, tool, synthesis)

**Phase 3** - Week 3 (all 4 sub-phases): API integration, registry updates, error handling, testing

**Phase 4** - Week 4 (all 3 sub-phases): Documentation, code comments, performance review

**Total time estimate**: 24 hours of systematic implementation (demos/integration testing included)

**Order**: Follow phases sequentially. Don't move to Phase 2.3 before Phase 2.2 complete.
