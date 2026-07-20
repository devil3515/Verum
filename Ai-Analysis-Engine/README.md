# AI Analysis Engine

A backend service for Verum's data analysis pipeline.

The engine ingests CSV files, runs a multi-stage AI analysis workflow, stores results in SQLite, and exposes live SSE streaming plus chat capabilities for follow-up questions.

## Project structure

- `src/app.py` — FastAPI application and HTTP/SSE endpoints
- `src/analysis_engine/graph.py` — pipeline graph builder and execution flow
- `src/analysis_engine/state.py` — shared pipeline data schema
- `src/analysis_engine/db/` — SQLAlchemy database engine, models, and persistence logic
- `src/analysis_engine/agent/` — chat agent orchestration
- `src/analysis_engine/llm/` — LLM configuration and client wrapper
- `src/analysis_engine/tools/` — tool implementations for data cleaning, analysis, verification, sandbox execution, and web search
- `charts/` — generated chart artifacts
- `uploads/` — CSV uploads used as pipeline inputs
- `verum.db` — default SQLite database file

## Requirements

Install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

The service loads environment variables via `python-dotenv`. Required variables:

- `LLM_BASE_URL` — base URL for the LLM API
- `LLM_API_KEY` — API key for the LLM provider
- `LLM_MODEL` — model name to use

Optional variables:

- `LLM_TEMPERATURE` — LLM temperature (default `0.1`)
- `LLM_MAX_TOKENS` — max tokens (default `8000`)
- `LLM_PROJECT` — OpenAI project header (default `default`)
- `DATABASE_URL` — database connection string (default `sqlite:///verum.db`)
- `VERUM_UPLOAD_DIR` — CSV upload directory (default `./uploads`)
- `VERUM_CHARTS_DIR` — chart output directory (default `./charts`)

Create a `.env` file in the repo root with keys like:

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key
LLM_MODEL=gpt-4o
```

## Run the service

From the project root:

```bash
cd /Users/Rahul Saini/Desktop/Analysis Agent/Ai-Analysis-Engine
source .venv/bin/activate
cd src
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The app will initialize the database automatically on startup.

## API overview

### Health

- `GET /health`
  - Returns service status.

### Run pipeline

- `POST /api/start`
  - `file`: CSV file upload
  - `question`: optional analysis prompt
  - Response: `{ "run_id": "..." }`

### Stream live run events

- `GET /api/stream/{run_id}`
  - Returns an SSE stream of pipeline events for an active run.

### Replay completed run events

- `GET /api/runs/{run_id}/events`
  - Returns a chronological event replay from the database.

### Run history

- `GET /api/runs`
  - Returns a list of past runs.

- `GET /api/runs/{run_id}/report`
  - Returns full run output, including claims, charts, and cleaning log.

### Chat support

- `POST /api/chat/start`
  - Body: `{ "run_id": "..." }`
  - Returns a chat session tied to a completed run.

- `POST /api/chat/message`
  - Body: `{ "session_id": "...", "message": "..." }`
  - Returns an SSE stream of chat events and the assistant answer.

- `GET /api/chat/{session_id}/history`
  - Returns the persisted chat conversation history.

## Notes

- Uploaded files are saved under `uploads/` and referenced by the pipeline.
- Charts are written to `charts/` and served via `GET /charts/{ref}`.
- The default SQLite database file is `verum.db`.

## Local development

For quick experiments, you can run the graph directly:

```bash
python src/analysis_engine/graph.py path/to/test.csv "Your question here"
```

This bypasses FastAPI and executes the pipeline locally.

## License

No license is specified in this repo.
