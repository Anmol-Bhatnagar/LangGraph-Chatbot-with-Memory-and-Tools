# My Agent Service

Production-ready LangGraph Chatbot Service featuring dynamic Dual Memory (Short-Term Trimming & SQLite-persisted Long-Term Memories) exposed via a restful FastAPI backend and a Streamlit dashboard.

## Folder Structure
```
my-agent-service/
├── .github/workflows/        # CI/CD pipelines
├── docker/                   # Dockerfiles for dev and production
├── src/                      # Core application source
│   ├── config/               # Settings & logging configurations
│   ├── agents/               # Core LLM/LangGraph graph, nodes, and state definitions
│   ├── api/                  # FastAPI routers, dependencies, and schemas
│   ├── services/             # SQLite DB query layer for memory persistence
│   ├── app.py                # FastAPI lifecycle & middleware definitions
│   └── main.py               # ASGI Entry point
├── tests/                    # Unit and Integration test layout
├── streamlit_app.py          # Frontend dashboard
└── pyproject.toml            # Project definition & metadata
```

## Running the Services

### Setup Environment
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### Start FastAPI Backend
```bash
..\.venv\bin\python.exe -m uvicorn src.app:app --reload --port 8000
```
Swagger UI docs will be available at `http://localhost:8000/docs`.

### Start Streamlit Frontend
```bash
..\.venv\bin\python.exe -m streamlit run streamlit_app.py
```
Streamlit UI dashboard will be available at `http://localhost:8501`.
