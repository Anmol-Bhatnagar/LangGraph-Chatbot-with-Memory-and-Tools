# LangGraph Chatbot Service

A production-oriented conversational AI service built with FastAPI, LangGraph, and Streamlit. The application combines short-term conversation history with long-term memory extraction so it can respond more contextually over time while remaining easy to deploy and operate.

## What this project does

- Exposes a REST API and streaming chat endpoint for conversational AI workloads
- Uses LangGraph to orchestrate memory loading, chat generation, memory extraction, and history trimming
- Persists conversations and memories locally with SQLite, with optional Postgres support through the LangGraph store integration
- Provides a lightweight Streamlit UI and a static frontend for local testing and demos
- Supports multiple LLM providers, including Google Gemini, OpenAI, and Groq

## Architecture overview

The service is organized into four main layers:

- API layer: FastAPI routers and request/response schemas in [src/api](src/api)
- Agent workflow: LangGraph nodes and state definitions in [src/agents](src/agents)
- Persistence layer: SQLite-backed conversation/memory services in [src/services](src/services)
- Presentation layer: Streamlit and static frontend assets in [streamlit_app.py](streamlit_app.py) and [frontend](frontend)

For a deeper walkthrough, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project structure

```text
my-agent-service/
├── docker/                  # Development and production container images
├── docs/                    # Architecture and deployment documentation
├── frontend/                # Static frontend assets served by FastAPI
├── src/
│   ├── agents/              # LangGraph nodes, graph definitions, prompts, and state
│   ├── api/                 # FastAPI routers, schemas, and auth dependencies
│   ├── config/              # Settings and logging configuration
│   └── services/            # Memory and conversation persistence helpers
├── tests/                   # Unit and integration tests
├── .env.example             # Environment variable template
├── pyproject.toml           # Project metadata and dependencies
├── render.yaml              # Render deployment configuration
├── requirements.txt         # Runtime dependencies
├── streamlit_app.py         # Streamlit dashboard entrypoint
└── src/app.py               # FastAPI application entrypoint
```

## Prerequisites

- Python 3.10 or newer
- A virtual environment tool such as venv or conda
- At least one LLM provider API key:
  - GEMINI_API_KEY for Google Gemini
  - OPENAI_API_KEY for OpenAI
  - GROQ_API_KEY for Groq fallback or primary usage

## Quick start

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Copy the environment template and configure your secrets.

```bash
cp .env.example .env
```

Update the values in [.env.example](.env.example) or the generated [.env](.env) file with your provider keys and desired settings.

4. Run the API server.

```bash
python -m uvicorn src.app:app --reload --port 8000
```

The FastAPI app will be available at:

- API docs: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/api/v1/health
- Frontend entry: http://localhost:8000/

5. Run the Streamlit dashboard in a separate terminal.

```bash
streamlit run streamlit_app.py
```

The UI will be available at http://localhost:8501.

## Configuration

The application reads configuration from environment variables. The most important ones are:

| Variable | Purpose |
| --- | --- |
| GEMINI_API_KEY | API key used for Google Gemini models |
| OPENAI_API_KEY | API key used for OpenAI models |
| GROQ_API_KEY | Optional fallback or primary provider key for Groq |
| APP_ENV | Runtime environment, typically development or production |
| APP_PORT | Port used by the FastAPI app |
| DB_PATH | SQLite database file for conversations and memories |
| DATABASE_URL | Optional Postgres connection string for the LangGraph store |
| LANGCHAIN_TRACING_V2 | Enables LangChain tracing when set to true |
| LANGCHAIN_API_KEY | Optional LangChain tracing API key |

## API usage

### Chat endpoint

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me" \
  -d '{
    "user_id": "demo-user",
    "provider": "google",
    "model": "gemini-2.5-flash",
    "message": "Hello, I like concise answers."
  }'
```

### Streaming chat endpoint

```bash
curl -N http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me" \
  -d '{
    "user_id": "demo-user",
    "provider": "google",
    "model": "gemini-2.5-flash",
    "message": "Stream this response to me."
  }'
```

## Testing

Run the test suite with:

```bash
pytest
```

## Deployment

The repository includes deployment assets for cloud hosting:

- [render.yaml](render.yaml) for Render web service deployment
- [docker/Dockerfile.dev](docker/Dockerfile.dev) and [docker/Dockerfile.prod](docker/Dockerfile.prod) for container-based deployment

For detailed deployment guidance, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Production considerations

For production deployments, consider the following:

- Replace the permissive CORS configuration with explicit allowed origins
- Move API keys and secrets into a secure secret store rather than relying on local environment files
- Use Postgres instead of SQLite for higher availability and concurrency if the workload grows
- Enable structured logging and monitoring around the API and LangGraph workflow
- Protect the API with authentication and rate limiting in front of the service

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
