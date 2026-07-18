# Deployment

This project is prepared for local development and container or platform deployment.

## Local development

Run the API locally:

```bash
python -m uvicorn src.app:app --reload --port 8000
```

Run the Streamlit UI in a separate terminal:

```bash
streamlit run streamlit_app.py
```

## Docker

The repository includes two container build targets:

- [docker/Dockerfile.dev](docker/Dockerfile.dev)
- [docker/Dockerfile.prod](docker/Dockerfile.prod)

Example build:

```bash
docker build -f docker/Dockerfile.prod -t my-agent-service .
```

## Render

The [render.yaml](render.yaml) file configures a web service for Render. It installs dependencies and starts the FastAPI app with:

```bash
uvicorn src.app:app --host 0.0.0.0 --port $PORT
```

Set the following environment variables in your hosting provider:

- GEMINI_API_KEY or OPENAI_API_KEY or GROQ_API_KEY
- APP_ENV
- APP_PORT
- DB_PATH
- DATABASE_URL (optional for Postgres-backed store)

## Production notes

- Use a managed database for production workloads if the service will handle concurrent traffic.
- Consider reverse proxying behind an API gateway for authentication and rate limiting.
- Keep secrets in a managed secret store rather than plaintext environment files.
