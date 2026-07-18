# Architecture

This document explains the main runtime components of the chatbot service and how they interact.

## Runtime layers

### 1. API layer

The FastAPI application in [src/app.py](src/app.py) wires together the routers and middleware. It serves:

- the chat API at /api/v1/chat
- the streaming chat API at /api/v1/chat/stream
- the health endpoint at /api/v1/health
- a simple frontend entry point at /

### 2. Agent workflow layer

The LangGraph workflow in [src/agents/graph.py](src/agents/graph.py) composes three logical steps:

1. Load memories from the store
2. Invoke the LLM to generate a response
3. Optionally extract new memories or trim conversation history in the background

The actual node implementations live in [src/agents/nodes.py](src/agents/nodes.py).

### 3. Persistence layer

Conversation history and long-term memories are stored through helpers in [src/services/memory.py](src/services/memory.py):

- chat_history stores the active conversation history per user and conversation
- memories stores long-term facts extracted from user interactions
- conversations tracks conversation titles and last-updated timestamps

By default, the service uses SQLite. If DATABASE_URL is configured, the LangGraph store can switch to Postgres.

### 4. Presentation layer

The repository includes:

- [streamlit_app.py](streamlit_app.py) for a Streamlit-based interactive demo
- [frontend/index.html](frontend/index.html) and related assets for a lightweight static UI

## Request lifecycle

1. A client sends a chat request to the FastAPI router.
2. The router validates the request, resolves the API key, and loads prior history from SQLite.
3. The LangGraph workflow runs with the current message set and any retrieved memory.
4. The assistant response is saved back to chat history.
5. A background memory workflow extracts new facts and trims older history when necessary.

## Design choices

- Short-term memory is represented by recent messages in the conversation timeline.
- Long-term memories are distilled facts that persist across conversations.
- The architecture separates state management from the LLM call so the workflow remains easier to extend with tools, evaluation, or additional memory strategies.
