from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.logging_cfg import setup_logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.routers.v1_chat import router as chat_router
from src.api.routers.health import router as health_router

# Setup logging
setup_logging()

app = FastAPI(
    title="LangGraph Chatbot with Long-Term Memory",
    description="Production RESTful API backend for LangGraph Chatbot with Short-Term and Long-Term Memory",
    version="1.0.0"
)

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Register endpoints routers
app.include_router(chat_router)
app.include_router(health_router)

@app.get("/")
def read_root():
    """Serve the static HTML frontend dashboard."""
    return FileResponse("frontend/index.html")
