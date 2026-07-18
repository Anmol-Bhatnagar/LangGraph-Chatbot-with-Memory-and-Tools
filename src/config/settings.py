import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environmental variables from .env if present
load_dotenv()

class Settings(BaseModel):
    APP_NAME: str = Field(default="LangGraph Chatbot Memory Service")
    APP_ENV: str = Field(default="development")
    APP_PORT: int = Field(default=8000)
    DB_PATH: str = Field(default="memories.db")
    DB_TIMEOUT: float = Field(default=30.0)
    DB_WAL_MODE: bool = Field(default=True)
    
    # API Keys (can also be loaded dynamically in request)
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    
    # Langchain Tracing config
    LANGCHAIN_TRACING_V2: bool = Field(default=False)
    LANGCHAIN_API_KEY: Optional[str] = Field(default=None)
    LANGCHAIN_PROJECT: str = Field(default="my-agent-service")

# Instantiate settings singleton
settings = Settings(
    APP_NAME=os.getenv("APP_NAME", "LangGraph Chatbot Memory Service"),
    APP_ENV=os.getenv("APP_ENV", "development"),
    APP_PORT=int(os.getenv("APP_PORT", "8000")),
    DB_PATH=os.getenv("DB_PATH", "memories.db"),
    DB_TIMEOUT=float(os.getenv("DB_TIMEOUT", "30.0")),
    DB_WAL_MODE=os.getenv("DB_WAL_MODE", "true").lower() in ("true", "1", "yes"),
    GEMINI_API_KEY=os.getenv("GEMINI_API_KEY"),
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
    LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes"),
    LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY"),
    LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT", "my-agent-service")
)
