from typing import List, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    user_id: str = Field("default_user", description="Unique identifier for the user profile.")
    conversation_id: Optional[str] = Field(default=None, description="Unique identifier for the conversation session.")
    message: str = Field(..., description="Message string sent by the user.")
    provider: str = Field("google", description="LLM provider: 'google' (Gemini) or 'groq'.")
    model: str = Field("gemini-2.0-flash", description="Model name to run in the backend.")
    api_key: Optional[str] = Field(default=None, description="API key for the selected provider. If not provided, fallback to server environment variables.")
    limit: int = Field(6, description="Short-term message limit before trimming.")

class MessageSchema(BaseModel):
    role: str
    content: str
    id: Optional[str] = None

class ConversationSchema(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's reply to the message.")
    active_messages: List[MessageSchema] = Field(..., description="List of messages currently active in short-term memory.")
    long_term_memories: List[str] = Field(..., description="List of long-term memories retrieved for this user.")

class MemoryRequest(BaseModel):
    content: str = Field(..., description="The user fact or preference to save manually.")

class MemorySchema(BaseModel):
    id: int
    content: str
    created_at: str

class GenericResponse(BaseModel):
    success: bool
    message: str
