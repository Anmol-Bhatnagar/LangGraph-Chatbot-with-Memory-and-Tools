import os
import uuid
import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends, status
from langchain_core.messages import HumanMessage, AIMessage
from src.config.settings import settings

from src.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    MessageSchema,
    MemoryRequest,
    MemorySchema,
    GenericResponse
)
from src.agents.graph import chatbot_app
from src.services.memory import (
    load_chat_history,
    save_chat_history,
    clear_chat_history,
    get_memories,
    save_memory,
    delete_memory,
    clear_all_memories
)
from src.api.dependencies.auth import verify_api_token

logger = logging.getLogger("ChatRouter")

router = APIRouter(prefix="/api/v1")

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest, token: str = Depends(verify_api_token)):
    """Submit a user message to the chatbot.
    
    Loads history from SQLite, runs the LangGraph compiled graph (updating memory and trimming if needed),
    saves the updated history, and returns the response with current memory states.
    """
    user_id = request.user_id
    api_key = request.api_key.strip() if request.api_key else ""
    
    if not api_key:
        if request.provider == "google":
            api_key = settings.GEMINI_API_KEY or ""
        elif request.provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY") or ""
            
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"API key for '{request.provider}' must be provided or configured on the server."
        )
        
    try:
        # 1. Load existing short-term messages from SQLite
        messages = load_chat_history(user_id)
        
        # 2. Append the new Human Message
        msg_id = str(uuid.uuid4())
        human_msg = HumanMessage(content=request.message, id=msg_id)
        messages.append(human_msg)
        
        # 3. Setup LangGraph run configurations
        config = {
            "configurable": {
                "user_id": user_id,
                "provider": request.provider,
                "model": request.model,
                "api_key": api_key,
                "limit": request.limit
            }
        }
        
        # 4. Invoke LangGraph chatbot compiled graph
        result = chatbot_app.invoke(
            {
                "messages": messages,
                "user_id": user_id
            },
            config=config
        )
        
        # 5. Extract updated messages (after chatbot response and any trimming)
        updated_messages = result.get("messages", [])
        
        # 6. Save updated short-term message log to SQLite
        save_chat_history(user_id, updated_messages)
        
        # 7. Extract the latest AI Message as the API reply
        ai_reply = ""
        for msg in reversed(updated_messages):
            if isinstance(msg, AIMessage):
                ai_reply = msg.content
                break
                
        # Format active messages list for client response
        active_messages_response = []
        for msg in updated_messages:
            if isinstance(msg, HumanMessage):
                active_messages_response.append(MessageSchema(role="user", content=msg.content, id=getattr(msg, "id", None)))
            elif isinstance(msg, AIMessage):
                active_messages_response.append(MessageSchema(role="assistant", content=msg.content, id=getattr(msg, "id", None)))
                
        return ChatResponse(
            response=ai_reply,
            active_messages=active_messages_response,
            long_term_memories=result.get("long_term_memories", [])
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the chat session: {str(e)}"
        )

@router.get("/memories/{user_id}", response_model=List[MemorySchema], tags=["Memories"])
def get_user_memories(user_id: str):
    """Retrieve all stored long-term memories for a user."""
    return get_memories(user_id)

@router.post("/memories/{user_id}", response_model=GenericResponse, tags=["Memories"])
def create_user_memory(user_id: str, request: MemoryRequest):
    """Manually add a memory fact for a user."""
    memory_id = save_memory(user_id, request.content)
    if memory_id > 0:
        return GenericResponse(
            success=True,
            message=f"Memory successfully stored with ID: {memory_id}"
        )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to save memory to database."
    )

@router.delete("/memories/{user_id}/{memory_id}", response_model=GenericResponse, tags=["Memories"])
def delete_user_memory(user_id: str, memory_id: int):
    """Delete a specific memory by its ID for a user."""
    success = delete_memory(user_id, memory_id)
    if success:
        return GenericResponse(
            success=True,
            message=f"Memory {memory_id} successfully deleted."
        )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Memory {memory_id} not found or could not be deleted."
    )

@router.delete("/memories/{user_id}", response_model=GenericResponse, tags=["Memories"])
def clear_user_memories(user_id: str):
    """Clear all long-term memories stored in SQLite for a user."""
    success = clear_all_memories(user_id)
    if success:
        return GenericResponse(
            success=True,
            message=f"All long-term memories for user '{user_id}' cleared successfully."
        )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to clear long-term memories."
    )

@router.post("/chat/clear/{user_id}", response_model=GenericResponse, tags=["Chat"])
def clear_user_chat_history(user_id: str):
    """Clear the active conversation message history (short-term memory) for a user."""
    success = clear_chat_history(user_id)
    if success:
        return GenericResponse(
            success=True,
            message=f"Short-term chat history for user '{user_id}' cleared successfully."
        )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to clear short-term chat history."
    )

@router.get("/chat/{user_id}", response_model=List[MessageSchema], tags=["Chat"])
def get_user_chat_history(user_id: str):
    """Retrieve the active conversation message history (short-term memory) for a user."""
    try:
        messages = load_chat_history(user_id)
        active_messages_response = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                active_messages_response.append(MessageSchema(role="user", content=msg.content, id=getattr(msg, "id", None)))
            elif isinstance(msg, AIMessage):
                active_messages_response.append(MessageSchema(role="assistant", content=msg.content, id=getattr(msg, "id", None)))
        return active_messages_response
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while loading chat history: {str(e)}"
        )

@router.get("/config", tags=["Config"])
def get_config_status():
    """Retrieve the API key availability for each provider on the server."""
    return {
        "google": bool(settings.GEMINI_API_KEY),
        "groq": bool(os.environ.get("GROQ_API_KEY"))
    }
