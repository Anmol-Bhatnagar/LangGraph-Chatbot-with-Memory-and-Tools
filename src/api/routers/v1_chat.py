import os
import json
import uuid
import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from src.config.settings import settings

from src.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    MessageSchema,
    MemoryRequest,
    MemorySchema,
    GenericResponse,
    ConversationSchema
)
from src.agents.graph import chatbot_app, memory_app
from src.services.memory import (
    load_chat_history,
    save_chat_history,
    clear_chat_history,
    get_memories,
    save_memory,
    delete_memory,
    clear_all_memories,
    get_conversations,
    create_conversation,
    delete_conversation,
    touch_conversation
)
from src.api.dependencies.auth import verify_api_token

logger = logging.getLogger("ChatRouter")

router = APIRouter(prefix="/api/v1")

def run_background_memory_tasks(user_id: str, messages: list, config: dict, conversation_id: str = "default_conversation"):
    """Run memory extraction and history trimming in the background, updating SQLite."""
    try:
        logger.info(f"Starting background memory extraction/trimming for user '{user_id}' in conv '{conversation_id}'...")
        result = memory_app.invoke(
            {
                "messages": messages,
                "user_id": user_id
            },
            config=config
        )
        # Save pruned history to SQLite if anything was changed/removed
        updated_messages = result.get("messages", [])
        save_chat_history(user_id, updated_messages, conversation_id=conversation_id)
        logger.info(f"Completed background memory extraction/trimming for user '{user_id}' in conv '{conversation_id}'.")
    except Exception as e:
        logger.error(f"Error in background memory tasks for user '{user_id}': {e}")

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest, background_tasks: BackgroundTasks, token: str = Depends(verify_api_token)):
    """Submit a user message to the chatbot.
    
    Loads history from SQLite, runs the LangGraph compiled graph (updating memory and trimming if needed),
    saves the updated history, and returns the response with current memory states.
    """
    user_id = request.user_id
    conversation_id = request.conversation_id or "default_conversation"
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
        # Auto-name conversation title if it is a default/generic title
        convs = get_conversations(user_id)
        current_title = "New Conversation"
        for c in convs:
            if c["id"] == conversation_id:
                current_title = c["title"]
                break
        
        if current_title in ("New Conversation", "Default Conversation"):
            new_title = request.message[:30] + "..." if len(request.message) > 30 else request.message
            touch_conversation(user_id, conversation_id, title=new_title)

        # 1. Load existing short-term messages from SQLite
        messages = load_chat_history(user_id, conversation_id=conversation_id)
        
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
        save_chat_history(user_id, updated_messages, conversation_id=conversation_id)
        
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
                
        # 8. Queue background memory extraction and trimming
        background_tasks.add_task(
            run_background_memory_tasks,
            user_id=user_id,
            messages=updated_messages,
            config=config,
            conversation_id=conversation_id
        )
        
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

@router.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest, background_tasks: BackgroundTasks, token: str = Depends(verify_api_token)):
    """Submit a user message and stream the chatbot reply token-by-token.
    
    Uses Server-Sent Events (SSE) to send chunks to the client in real-time,
    saves the final history, and yields the final conversation metadata on completion.
    """
    user_id = request.user_id
    conversation_id = request.conversation_id or "default_conversation"
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
        
    async def event_generator():
        try:
            # Auto-name conversation title if it is a default/generic title
            convs = get_conversations(user_id)
            current_title = "New Conversation"
            for c in convs:
                if c["id"] == conversation_id:
                    current_title = c["title"]
                    break
            
            if current_title in ("New Conversation", "Default Conversation"):
                new_title = request.message[:30] + "..." if len(request.message) > 30 else request.message
                touch_conversation(user_id, conversation_id, title=new_title)

            # 1. Load history
            messages = load_chat_history(user_id, conversation_id=conversation_id)
            
            # 2. Append new message
            msg_id = str(uuid.uuid4())
            human_msg = HumanMessage(content=request.message, id=msg_id)
            messages.append(human_msg)
            
            # 3. Config
            config = {
                "configurable": {
                    "user_id": user_id,
                    "provider": request.provider,
                    "model": request.model,
                    "api_key": api_key,
                    "limit": request.limit
                }
            }
            
            # 4. Stream events from LangGraph
            final_output = None
            async for event in chatbot_app.astream_events(
                {"messages": messages, "user_id": user_id},
                config=config,
                version="v2"
            ):
                event_type = event["event"]
                name = event["name"]
                
                # Check for token chunks from the chatbot node
                if event_type == "on_chat_model_stream" and event.get("metadata", {}).get("langgraph_node") == "chatbot":
                    chunk_content = event["data"]["chunk"].content
                    if chunk_content:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk_content})}\n\n"
                        
                # Check for root chain end to grab the final state
                elif event_type == "on_chain_end" and name == "LangGraph":
                    final_output = event["data"]["output"]
            
            # 5. Save updated state and output metadata if execution succeeded
            if final_output:
                updated_messages = final_output.get("messages", [])
                save_chat_history(user_id, updated_messages, conversation_id=conversation_id)
                
                ai_reply = ""
                for msg in reversed(updated_messages):
                    if isinstance(msg, AIMessage):
                        ai_reply = msg.content
                        break
                        
                active_messages_response = []
                for msg in updated_messages:
                    if isinstance(msg, HumanMessage):
                        active_messages_response.append({
                            "role": "user",
                            "content": msg.content,
                            "id": getattr(msg, "id", None)
                        })
                    elif isinstance(msg, AIMessage):
                        active_messages_response.append({
                            "role": "assistant",
                            "content": msg.content,
                            "id": getattr(msg, "id", None)
                        })
                
                # 6. Queue background memory extraction and trimming
                background_tasks.add_task(
                    run_background_memory_tasks,
                    user_id=user_id,
                    messages=updated_messages,
                    config=config,
                    conversation_id=conversation_id
                )
                        
                yield f"data: {json.dumps({
                    'type': 'done',
                    'response': ai_reply,
                    'active_messages': active_messages_response,
                    'long_term_memories': final_output.get('long_term_memories', [])
                })}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'content': 'LangGraph execution did not produce final state.'})}\n\n"
                
        except Exception as e:
            logger.error(f"Error in chat_stream event generator: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/memories/{user_id}", response_model=List[MemorySchema], tags=["Memories"])
def get_user_memories(user_id: str):
    """Retrieve all stored long-term memories for a user."""
    return get_memories(user_id)

@router.post("/memories/{user_id}", response_model=GenericResponse, tags=["Memories"])
def create_user_memory(user_id: str, request: MemoryRequest):
    """Manually add a memory fact for a user."""
    memory_id = save_memory(user_id, request.content)
    if memory_id:
        return GenericResponse(
            success=True,
            message=f"Memory successfully stored with ID: {memory_id}"
        )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to save memory to database."
    )

@router.delete("/memories/{user_id}/{memory_id}", response_model=GenericResponse, tags=["Memories"])
def delete_user_memory(user_id: str, memory_id: str):
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
    """Clear all long-term memories stored for a user."""
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

# ==========================================
# Conversation Endpoints
# ==========================================

@router.get("/conversations/{user_id}", response_model=List[ConversationSchema], tags=["Conversations"])
def get_user_conversations(user_id: str):
    """Retrieve all conversations for a user."""
    return get_conversations(user_id)

@router.post("/conversations/{user_id}", response_model=ConversationSchema, tags=["Conversations"])
def create_user_conversation(user_id: str, title: str = "New Conversation"):
    """Create a new conversation session for a user."""
    conv_id = create_conversation(user_id, title=title)
    # Return the created conversation details
    return ConversationSchema(
        id=conv_id,
        title=title,
        created_at="",
        updated_at=""
    )

@router.delete("/conversations/{user_id}/{conversation_id}", response_model=GenericResponse, tags=["Conversations"])
def delete_user_conversation(user_id: str, conversation_id: str):
    """Delete a conversation session and all its messages."""
    success = delete_conversation(user_id, conversation_id)
    if success:
        return GenericResponse(
            success=True,
            message=f"Conversation '{conversation_id}' deleted successfully."
        )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to delete conversation '{conversation_id}'."
    )

@router.post("/chat/clear/{user_id}", response_model=GenericResponse, tags=["Chat"])
def clear_user_chat_history(user_id: str, conversation_id: str = "default_conversation"):
    """Clear the active conversation message history (short-term memory) for a user."""
    success = clear_chat_history(user_id, conversation_id=conversation_id)
    if success:
        return GenericResponse(
            success=True,
            message=f"Short-term chat history for user '{user_id}' and conv '{conversation_id}' cleared successfully."
        )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to clear short-term chat history."
    )

@router.get("/chat/{user_id}", response_model=List[MessageSchema], tags=["Chat"])
def get_user_chat_history(user_id: str, conversation_id: str = "default_conversation"):
    """Retrieve the active conversation message history (short-term memory) for a user."""
    try:
        messages = load_chat_history(user_id, conversation_id=conversation_id)
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
