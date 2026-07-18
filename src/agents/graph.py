import logging
from langgraph.graph import StateGraph, START, END

from src.agents.state import ChatState
from src.agents.nodes import (
    load_memories_node,
    chatbot_node,
    extract_memory_node,
    trim_history_node
)
from src.services.memory import init_db

from langgraph.store.memory import InMemoryStore
from src.services.memory import get_db_connection

logger = logging.getLogger("AgentGraph")

# Initialize database tables on module load
init_db()

# Initialize LangGraph Store for long-term memory
store = InMemoryStore()

def sync_db_to_store():
    """Load all existing persistent memories from SQLite into LangGraph store at startup."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, user_id, content FROM memories")
        rows = cursor.fetchall()
        conn.close()
        for mem_id, user_id, content in rows:
            store.put((user_id,), str(mem_id), {"content": content})
        logger.info(f"Successfully loaded {len(rows)} memories from SQLite into LangGraph store.")
    except Exception as e:
        logger.error(f"Error loading memories from SQLite into store: {e}")

sync_db_to_store()

# Build the primary chatbot workflow (user-facing)
chatbot_workflow = StateGraph(ChatState)

# Add chatbot nodes
chatbot_workflow.add_node("load_memories", load_memories_node)
chatbot_workflow.add_node("chatbot", chatbot_node)

# Add chatbot edges
chatbot_workflow.add_edge(START, "load_memories")
chatbot_workflow.add_edge("load_memories", "chatbot")
chatbot_workflow.add_edge("chatbot", END)

# Compile primary chatbot graph
chatbot_app = chatbot_workflow.compile(store=store)
logger.info("LangGraph primary chatbot_app compiled successfully.")

# Build the background memory workflow (async processing)
memory_workflow = StateGraph(ChatState)

# Add memory nodes
memory_workflow.add_node("extract_memory", extract_memory_node)
memory_workflow.add_node("trim_history", trim_history_node)

# Add memory edges
memory_workflow.add_edge(START, "extract_memory")
memory_workflow.add_edge("extract_memory", "trim_history")
memory_workflow.add_edge("trim_history", END)

# Compile background memory graph
memory_app = memory_workflow.compile(store=store)
logger.info("LangGraph background memory_app compiled successfully.")
