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

logger = logging.getLogger("AgentGraph")

# Initialize database tables on module load
init_db()

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
chatbot_app = chatbot_workflow.compile()
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
memory_app = memory_workflow.compile()
logger.info("LangGraph background memory_app compiled successfully.")
