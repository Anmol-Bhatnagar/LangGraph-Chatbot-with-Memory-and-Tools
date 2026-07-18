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

# Build the workflow graph
workflow = StateGraph(ChatState)

# Add Nodes
workflow.add_node("load_memories", load_memories_node)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("extract_memory", extract_memory_node)
workflow.add_node("trim_history", trim_history_node)

# Add Edges
workflow.add_edge(START, "load_memories")
workflow.add_edge("load_memories", "chatbot")
workflow.add_edge("chatbot", "extract_memory")
workflow.add_edge("extract_memory", "trim_history")
workflow.add_edge("trim_history", END)

# Compile Graph
chatbot_app = workflow.compile()
logger.info("LangGraph StateGraph compiled successfully.")
