from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    """LangGraph State representing the chatbot conversation state."""
    messages: Annotated[List[BaseMessage], add_messages]
    global_memories: List[str]
    conversation_memories: List[str]
    user_id: str
    pending_clarifications: List[str]
