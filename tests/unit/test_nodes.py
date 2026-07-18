import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.nodes import (
    load_memories_node,
    chatbot_node,
    extract_memory_node,
    trim_history_node,
    MemoryExtraction,
    PrunedMemoryExtraction
)
from src.services.memory import clear_all_memories, get_memories

class TestGraphNodes(unittest.TestCase):
    def setUp(self):
        clear_all_memories("test_user_node")
        
    def test_load_memories_node(self):
        state = {"user_id": "test_user_node", "messages": [], "long_term_memories": []}
        res = load_memories_node(state, {})
        self.assertEqual(res["long_term_memories"], [])
        
    @patch("src.agents.nodes.get_llm")
    def test_chatbot_node(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Hello there!", id="ai1")
        
        state = {
            "user_id": "test_user_node",
            "messages": [HumanMessage(content="Hi", id="h1")],
            "long_term_memories": []
        }
        res = chatbot_node(state, {})
        self.assertEqual(len(res["messages"]), 1)
        self.assertEqual(res["messages"][0].content, "Hello there!")
        
    @patch("src.agents.nodes.get_llm")
    def test_extract_memory_node(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(new_memories=["User prefers Python"])
        mock_llm.with_structured_output.return_value = mock_structured
        
        state = {
            "user_id": "test_user_node",
            "messages": [
                HumanMessage(content="I write python code.", id="h1"),
                AIMessage(content="Awesome, python is great!", id="ai1")
            ],
            "long_term_memories": []
        }
        extract_memory_node(state, {})
        
        mems = get_memories("test_user_node")
        self.assertEqual(len(mems), 1)
        self.assertEqual(mems[0]["content"], "User prefers Python")
        
    @patch("src.agents.nodes.get_llm")
    def test_trim_history_node(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(extracted_knowledge=["Pruned message highlight: Python dev"])
        mock_llm.with_structured_output.return_value = mock_structured
        
        state = {
            "user_id": "test_user_node",
            "messages": [
                HumanMessage(content="Hi", id="msg_1"),
                AIMessage(content="Hello", id="msg_2"),
                HumanMessage(content="How are you?", id="msg_3"),
                AIMessage(content="Good!", id="msg_4"),
                HumanMessage(content="What's up?", id="msg_5")
            ],
            "long_term_memories": []
        }
        # Configure limit = 3 messages, which will trigger pruning of 2 messages
        config = {"configurable": {"limit": 3}}
        res = trim_history_node(state, config)
        
        self.assertEqual(len(res["messages"]), 2) # Returns 2 RemoveMessage instances
        self.assertTrue(all(m.id in ["msg_1", "msg_2"] for m in res["messages"]))
