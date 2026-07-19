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

class TestGraphNodes(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_all_memories("test_user_node")
        
    def test_load_memories_node(self):
        state = {"user_id": "test_user_node", "messages": [], "global_memories": [], "conversation_memories": []}
        res = load_memories_node(state, {})
        self.assertEqual(res["global_memories"], [])
        self.assertEqual(res["conversation_memories"], [])
        
    @patch("src.agents.nodes.get_llm")
    async def test_chatbot_node(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Define mock async generator for astream
        async def mock_astream(*args, **kwargs):
            yield AIMessage(content="Hello there!", id="ai1")
            
        mock_llm.astream = mock_astream
        
        state = {
            "user_id": "test_user_node",
            "messages": [HumanMessage(content="Hi", id="h1")],
            "global_memories": [],
            "conversation_memories": []
        }
        res = await chatbot_node(state, {})
        self.assertEqual(len(res["messages"]), 1)
        self.assertEqual(res["messages"][0].content, "Hello there!")
        
    @patch("src.agents.nodes.get_llm")
    def test_extract_memory_node(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # mock structured output
        mock_structured = MagicMock()
        from src.agents.nodes import MemoryAnalysis, MemoryAction
        mock_structured.invoke.return_value = MemoryAnalysis(
            actions=[MemoryAction(action_type="ADD", fact="User prefers Python")]
        )
        mock_llm.with_structured_output.return_value = mock_structured
        
        state = {
            "user_id": "test_user_node",
            "messages": [
                HumanMessage(content="I write python code.", id="h1"),
                AIMessage(content="Awesome, python is great!", id="ai1")
            ],
            "global_memories": [],
            "conversation_memories": []
        }
        extract_memory_node(state, {})
        
        mems = get_memories("test_user_node")
        self.assertEqual(len(mems), 1)
        self.assertEqual(mems[0]["content"], "User prefers Python")
        
    @patch("src.agents.nodes.get_llm")
    def test_trim_history_node(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # mock structured output
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
            "global_memories": [],
            "conversation_memories": []
        }
        # Configure limit = 3 messages, which will trigger pruning of 2 messages
        config = {"configurable": {"limit": 3}}
        res = trim_history_node(state, config)
        
        self.assertEqual(len(res["messages"]), 2) # Returns 2 RemoveMessage instances
        self.assertTrue(all(m.id in ["msg_1", "msg_2"] for m in res["messages"]))

    @patch("src.agents.nodes.get_llm")
    def test_message_counter_frequency_trigger(self, mock_get_llm):
        import uuid
        conv_id = f"test_conv_freq_{uuid.uuid4().hex[:6]}"

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_structured = MagicMock()
        from src.agents.nodes import MemoryAnalysis, MemoryAction
        mock_structured.invoke.return_value = MemoryAnalysis(actions=[])
        mock_llm.with_structured_output.return_value = mock_structured

        # Mock title invoke response
        title_response_mock = MagicMock()
        title_response_mock.content = "New Conversation Title"
        mock_llm.invoke.return_value = title_response_mock

        state = {
            "user_id": "test_user_node",
            "messages": [
                HumanMessage(content="Hello", id="h1"),
                AIMessage(content="Hi there!", id="ai1")
            ]
        }
        
        # Test frequency = 3. 
        config = {
            "configurable": {
                "conversation_id": conv_id,
                "global_memory_frequency": 3,
                "provider": "google",
                "model": "gemini-2.5-flash"
            }
        }

        # 1. First invocation: counter increments to 1, trigger should not fire (mock_llm.invoke for title not called)
        extract_memory_node(state, config)
        
        # Check counter value in DB
        from src.services.memory import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT msg_count_since_update, title FROM conversations WHERE id = ?", (conv_id,))
        row = cursor.fetchone()
        self.assertEqual(row[0], 1)
        self.assertNotEqual(row[1], "New Conversation Title")
        conn.close()

        # 2. Second invocation: counter increments to 2, trigger still should not fire
        extract_memory_node(state, config)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT msg_count_since_update FROM conversations WHERE id = ?", (conv_id,))
        self.assertEqual(cursor.fetchone()[0], 2)
        conn.close()

        # 3. Third invocation: counter increments to 3 (reaches frequency 3), trigger fires, mock_llm.invoke called for title, resets to 0
        extract_memory_node(state, config)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT msg_count_since_update, title FROM conversations WHERE id = ?", (conv_id,))
        row = cursor.fetchone()
        self.assertEqual(row[0], 0) # Counter reset
        self.assertEqual(row[1], "New Conversation Title") # Title updated
        conn.close()

