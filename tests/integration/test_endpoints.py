import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

# Mock LLM calls globally for API test
mock_llm = MagicMock()
mock_structured_extract = MagicMock()
mock_structured_trim = MagicMock()

def side_effect(schema):
    from src.agents.nodes import MemoryExtraction, PrunedMemoryExtraction
    if schema == MemoryExtraction:
        return mock_structured_extract
    elif schema == PrunedMemoryExtraction:
        return mock_structured_trim
    return mock_llm

mock_llm.with_structured_output.side_effect = side_effect
mock_llm.invoke.return_value = AIMessage(content="Hello Alice! Nice to meet you.", id="ai_msg_1")
mock_structured_extract.invoke.return_value = MagicMock(new_memories=["User name is Alice"])
mock_structured_trim.invoke.return_value = MagicMock(extracted_knowledge=["Pruned message highlight: Python coding"])

with patch("src.agents.nodes.get_llm", return_value=mock_llm):
    from src.app import app
    from src.services.memory import clear_all_memories, clear_chat_history

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.user_id = "test_user_integration"
        clear_all_memories(self.user_id)
        clear_chat_history(self.user_id)
        
    @patch("src.agents.nodes.get_llm", return_value=mock_llm)
    def test_workflow_endpoints(self, mock_llm_patch):
        # Health check
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "ok"})
        
        # Chat first message
        chat_payload = {
            "user_id": self.user_id,
            "message": "Hello, my name is Alice.",
            "provider": "google",
            "model": "gemini-2.5-flash",
            "api_key": "mock_key",
            "limit": 4
        }
        res = self.client.post("/api/v1/chat", json=chat_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["response"], "Hello Alice! Nice to meet you.")
        self.assertEqual(len(data["active_messages"]), 2)
        
        # Get memories
        res = self.client.get(f"/api/v1/memories/{self.user_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)
        self.assertEqual(res.json()[0]["content"], "User name is Alice")

        # Get chat history (active messages)
        res = self.client.get(f"/api/v1/chat/{self.user_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 2)
        self.assertEqual(res.json()[0]["role"], "user")
        self.assertEqual(res.json()[0]["content"], "Hello, my name is Alice.")
        self.assertEqual(res.json()[1]["role"], "assistant")
        self.assertEqual(res.json()[1]["content"], "Hello Alice! Nice to meet you.")
        
        # Clear chat history
        res = self.client.post(f"/api/v1/chat/clear/{self.user_id}")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])
        
        # Verify chat history is cleared
        res = self.client.get(f"/api/v1/chat/{self.user_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 0)

    @patch("src.agents.nodes.get_llm", return_value=mock_llm)
    def test_conversation_isolation_and_shared_memories(self, mock_llm_patch):
        # 1. Create two conversations
        conv1_payload = {"user_id": self.user_id, "title": "Conv A"}
        res = self.client.post(f"/api/v1/conversations/{self.user_id}?title=Conv+A")
        self.assertEqual(res.status_code, 200)
        conv1_id = res.json()["id"]

        res = self.client.post(f"/api/v1/conversations/{self.user_id}?title=Conv+B")
        self.assertEqual(res.status_code, 200)
        conv2_id = res.json()["id"]

        # Verify both conversations are listed
        res = self.client.get(f"/api/v1/conversations/{self.user_id}")
        self.assertEqual(res.status_code, 200)
        conversations = res.json()
        self.assertTrue(any(c["id"] == conv1_id for c in conversations))
        self.assertTrue(any(c["id"] == conv2_id for c in conversations))

        # 2. Send message in Conv A
        chat_payload_a = {
            "user_id": self.user_id,
            "conversation_id": conv1_id,
            "message": "This is message A",
            "provider": "google",
            "model": "gemini-2.5-flash",
            "api_key": "mock_key"
        }
        res = self.client.post("/api/v1/chat", json=chat_payload_a)
        self.assertEqual(res.status_code, 200)

        # 3. Send message in Conv B
        chat_payload_b = {
            "user_id": self.user_id,
            "conversation_id": conv2_id,
            "message": "This is message B",
            "provider": "google",
            "model": "gemini-2.5-flash",
            "api_key": "mock_key"
        }
        res = self.client.post("/api/v1/chat", json=chat_payload_b)
        self.assertEqual(res.status_code, 200)

        # 4. Verify short-term memory is isolated
        # Conv A should only have message A
        res = self.client.get(f"/api/v1/chat/{self.user_id}?conversation_id={conv1_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 2)
        self.assertEqual(res.json()[0]["content"], "This is message A")

        # Conv B should only have message B
        res = self.client.get(f"/api/v1/chat/{self.user_id}?conversation_id={conv2_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 2)
        self.assertEqual(res.json()[0]["content"], "This is message B")

        # 5. Verify long-term memory is shared (same memory database list for the user)
        res = self.client.get(f"/api/v1/memories/{self.user_id}")
        self.assertEqual(res.status_code, 200)
        # Both conversations trigger memory extraction, writing to the same user profile
        self.assertTrue(len(res.json()) > 0)
        
        # 6. Delete Conv A
        res = self.client.delete(f"/api/v1/conversations/{self.user_id}/{conv1_id}")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])

        # Check list again
        res = self.client.get(f"/api/v1/conversations/{self.user_id}")
        self.assertEqual(res.status_code, 200)
        conversations = res.json()
        self.assertFalse(any(c["id"] == conv1_id for c in conversations))
        self.assertTrue(any(c["id"] == conv2_id for c in conversations))
