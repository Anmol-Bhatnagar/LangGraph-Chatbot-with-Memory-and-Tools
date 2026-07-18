import sqlite3
import logging
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from src.config.settings import settings

logger = logging.getLogger("DBService")

def get_db_connection():
    """Create a connection to the SQLite database with WAL mode and custom timeout for concurrent load."""
    conn = sqlite3.connect(settings.DB_PATH, timeout=settings.DB_TIMEOUT)
    if settings.DB_WAL_MODE:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    """Initialize the SQLite database for long-term memory and short-term history."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create memories table (long term memory)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chat_history table (short term message history)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                message_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Check if conversation_id column exists in chat_history
        cursor.execute("PRAGMA table_info(chat_history)")
        columns = [info[1] for info in cursor.fetchall()]
        if "conversation_id" not in columns:
            logger.info("Migrating chat_history table: adding conversation_id column...")
            cursor.execute("ALTER TABLE chat_history ADD COLUMN conversation_id TEXT DEFAULT 'default_conversation'")
            
        conn.commit()
        conn.close()
        logger.info(f"SQLite database initialized successfully at: {settings.DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing SQLite database: {e}")

def save_chat_history(user_id: str, messages: List[BaseMessage], conversation_id: str = "default_conversation"):
    """Save the active chat history to SQLite for a specific conversation (overwriting old history)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE user_id = ? AND conversation_id = ?", (user_id, conversation_id))
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "human"
            elif isinstance(msg, AIMessage):
                role = "ai"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                continue
            cursor.execute(
                "INSERT INTO chat_history (user_id, role, content, message_id, conversation_id) VALUES (?, ?, ?, ?, ?)",
                (user_id, role, msg.content, getattr(msg, "id", "") or "", conversation_id)
            )
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(messages)} messages to chat history for user '{user_id}' and conv '{conversation_id}'")
        touch_conversation(user_id, conversation_id)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

def load_chat_history(user_id: str, conversation_id: str = "default_conversation") -> List[BaseMessage]:
    """Load the saved chat history from SQLite for a specific conversation."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content, message_id FROM chat_history WHERE user_id = ? AND conversation_id = ? ORDER BY id ASC",
            (user_id, conversation_id)
        )
        rows = cursor.fetchall()
        conn.close()
        messages = []
        for role, content, message_id in rows:
            if role == "human":
                messages.append(HumanMessage(content=content, id=message_id))
            elif role == "ai":
                messages.append(AIMessage(content=content, id=message_id))
            elif role == "system":
                messages.append(SystemMessage(content=content, id=message_id))
        return messages
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []

def clear_chat_history(user_id: str, conversation_id: str = "default_conversation") -> bool:
    """Delete chat history for a specific conversation from SQLite."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE user_id = ? AND conversation_id = ?", (user_id, conversation_id))
        conn.commit()
        conn.close()
        logger.info(f"Cleared chat history for user '{user_id}' and conv '{conversation_id}'")
        return True
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return False

# ==========================================
# Conversation Helpers
# ==========================================

def get_conversations(user_id: str) -> List[Dict[str, Any]]:
    """Retrieve all conversations for a user, sorted by updated_at DESC."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        # If no conversations exist, auto-create a default one
        if not rows:
            create_conversation(user_id, title="Default Conversation", conversation_id="default_conversation")
            return [{"id": "default_conversation", "title": "Default Conversation", "created_at": "", "updated_at": ""}]
            
        return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}")
        return []

def create_conversation(user_id: str, title: str = "New Conversation", conversation_id: str = None) -> str:
    """Create a new conversation session for a user."""
    import uuid
    conv_id = conversation_id or f"conv_{uuid.uuid4().hex[:10]}"
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Check if conversation already exists
        cursor.execute("SELECT 1 FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
        if cursor.fetchone():
            conn.close()
            return conv_id
            
        cursor.execute(
            "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
            (conv_id, user_id, title)
        )
        conn.commit()
        conn.close()
        logger.info(f"Created conversation '{conv_id}' with title '{title}' for user '{user_id}'")
        return conv_id
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return conv_id

def delete_conversation(user_id: str, conversation_id: str) -> bool:
    """Delete a conversation and all its short-term history."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id))
        cursor.execute("DELETE FROM chat_history WHERE conversation_id = ? AND user_id = ?", (conversation_id, user_id))
        conn.commit()
        conn.close()
        logger.info(f"Deleted conversation '{conversation_id}' and its messages for user '{user_id}'")
        return True
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        return False

def touch_conversation(user_id: str, conversation_id: str, title: str = None):
    """Update conversation updated_at time and optionally its title."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Make sure the conversation entry exists
        cursor.execute("SELECT 1 FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
                (conversation_id, user_id, title or "Default Conversation")
            )
        
        if title:
            cursor.execute(
                "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
                (title, conversation_id, user_id)
            )
        else:
            cursor.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
                (conversation_id, user_id)
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error touching conversation: {e}")

def save_memory(user_id: str, content: str) -> str:
    """Save a memory content for a given user using the LangGraph store."""
    try:
        from src.agents.graph import store
        import uuid
        memory_id = str(uuid.uuid4())
        store.put((user_id,), memory_id, {"content": content})
        logger.info(f"Saved memory {memory_id} to LangGraph store for user '{user_id}': {content}")
        return memory_id
    except Exception as e:
        logger.error(f"Error saving memory to LangGraph store: {e}")
        return ""

def get_memories(user_id: str) -> List[Dict[str, Any]]:
    """Retrieve all memory content strings from the LangGraph store for a user."""
    try:
        from src.agents.graph import store
        items = store.search((user_id,))
        return [
            {
                "id": item.key,
                "content": item.value["content"],
                "created_at": item.created_at.isoformat() if (hasattr(item, "created_at") and item.created_at) else ""
            }
            for item in items
        ]
    except Exception as e:
        logger.error(f"Error retrieving memories from LangGraph store: {e}")
        return []

def delete_memory(user_id: str, memory_id: str) -> bool:
    """Delete a memory string from LangGraph store by its key for a user."""
    try:
        from src.agents.graph import store
        store.delete((user_id,), memory_id)
        logger.info(f"Deleted memory {memory_id} from LangGraph store for user '{user_id}'")
        return True
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        return False

def clear_all_memories(user_id: str) -> bool:
    """Delete all memories for a user from LangGraph store."""
    try:
        from src.agents.graph import store
        items = store.search((user_id,))
        for item in items:
            store.delete((user_id,), item.key)
        logger.info(f"Cleared all memories from LangGraph store for user '{user_id}'")
        return True
    except Exception as e:
        logger.error(f"Error clearing memories: {e}")
        return False
