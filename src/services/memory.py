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
        conn.commit()
        conn.close()
        logger.info(f"SQLite database initialized successfully at: {settings.DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing SQLite database: {e}")

def save_chat_history(user_id: str, messages: List[BaseMessage]):
    """Save the active chat history to SQLite (overwriting old history)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
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
                "INSERT INTO chat_history (user_id, role, content, message_id) VALUES (?, ?, ?, ?)",
                (user_id, role, msg.content, getattr(msg, "id", "") or "")
            )
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(messages)} messages to chat history for user '{user_id}'")
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

def load_chat_history(user_id: str) -> List[BaseMessage]:
    """Load the saved chat history from SQLite for a user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content, message_id FROM chat_history WHERE user_id = ? ORDER BY id ASC",
            (user_id,)
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

def clear_chat_history(user_id: str) -> bool:
    """Delete all chat history for a user from SQLite."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        logger.info(f"Cleared chat history for user '{user_id}'")
        return True
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return False

def save_memory(user_id: str, content: str) -> int:
    """Save a memory content for a given user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memories (user_id, content) VALUES (?, ?)",
            (user_id, content)
        )
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"Saved memory {memory_id} for user '{user_id}': {content}")
        return memory_id
    except Exception as e:
        logger.error(f"Error saving memory to database: {e}")
        return -1

def get_memories(user_id: str) -> List[Dict[str, Any]]:
    """Retrieve all memory content strings stored in the SQLite DB for a user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"id": r[0], "content": r[1], "created_at": r[2]} for r in rows]
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        return []

def delete_memory(user_id: str, memory_id: int) -> bool:
    """Delete a memory string from SQLite by its unique ID for a user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM memories WHERE user_id = ? AND id = ?",
            (user_id, memory_id)
        )
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        logger.info(f"Deleted memory {memory_id} for user '{user_id}': status {rows_affected > 0}")
        return rows_affected > 0
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        return False

def clear_all_memories(user_id: str) -> bool:
    """Delete all memories for a user from SQLite."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        logger.info(f"Cleared all {rows_affected} memories for user '{user_id}'")
        return True
    except Exception as e:
        logger.error(f"Error clearing memories: {e}")
        return False
