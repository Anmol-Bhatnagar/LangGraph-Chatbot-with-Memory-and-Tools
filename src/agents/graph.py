import logging
import os
import sqlite3
import json
from datetime import datetime, timezone
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore

from src.agents.state import ChatState
from src.agents.nodes import (
    load_memories_node,
    chatbot_node,
    extract_memory_node,
    trim_history_node
)
from src.services.memory import init_db, get_db_connection
from src.config.settings import settings

logger = logging.getLogger("AgentGraph")

# Initialize database tables on module load
init_db()

# ==========================================
# Persistent SQLite-backed Store for Local Development
# ==========================================
class SqliteStore(InMemoryStore):
    def __init__(self, db_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_path = db_path
        self._init_sqlite_db()
        self._load_from_sqlite()

    def _init_sqlite_db(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS langgraph_store (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace, key)
            )
        """)
        conn.commit()
        conn.close()

    def _load_from_sqlite(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT namespace, key, value FROM langgraph_store")
        rows = cursor.fetchall()
        conn.close()
        from langgraph.store.base import Item
        for ns_str, key, val_str in rows:
            try:
                ns = tuple(json.loads(ns_str))
                val = json.loads(val_str)
                self._data[ns][key] = Item(
                    value=val,
                    key=key,
                    namespace=ns,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
            except Exception as e:
                logger.error(f"Error loading row from SQLite store: {e}")
        logger.info(f"Loaded {len(rows)} namespace items from SQLite store.")

    def put(self, namespace, key, value, index=None):
        # Call parent put to update memory dict
        super().put(namespace, key, value, index)
        # Persist to SQLite
        self._persist_put(namespace, key, value)

    def delete(self, namespace, key):
        super().delete(namespace, key)
        self._persist_put(namespace, key, None)

    async def aput(self, namespace, key, value, index=None, **kwargs):
        await super().aput(namespace, key, value, index, **kwargs)
        self._persist_put(namespace, key, value)

    async def adelete(self, namespace, key):
        await super().adelete(namespace, key)
        self._persist_put(namespace, key, None)

    def _persist_put(self, namespace, key, value):
        try:
            ns_str = json.dumps(namespace)
            conn = get_db_connection()
            cursor = conn.cursor()
            if value is None:
                cursor.execute(
                    "DELETE FROM langgraph_store WHERE namespace = ? AND key = ?",
                    (ns_str, key)
                )
            else:
                val_str = json.dumps(value)
                cursor.execute("""
                    INSERT INTO langgraph_store (namespace, key, value, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(namespace, key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = CURRENT_TIMESTAMP
                """, (ns_str, key, val_str))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error persisting put to SQLite store: {e}")

# ==========================================
# Initialize Store (Postgres for Prod, SQLite for Local)
# ==========================================
DATABASE_URL = os.environ.get("DATABASE_URL")
store = None

if DATABASE_URL:
    try:
        from langgraph.store.postgres import PostgresStore
        store = PostgresStore.from_conn_string(DATABASE_URL)
        store.setup()
        logger.info("Using production-grade PostgresStore for long-term memory.")
    except Exception as e:
        logger.error(f"Failed to initialize PostgresStore: {e}. Falling back to SQLite store.")

if store is None:
    store = SqliteStore(db_path=settings.DB_PATH)
    logger.info("Using persistent SqliteStore for long-term memory.")

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
