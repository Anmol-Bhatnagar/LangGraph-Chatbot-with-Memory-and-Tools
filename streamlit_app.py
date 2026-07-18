import streamlit as st
import os
import uuid
import requests
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://127.0.0.1:8000")

def check_backend_health() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{BACKEND_API_URL}/healthz", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def api_get_chat_history(user_id: str) -> list:
    """Fetch short-term chat history from FastAPI backend."""
    try:
        response = requests.get(f"{BACKEND_API_URL}/api/v1/chat/{user_id}", timeout=10)
        if response.status_code == 200:
            msgs = []
            for msg in response.json():
                if msg["role"] == "user":
                    msgs.append(HumanMessage(content=msg["content"], id=msg.get("id")))
                elif msg["role"] == "assistant":
                    msgs.append(AIMessage(content=msg["content"], id=msg.get("id")))
            return msgs
    except Exception as e:
        st.error(f"Error fetching chat history from backend: {e}")
    return []

def api_clear_chat_history(user_id: str) -> bool:
    """Clear short-term chat history on FastAPI backend."""
    try:
        response = requests.post(f"{BACKEND_API_URL}/api/v1/chat/clear/{user_id}", timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error clearing chat history: {e}")
    return False

def api_chat(user_id: str, message: str, provider: str, model: str, api_key: str, limit: int) -> dict:
    """Invoke the chatbot API."""
    payload = {
        "user_id": user_id,
        "message": message,
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "limit": limit
    }
    headers = {"X-API-Token": "streamlit-frontend-client"}
    try:
        response = requests.post(f"{BACKEND_API_URL}/api/v1/chat", json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            detail = response.json().get("detail", "Unknown error")
            st.error(f"Backend API Error ({response.status_code}): {detail}")
    except Exception as e:
        st.error(f"Error communicating with backend API: {e}")
    return None

def api_get_memories(user_id: str) -> list:
    """Fetch long-term memories from FastAPI backend."""
    try:
        response = requests.get(f"{BACKEND_API_URL}/api/v1/memories/{user_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching long-term memories: {e}")
    return []

def api_save_memory(user_id: str, content: str) -> bool:
    """Save a custom fact/memory to FastAPI backend."""
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/api/v1/memories/{user_id}", 
            json={"content": content}, 
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error saving memory to backend: {e}")
    return False

def api_delete_memory(user_id: str, memory_id: int) -> bool:
    """Delete a memory from FastAPI backend."""
    try:
        response = requests.delete(f"{BACKEND_API_URL}/api/v1/memories/{user_id}/{memory_id}", timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error deleting memory on backend: {e}")
    return False

def api_clear_all_memories(user_id: str) -> bool:
    """Clear all memories for a user on FastAPI backend."""
    try:
        response = requests.delete(f"{BACKEND_API_URL}/api/v1/memories/{user_id}", timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error clearing memories on backend: {e}")
    return False

# ==========================================
# 1. UI Configuration & Custom Styling
# ==========================================

st.set_page_config(
    page_title="LangGraph Chatbot with Dual Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek, modern styling
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #6366F1, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .memory-card {
        background-color: #1E293B;
        border-left: 4px solid #6366F1;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        color: #E2E8F0;
    }
    .memory-id {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }
    .stButton>button {
        border-radius: 8px;
    }
    .sidebar-api-ok {
        color: #10B981;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 5px;
    }
    .sidebar-api-warn {
        color: #F59E0B;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. Sidebar configuration panel
# ==========================================

with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/brain.png", width=80)
    st.markdown("## Configuration Panel")
    st.markdown("Set up your LLM details and memory settings below.")
    
    st.divider()
    
    # 2a. LLM Provider Choice
    provider = st.selectbox(
        "LLM Provider",
        options=["google", "groq"],
        format_func=lambda x: "Google Gemini" if x == "google" else "Groq Cloud",
        help="Select the AI service provider you want to use."
    )
    
    # Model configuration depending on provider
    if provider == "google":
        model = st.selectbox(
            "Model Name",
            options=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.5-pro"],
            index=0,
            help="gemini-2.5-flash is recommended for general chat; gemini-2.5-pro is slower but more capable."
        )
        api_key_placeholder = "Enter Gemini API Key..."
        api_env_var = "GEMINI_API_KEY"
    else:
        model = st.selectbox(
            "Model Name",
            options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
            index=0,
            help="llama-3.3-70b-versatile is highly intelligent; llama-3.1-8b-instant is fast and light."
        )
        api_key_placeholder = "Enter Groq API Key..."
        api_env_var = "GROQ_API_KEY"
        
    # Get API key from environment variable or input
    env_api_key = os.environ.get(api_env_var, "")
    api_key = st.text_input(
        f"API Key ({api_env_var})",
        type="password",
        value=env_api_key if env_api_key else "",
        placeholder=api_key_placeholder,
        help="Provide your API key. If already set as an environment variable, it will be auto-filled."
    )
    
    if api_key:
        st.markdown(f'<div class="sidebar-api-ok">✓ Key loaded successfully.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sidebar-api-warn">⚠ Key required to chat.</div>', unsafe_allow_html=True)
        
    # Check backend server status
    backend_status = check_backend_health()
    if backend_status:
        st.markdown(f'<div class="sidebar-api-ok">✓ Backend API Online</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sidebar-api-warn">⚠ Backend API Offline ({BACKEND_API_URL})</div>', unsafe_allow_html=True)
        
    st.divider()
    
    # 2b. Memory Configurations
    st.markdown("### Memory Constraints")
    
    user_id = st.text_input(
        "User/Session ID",
        value="default_user",
        help="SQLite memories are segmented by User ID. Switch ID to chat with a different user profile."
    )
    
    limit = st.slider(
        "Short-Term Memory Limit",
        min_value=4,
        max_value=20,
        value=6,
        step=2,
        help="The maximum number of recent messages kept in active memory. Exceeding this limit triggers historical trimming and long-term summarization."
    )
    
    st.divider()
    
    # 2c. Session Actions
    st.markdown("### Actions")
    
    col_clear_chat, col_clear_db = st.columns(2)
    with col_clear_chat:
        if st.button("Clear Chat", help="Clear the current conversation state history"):
            if api_clear_chat_history(user_id):
                if "graph_state" in st.session_state:
                    st.session_state.graph_state["messages"] = []
                st.toast("Short-term chat history cleared!")
                st.rerun()
                
    with col_clear_db:
        if st.button("Clear SQLite DB", type="secondary", help="Clear all stored long-term facts for this user from the SQLite database"):
            if api_clear_all_memories(user_id):
                if "graph_state" in st.session_state:
                    st.session_state.graph_state["long_term_memories"] = []
                st.toast("SQLite long-term memory cleared!")
                st.rerun()


# ==========================================
# 3. Main Layout & Chat Setup
# ==========================================

st.markdown('<div class="main-header">LangGraph Memory Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Chatbot featuring Short-term Conversation Trimming & SQLite-persisted Long-term Semantic Memories.</div>', unsafe_allow_html=True)

# Initialize Session State for LangGraph
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [],
        "long_term_memories": [],
        "user_id": user_id
    }
    # Initial load of messages from backend
    st.session_state.graph_state["messages"] = api_get_chat_history(user_id)

# Ensure the active user ID matches the sidebar
if st.session_state.graph_state.get("user_id") != user_id:
    st.session_state.graph_state["user_id"] = user_id
    # Fetch existing chat history for the new user profile
    st.session_state.graph_state["messages"] = api_get_chat_history(user_id)
    st.toast(f"Switched user profile to: '{user_id}'")

# Split page into chat section (left) and memory debugging panel (right)
col_chat, col_debug = st.columns([3, 2])


# ==========================================
# 4. Chat Log rendering (Left Column)
# ==========================================

with col_chat:
    st.markdown("### Conversation")
    
    # Prompt user for API key if missing
    if not api_key:
        st.warning("Please configure your API Key in the sidebar to start chatting.")
    
    # Container for message history
    chat_container = st.container(height=500)
    
    with chat_container:
        messages = st.session_state.graph_state.get("messages", [])
        
        # If no messages, render welcome
        if not messages:
            st.info(
                f"Welcome! Start chatting with the agent. The short-term memory limit is set to "
                f"**{limit} messages** in the sidebar. Once exceeded, older messages will be "
                f"automatically pruned and summarized into long-term memory."
            )
            
        for msg in messages:
            # Skip LangGraph SystemMessages and RemoveMessages in main chat UI
            if isinstance(msg, SystemMessage):
                continue
            
            if isinstance(msg, HumanMessage):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg.content)

    # Input text box
    user_input = st.chat_input("Say something...", disabled=not api_key)
    
    if user_input:
        # 1. Display User Message immediately
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
        
        # 2. Invoke LangGraph backend API
        with st.spinner("Agent is thinking & updating memory..."):
            try:
                api_result = api_chat(
                    user_id=user_id,
                    message=user_input,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    limit=limit
                )
                if api_result:
                    # Map the active_messages back to Langchain messages
                    msgs = []
                    for msg in api_result["active_messages"]:
                        if msg["role"] == "user":
                            msgs.append(HumanMessage(content=msg["content"], id=msg.get("id")))
                        elif msg["role"] == "assistant":
                            msgs.append(AIMessage(content=msg["content"], id=msg.get("id")))
                            
                    st.session_state.graph_state = {
                        "messages": msgs,
                        "long_term_memories": api_result["long_term_memories"],
                        "user_id": user_id
                    }
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error executing chat loop: {e}")


# ==========================================
# 5. Memory Debugging & Visualization (Right Column)
# ==========================================

with col_debug:
    st.markdown("### Memory Debugger")
    
    # Set up Tabs for Short Term, Long Term, and How it Works
    tab_long, tab_short, tab_info = st.tabs([
        "📁 Long-Term Memory (SQLite)", 
        "⏳ Short-Term Memory (Active Messages)", 
        "⚙ How it Works"
    ])
    
    # 5a. Long-term memory tab
    with tab_long:
        st.markdown("#### Stored SQLite Facts")
        st.markdown(
            "These facts are fetched from the local SQLite database at the start of the chat. "
            "They are injected into the chatbot's system prompt to give it long-term recall."
        )
        
        # Fetch from backend API
        db_memories = api_get_memories(user_id)
        
        if db_memories:
            for mem in db_memories:
                col_mem_text, col_mem_del = st.columns([5, 1])
                with col_mem_text:
                    st.markdown(
                        f'<div class="memory-card">'
                        f'<div>{mem["content"]}</div>'
                        f'<div class="memory-id">ID: {mem["id"]} | Created: {mem["created_at"]}</div>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                with col_mem_del:
                    # Individual memory deletion
                    if st.button("🗑", key=f"del_{mem['id']}", help=f"Delete memory ID {mem['id']}"):
                        if api_delete_memory(user_id, mem['id']):
                            st.toast("Memory deleted!")
                            st.rerun()
        else:
            st.info("No long-term memories found for this user in the SQLite database. Try chatting to let the LLM extract some facts!")
            
        st.divider()
        
        # Admin tool to manually add a long-term memory
        st.markdown("##### Add Memory Manually (Admin)")
        manual_mem = st.text_input("Enter a custom fact to store:", placeholder="e.g. User is allergic to peanuts.")
        if st.button("Save to SQLite"):
            if manual_mem.strip():
                if api_save_memory(user_id, manual_mem.strip()):
                    st.toast("Saved manual memory!")
                    st.rerun()

    # 5b. Short-term memory tab
    with tab_short:
        st.markdown("#### Active Graph Messages")
        st.markdown(
            "These are the exact message objects currently in the LangGraph state. "
            f"When this count exceeds the limit of **{limit}**, trimming will trigger."
        )
        
        active_msgs = st.session_state.graph_state.get("messages", [])
        st.metric("Active Messages Count", len(active_msgs))
        
        if active_msgs:
            for i, msg in enumerate(active_msgs):
                # Determine message role color and type name
                if isinstance(msg, HumanMessage):
                    role_type = "HumanMessage"
                    bg_color = "#0B5394"
                elif isinstance(msg, AIMessage):
                    role_type = "AIMessage"
                    bg_color = "#38761D"
                else:
                    role_type = "SystemMessage / Other"
                    bg_color = "#741B47"
                    
                st.markdown(
                    f'<div style="background-color: {bg_color}; padding: 10px; border-radius: 6px; margin-bottom: 8px; font-size: 0.9rem;">'
                    f'<strong>[{i}] {role_type}</strong> (ID: {getattr(msg, "id", "N/A")})<br/>'
                    f'<div style="margin-top: 5px; opacity: 0.95; white-space: pre-wrap;">{msg.content}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Short-term chat history is empty.")

    # 5c. Help / explanation tab
    with tab_info:
        st.markdown("#### System Architecture")
        st.markdown("""
        This chatbot demonstrates an advanced AI memory paradigm using **LangGraph** and **SQLite**:
        
        1. **Short-Term Memory**:
           - Consists of the direct list of messages (`HumanMessage` and `AIMessage`) inside the LangGraph state.
           - This history is capped (e.g. at 6 messages) to prevent context-window bloat, reduce latency, and control costs.
        
        2. **Long-Term Memory**:
           - A local SQLite database (`memories.db`) storing facts extracted by the LLM.
           - These facts are automatically injected into the **System Prompt** on each interaction.
        
        3. **Dynamic Extraction Nodes**:
           - **Turn Analyzer**: Runs after every chat message. The LLM reviews the latest exchange and decides whether to extract user facts or preferences to save in SQLite.
           - **History Pruning**: Triggers when history exceeds the limit. The older messages are summarized by the LLM into keypoints, saved in SQLite, and removed from the active message state.
        """)
