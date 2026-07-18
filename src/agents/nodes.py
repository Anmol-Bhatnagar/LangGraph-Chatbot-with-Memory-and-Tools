import os
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.agents.state import ChatState
from src.services.memory import get_memories, save_memory

logger = logging.getLogger("AgentNodes")


# ==========================================
# 1. Structured Output Schemas
# ==========================================

class MemoryExtraction(BaseModel):
    new_memories: List[str] = Field(
        default_factory=list,
        description="Concise facts or preferences worth remembering long-term about the user (e.g. 'User likes dark mode', 'User is a developer'). Ignore casual talk or current conversation states. If nothing new, return an empty list."
    )

class PrunedMemoryExtraction(BaseModel):
    extracted_knowledge: List[str] = Field(
        default_factory=list,
        description="Key summaries, decisions, or facts from the archived conversation segment to retain in long-term memory. Return an empty list if there's no persistent value in these messages."
    )


# ==========================================
# 2. LLM Factory Loader
# ==========================================

def get_llm(provider: str, model_name: str, api_key: str):
    """Retrieve the LLM based on user configuration and credentials."""
    if provider == "google":
        if not api_key:
            raise ValueError("API Key must be provided to initialize the Google model.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
            max_retries=1
        )
    elif provider == "groq":
        g_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not g_key:
            raise ValueError("Groq API Key must be provided (either passed in or set as GROQ_API_KEY env var).")
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            api_key=g_key,
            temperature=0.7,
            max_retries=1
        )
    elif provider == "openai":
        if not api_key:
            raise ValueError("API Key must be provided to initialize the OpenAI model.")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.7,
            max_retries=1
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ==========================================
# 3. Prompt Template Loader
# ==========================================

def load_prompt_template(filename: str) -> str:
    """Load a version-controlled prompt template from the prompts folder."""
    path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ==========================================
# 4. Pure Business Logic Graph Nodes
# ==========================================

def load_memories_node(state: ChatState, config: RunnableConfig, *, store=None) -> dict:
    """Retrieve long-term memories from LangGraph store and add to the graph state."""
    user_id = state.get("user_id", "default_user")
    if store is not None:
        memories_items = store.search((user_id,))
        memories_text = [item.value["content"] for item in memories_items]
    else:
        memories_list = get_memories(user_id)
        memories_text = [m["content"] for m in memories_list]
    return {"long_term_memories": memories_text}

async def chatbot_node(state: ChatState, config: RunnableConfig) -> dict:
    """Invoke the LLM using message history and the loaded long-term memories with real-time streaming."""
    configurable = config.get("configurable", {})
    provider = configurable.get("provider", "google")
    model_name = configurable.get("model", "gemini-2.5-flash")
    api_key = configurable.get("api_key", "")
    
    # Construct System Message with Memories
    memories = state.get("long_term_memories", [])
    if memories:
        memories_bullet = "\n".join([f"- {m}" for m in memories])
        try:
            base_template = load_prompt_template("system_base.txt")
            system_instruction = base_template.format(memories=memories_bullet)
        except Exception as e:
            logger.error(f"Failed to load system_base.txt prompt: {e}")
            system_instruction = (
                "You are a helpful and intelligent chatbot with short-term and long-term memory capabilities.\n"
                "Here is what you know about the user from their long-term memory profile:\n"
                f"{memories_bullet}"
            )
    else:
        system_instruction = (
            "You are a helpful and intelligent chatbot with short-term and long-term memory capabilities.\n"
            "Currently, there is no prior long-term memory recorded about the user."
        )
        
    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    
    try:
        llm = get_llm(provider, model_name, api_key)
        response_content = ""
        async for chunk in llm.astream(messages):
            response_content += chunk.content
        response = AIMessage(content=response_content)
    except Exception as e:
        logger.warning(f"Primary LLM provider '{provider}' failed: {e}. Trying Groq fallback...")
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if provider != "groq" and groq_api_key:
            try:
                llm = get_llm("groq", "llama-3.3-70b-versatile", groq_api_key)
                response_content = ""
                async for chunk in llm.astream(messages):
                    response_content += chunk.content
                response = AIMessage(content=response_content)
                logger.info("Successfully fell back to Groq.")
            except Exception as fallback_err:
                logger.error(f"Groq fallback failed: {fallback_err}")
                raise e
        else:
            raise e
            
    return {"messages": [response]}

def extract_memory_node(state: ChatState, config: RunnableConfig, *, store=None) -> dict:
    """Use the LLM to decide what facts from the latest human-AI exchange to save to SQLite and LangGraph store."""
    configurable = config.get("configurable", {})
    provider = configurable.get("provider", "google")
    model_name = configurable.get("model", "gemini-2.5-flash")
    api_key = configurable.get("api_key", "")
    user_id = state.get("user_id", "default_user")
    
    # We need at least a human message and an AI response to analyze the exchange
    if len(state["messages"]) < 2:
        return {}
        
    last_human_msg = None
    last_ai_msg = None
    
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and not last_human_msg:
            last_human_msg = msg.content
        elif isinstance(msg, AIMessage) and not last_ai_msg:
            last_ai_msg = msg.content
        if last_human_msg and last_ai_msg:
            break
            
    if not last_human_msg or not last_ai_msg:
        return {}
        
    analysis_prompt = (
        "Analyze the following conversation turn between the User and the AI.\n"
        "Identify if the user shared any persistent facts, preferences, or important biographical info "
        "worth remembering long-term (e.g. user name, user job, favorite food, specific preferences).\n"
        "Ignore short-term topics, greetings, or questions.\n\n"
        f"User: {last_human_msg}\n"
        f"AI: {last_ai_msg}\n\n"
        "Extract new facts to remember as concise statements."
    )
    
    extracted = None
    try:
        llm = get_llm(provider, model_name, api_key)
        structured_llm = llm.with_structured_output(MemoryExtraction)
        extracted = structured_llm.invoke([HumanMessage(content=analysis_prompt)])
    except Exception as e:
        logger.warning(f"Memory extraction failed with primary provider '{provider}': {e}. Trying Groq fallback...")
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if provider != "groq" and groq_api_key:
            try:
                llm = get_llm("groq", "llama-3.3-70b-versatile", groq_api_key)
                structured_llm = llm.with_structured_output(MemoryExtraction)
                extracted = structured_llm.invoke([HumanMessage(content=analysis_prompt)])
            except Exception as fallback_err:
                logger.error(f"Groq fallback memory extraction failed: {fallback_err}")
        else:
            logger.error(f"Memory extraction failed completely: {e}")
            
    if extracted:
        new_facts = extracted.new_memories
        for fact in new_facts:
            memory_id = save_memory(user_id, fact)
            if store is not None:
                store.put((user_id,), str(memory_id), {"content": fact})
        
    return {}

def trim_history_node(state: ChatState, config: RunnableConfig, *, store=None) -> dict:
    """Trim short-term history when it exceeds the configured limit, extracting highlights first."""
    configurable = config.get("configurable", {})
    provider = configurable.get("provider", "google")
    model_name = configurable.get("model", "gemini-2.5-flash")
    api_key = configurable.get("api_key", "")
    user_id = state.get("user_id", "default_user")
    limit = configurable.get("limit", 6)
    
    messages = state["messages"]
    
    if len(messages) <= limit:
        return {}
        
    num_to_prune = len(messages) - limit
    pruned_messages = messages[:num_to_prune]
    
    if pruned_messages:
        logger.info(f"Trimming short term memory: pruning {num_to_prune} messages.")
        
        pruned_chat_text = []
        for msg in pruned_messages:
            role = "User" if isinstance(msg, HumanMessage) else ("AI" if isinstance(msg, AIMessage) else "System")
            pruned_chat_text.append(f"{role}: {msg.content}")
        chat_segment = "\n".join(pruned_chat_text)
        
        summary_prompt = (
            "The following conversation history segment is about to be deleted from active memory.\n"
            "Please extract any critical details, key decisions, or general knowledge highlights "
            "from this segment that should be preserved in the user's long-term profile.\n"
            "Return them as a list of concise statements. If nothing is worth preserving, return an empty list.\n\n"
            f"Segment:\n{chat_segment}"
        )
        
        extracted = None
        try:
            llm = get_llm(provider, model_name, api_key)
            structured_llm = llm.with_structured_output(PrunedMemoryExtraction)
            extracted = structured_llm.invoke([HumanMessage(content=summary_prompt)])
        except Exception as e:
            logger.warning(f"Memory pruning failed with primary provider '{provider}': {e}. Trying Groq fallback...")
            groq_api_key = os.environ.get("GROQ_API_KEY", "")
            if provider != "groq" and groq_api_key:
                try:
                    llm = get_llm("groq", "llama-3.3-70b-versatile", groq_api_key)
                    structured_llm = llm.with_structured_output(PrunedMemoryExtraction)
                    extracted = structured_llm.invoke([HumanMessage(content=summary_prompt)])
                except Exception as fallback_err:
                    logger.error(f"Groq fallback memory pruning failed: {fallback_err}")
            else:
                logger.error(f"Memory pruning failed completely: {e}")
                
        if extracted:
            knowledge_points = extracted.extracted_knowledge
            for point in knowledge_points:
                fact = f"Summary highlight: {point}"
                memory_id = save_memory(user_id, fact)
                if store is not None:
                    store.put((user_id,), str(memory_id), {"content": fact})
            
        return {"messages": [RemoveMessage(id=msg.id) for msg in pruned_messages if msg.id]}
        
    return {}
