import os
import logging
from typing import List, Optional
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

class MemoryAction(BaseModel):
    action_type: str = Field(
        ...,
        description="Type of action: 'ADD' (new memory), 'UPDATE' (replace/update a conflicting memory), 'NO_OP' (duplicate or irrelevant fact), 'ASK_CLARIFICATION' (conflict requiring user confirmation)"
    )
    fact: str = Field(..., description="The fact or preference statement to store.")
    existing_memory_id: Optional[str] = Field(
        None,
        description="The ID of the conflicting or existing memory to update/clarify."
    )
    clarifying_question: Optional[str] = Field(
        None,
        description="The clarifying question to ask the user if action_type is 'ASK_CLARIFICATION'."
    )

class MemoryAnalysis(BaseModel):
    actions: List[MemoryAction] = Field(
        default_factory=list,
        description="List of memory updates, conflict resolutions, and clarifications."
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
    """Retrieve long-term memories and pending clarifications from LangGraph store."""
    user_id = state.get("user_id", "default_user")
    pending_questions = []
    if store is not None:
        memories_items = store.search((user_id,))
        memories_text = [item.value["content"] for item in memories_items]
        
        # Load pending clarifications
        clarifications = store.search(("pending_clarifications", user_id))
        pending_questions = [c.value["question"] for c in clarifications]
    else:
        memories_list = get_memories(user_id)
        memories_text = [m["content"] for m in memories_list]
    return {
        "long_term_memories": memories_text,
        "pending_clarifications": pending_questions
    }

async def chatbot_node(state: ChatState, config: RunnableConfig) -> dict:
    """Invoke the LLM using message history, long-term memories, and pending clarifications with real-time streaming."""
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
        
    # Append pending clarifications if any
    pending = state.get("pending_clarifications", [])
    if pending:
        pending_bullet = "\n".join([f"- {q}" for q in pending])
        system_instruction += (
            "\n\nCRITICAL: There is a pending clarification you need to ask the user to resolve conflicting information in their memory profile:\n"
            f"{pending_bullet}\n"
            "You MUST ask the user this clarifying question naturally as part of your response to resolve the conflict."
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
    """Use the LLM to decide what facts from the latest human-AI exchange to save, update, or clarify in the store."""
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
        
    # Retrieve existing memories
    existing_mems = []
    if store is not None:
        try:
            existing_items = store.search((user_id,))
            existing_mems = [{"id": item.key, "content": item.value["content"]} for item in existing_items]
        except Exception as e:
            logger.error(f"Error loading existing memories for analysis: {e}")
    else:
        existing_mems = get_memories(user_id)
        
    memories_formatted = "\n".join([f"ID: {m['id']} - content: '{m['content']}'" for m in existing_mems])
    
    # Retrieve pending clarifications
    pending_clarifications = []
    if store is not None:
        try:
            clarifications = store.search(("pending_clarifications", user_id))
            pending_clarifications = [
                {
                    "id": c.key,
                    "question": c.value.get("question"),
                    "new_fact": c.value.get("new_fact"),
                    "old_fact_id": c.value.get("old_fact_id")
                }
                for c in clarifications
            ]
        except Exception as e:
            logger.error(f"Error loading pending clarifications for analysis: {e}")
            
    pending_formatted = "\n".join([
        f"Clarification ID: {pc['id']} - question: '{pc['question']}', new_fact: '{pc['new_fact']}', old_fact_id: '{pc['old_fact_id']}'"
        for pc in pending_clarifications
    ])
    
    analysis_prompt = (
        "Analyze the following conversation turn between the User and the AI to identify persistent facts worth remembering long-term (e.g., user name, job, college, preferences).\n"
        "Here are the user's EXISTING long-term memories:\n"
        f"{memories_formatted or 'No existing memories.'}\n\n"
        "Here are the PENDING clarifications currently being asked to the user:\n"
        f"{pending_formatted or 'No pending clarifications.'}\n\n"
        f"Latest turn:\n"
        f"User: {last_human_msg}\n"
        f"AI: {last_ai_msg}\n\n"
        "For any candidate facts, decide what actions to take. Output a list of actions matching the schema:\n"
        "Choose action_type from:\n"
        "- 'NO_OP': The fact is already present (duplicate) or is not a persistent fact.\n"
        "- 'ADD': This is a new fact with no contradictions to existing memories.\n"
        "- 'UPDATE': This is a direct correction/progression of a conflicting existing memory (e.g. 2nd year -> 3rd year). You must provide the existing_memory_id of the old memory to replace/delete.\n"
        "- 'ASK_CLARIFICATION': There is a conflict/contradiction with an existing memory that is unclear (e.g. user says they are XYZ, but memory says Anmol). Do not update yet. Provide the existing_memory_id and a polite, natural clarifying_question to ask the user.\n\n"
        "Additionally, check if the user has answered any of the PENDING clarifications. If they did, output an 'UPDATE' action to apply the chosen fact (using the pending clarification's old_fact_id as the existing_memory_id) and we will resolve it."
    )
    
    extracted = None
    try:
        llm = get_llm(provider, model_name, api_key)
        structured_llm = llm.with_structured_output(MemoryAnalysis)
        extracted = structured_llm.invoke([HumanMessage(content=analysis_prompt)])
    except Exception as e:
        logger.warning(f"Memory extraction failed with primary provider '{provider}': {e}. Trying Groq fallback...")
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if provider != "groq" and groq_api_key:
            try:
                llm = get_llm("groq", "llama-3.3-70b-versatile", groq_api_key)
                structured_llm = llm.with_structured_output(MemoryAnalysis)
                extracted = structured_llm.invoke([HumanMessage(content=analysis_prompt)])
            except Exception as fallback_err:
                logger.error(f"Groq fallback memory extraction failed: {fallback_err}")
        else:
            logger.error(f"Memory extraction failed completely: {e}")
            
    if extracted and extracted.actions:
        import uuid
        for action in extracted.actions:
            act_type = action.action_type.upper()
            fact = action.fact
            old_id = action.existing_memory_id
            
            if act_type == "ADD":
                if store is not None:
                    memory_id = str(uuid.uuid4())
                    store.put((user_id,), memory_id, {"content": fact})
                    logger.info(f"Added new memory via store: {fact}")
                else:
                    save_memory(user_id, fact)
                    
            elif act_type == "UPDATE":
                if store is not None:
                    if old_id:
                        try:
                            store.delete((user_id,), old_id)
                            logger.info(f"Deleted old memory ID {old_id} due to update.")
                        except Exception as e:
                            logger.warning(f"Could not delete old memory {old_id}: {e}")
                    memory_id = str(uuid.uuid4())
                    store.put((user_id,), memory_id, {"content": fact})
                    logger.info(f"Updated memory via store: {fact}")
                else:
                    if old_id:
                        from src.services.memory import delete_memory
                        delete_memory(user_id, old_id)
                    save_memory(user_id, fact)
                    
            elif act_type == "ASK_CLARIFICATION":
                if store is not None:
                    clarification_id = str(uuid.uuid4())
                    store.put(
                        ("pending_clarifications", user_id),
                        clarification_id,
                        {
                            "question": action.clarifying_question,
                            "new_fact": fact,
                            "old_fact_id": old_id
                        }
                    )
                    logger.info(f"Saved pending clarification: {action.clarifying_question}")
                    
        # Clean up pending clarifications that have been resolved
        if store is not None:
            for action in extracted.actions:
                if action.action_type.upper() in ("UPDATE", "NO_OP", "ADD") and action.existing_memory_id:
                    for pc in pending_clarifications:
                        if pc["old_fact_id"] == action.existing_memory_id or pc["new_fact"] == action.fact:
                            try:
                                store.delete(("pending_clarifications", user_id), pc["id"])
                                logger.info(f"Cleared pending clarification ID {pc['id']} (resolved).")
                            except Exception as e:
                                logger.warning(f"Could not delete pending clarification {pc['id']}: {e}")
                                
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
