// ==========================================================================
// 1. Constants & State Management
// ==========================================================================
const BASE_URL = window.location.origin; // Same origin (port 8000)

const PROVIDER_MODELS = {
    google: [
        { value: 'gemini-2.5-flash', label: 'gemini-2.5-flash (Recommended)' },
        { value: 'gemini-1.5-flash', label: 'gemini-1.5-flash' },
        { value: 'gemini-2.5-pro', label: 'gemini-2.5-pro' }
    ],
    groq: [
        { value: 'llama-3.3-70b-versatile', label: 'llama-3.3-70b-versatile' },
        { value: 'llama-3.1-8b-instant', label: 'llama-3.1-8b-instant' },
        { value: 'gemma2-9b-it', label: 'gemma2-9b-it' }
    ]
};

let activeSession = 'default_user';
let activeProvider = 'google';
let activeModel = 'gemini-2.5-flash';
let activeLimit = 6;
let isBackendOnline = false;
let serverConfig = { google: false, groq: false };

// DOM Elements
const providerSelect = document.getElementById('provider-select');
const modelSelect = document.getElementById('model-select');
const apiKeyInput = document.getElementById('api-key-input');
const apiKeyStatus = document.getElementById('api-key-status');
const sessionIdInput = document.getElementById('session-id-input');
const memoryLimitSlider = document.getElementById('memory-limit-slider');
const memoryLimitVal = document.getElementById('memory-limit-val');
const clearChatBtn = document.getElementById('clear-chat-btn');
const clearDbBtn = document.getElementById('clear-db-btn');
const backendStatus = document.getElementById('backend-status');

const activeSessionName = document.getElementById('active-session-name');
const chatMessagesContainer = document.getElementById('chat-messages-container');
const welcomeMessageCard = document.getElementById('welcome-message-card');
const chatTextarea = document.getElementById('chat-textarea');
const sendMsgBtn = document.getElementById('send-msg-btn');

const sqliteMemoriesContainer = document.getElementById('sqlite-memories-container');
const manualMemoryInput = document.getElementById('manual-memory-input');
const saveManualMemoryBtn = document.getElementById('save-manual-memory-btn');
const activeMsgsCount = document.getElementById('active-msgs-count');
const activeMessagesContainer = document.getElementById('active-messages-container');

// ==========================================================================
// 2. Initialization & Event Listeners
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    loadSettingsFromStorage();
    initTabButtons();
    initEventListeners();
    updateModelOptions();
    checkBackendHealth();
    syncSessionData();
});

function initEventListeners() {
    // LLM Provider select change
    providerSelect.addEventListener('change', (e) => {
        activeProvider = e.target.value;
        updateModelOptions();
        saveSettingsToStorage();
        updateApiKeyInputPlaceholder();
        checkApiKeyLoadStatus();
    });

    // Model select change
    modelSelect.addEventListener('change', (e) => {
        activeModel = e.target.value;
        saveSettingsToStorage();
    });

    // API Key input change
    apiKeyInput.addEventListener('input', () => {
        saveSettingsToStorage();
        checkApiKeyLoadStatus();
    });

    // Session ID change
    sessionIdInput.addEventListener('change', (e) => {
        const newSession = e.target.value.trim() || 'default_user';
        if (newSession !== activeSession) {
            activeSession = newSession;
            activeSessionName.textContent = activeSession;
            saveSettingsToStorage();
            syncSessionData();
            showToast(`Switched user profile to: "${activeSession}"`);
        }
    });

    // Memory Limit slider change
    memoryLimitSlider.addEventListener('input', (e) => {
        activeLimit = parseInt(e.target.value);
        memoryLimitVal.textContent = `${activeLimit} msgs`;
    });
    memoryLimitSlider.addEventListener('change', () => {
        saveSettingsToStorage();
    });

    // Clear Chat button click
    clearChatBtn.addEventListener('click', async () => {
        if (confirm('Are you sure you want to clear the active conversation history? This will not delete SQLite memories.')) {
            const success = await apiClearChatHistory();
            if (success) {
                clearChatUI();
                showToast('Short-term conversation history cleared!');
                syncSessionData();
            }
        }
    });

    // Clear SQLite DB button click
    clearDbBtn.addEventListener('click', async () => {
        if (confirm('CAUTION: Are you sure you want to clear all stored long-term memories for this user from SQLite?')) {
            const success = await apiClearAllMemories();
            if (success) {
                showToast('SQLite long-term memory cleared!');
                syncSessionData();
            }
        }
    });

    // Textarea input sizing & key submit
    chatTextarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    chatTextarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Send Message button click
    sendMsgBtn.addEventListener('click', sendMessage);

    // Save Manual Memory click
    saveManualMemoryBtn.addEventListener('click', async () => {
        const content = manualMemoryInput.value.trim();
        if (content) {
            const success = await apiSaveManualMemory(content);
            if (success) {
                manualMemoryInput.value = '';
                showToast('Manual memory saved successfully!');
                syncSessionData();
            }
        }
    });
}

function initTabButtons() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active classes
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked button
            btn.classList.add('active');
            
            // Activate corresponding tab content
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// ==========================================================================
// 3. UI Helpers
// ==========================================================================
function updateModelOptions() {
    modelSelect.innerHTML = '';
    const models = PROVIDER_MODELS[activeProvider] || [];
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.value;
        option.textContent = model.label;
        if (model.value === activeModel) {
            option.selected = true;
        }
        modelSelect.appendChild(option);
    });
    // If selected model is not in the new provider list, pick the first one
    if (models.length > 0 && !models.some(m => m.value === activeModel)) {
        activeModel = models[0].value;
        modelSelect.value = activeModel;
    }
}

function updateApiKeyInputPlaceholder() {
    const isServerConfigured = serverConfig[activeProvider];
    if (isServerConfigured) {
        apiKeyInput.placeholder = 'Configured in server .env';
    } else {
        apiKeyInput.placeholder = activeProvider === 'google' ? 'Enter Gemini API Key...' : 'Enter Groq API Key...';
    }
}

function checkApiKeyLoadStatus() {
    const key = apiKeyInput.value.trim();
    const isServerConfigured = serverConfig[activeProvider];
    if (key || isServerConfigured) {
        apiKeyStatus.className = 'status-indicator status-ok';
        apiKeyStatus.innerHTML = `
            <i class="fa-solid fa-circle-check status-ok-icon"></i>
            <span class="status-ok-text">${key ? 'Key loaded successfully.' : 'Using API key from server .env.'}</span>
        `;
        chatTextarea.disabled = !isBackendOnline;
        sendMsgBtn.disabled = !isBackendOnline;
    } else {
        apiKeyStatus.className = 'status-indicator';
        apiKeyStatus.innerHTML = `
            <i class="fa-solid fa-circle-exclamation status-warn-icon"></i>
            <span class="status-warn-text">Key required to chat.</span>
        `;
        chatTextarea.disabled = true;
        sendMsgBtn.disabled = true;
    }
}

function updateBackendStatus(online) {
    isBackendOnline = online;
    if (online) {
        backendStatus.innerHTML = `
            <i class="fa-solid fa-circle-dot status-online"></i>
            <span class="status-text">Backend API Online</span>
        `;
        checkApiKeyLoadStatus(); // Re-enable chat fields if key loaded
    } else {
        backendStatus.innerHTML = `
            <i class="fa-solid fa-circle-dot status-offline"></i>
            <span class="status-text">Backend API Offline (${BASE_URL})</span>
        `;
        chatTextarea.disabled = true;
        sendMsgBtn.disabled = true;
    }
}

function loadSettingsFromStorage() {
    activeSession = localStorage.getItem('chat_session_id') || 'default_user';
    activeProvider = localStorage.getItem('chat_provider') || 'google';
    activeModel = localStorage.getItem('chat_model') || 'gemini-2.5-flash';
    activeLimit = parseInt(localStorage.getItem('chat_limit') || '6');
    const key = localStorage.getItem(`api_key_${activeProvider}`) || '';

    sessionIdInput.value = activeSession;
    activeSessionName.textContent = activeSession;
    providerSelect.value = activeProvider;
    apiKeyInput.value = key;
    memoryLimitSlider.value = activeLimit;
    memoryLimitVal.textContent = `${activeLimit} msgs`;

    updateApiKeyInputPlaceholder();
}

function saveSettingsToStorage() {
    localStorage.setItem('chat_session_id', activeSession);
    localStorage.setItem('chat_provider', activeProvider);
    localStorage.setItem('chat_model', activeModel);
    localStorage.setItem('chat_limit', activeLimit);
    localStorage.setItem(`api_key_${activeProvider}`, apiKeyInput.value.trim());
}

function renderMessage(role, content) {
    // Hide welcome card if present
    welcomeMessageCard.style.display = 'none';

    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ${role === 'user' ? 'user' : 'assistant'}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = content;

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    chatMessagesContainer.appendChild(wrapper);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
}

function clearChatUI() {
    // Remove all message-wrappers
    const wrappers = chatMessagesContainer.querySelectorAll('.message-wrapper');
    wrappers.forEach(w => w.remove());
    welcomeMessageCard.style.display = 'block';
}

function showToast(message) {
    const toast = document.createElement('div');
    toast.style.position = 'fixed';
    toast.style.bottom = '2rem';
    toast.style.right = '2rem';
    toast.style.background = 'rgba(30, 41, 59, 0.95)';
    toast.style.border = '1px solid rgba(255, 255, 255, 0.1)';
    toast.style.borderLeft = '4px solid var(--primary)';
    toast.style.color = '#fff';
    toast.style.padding = '0.75rem 1.25rem';
    toast.style.borderRadius = '8px';
    toast.style.fontSize = '0.85rem';
    toast.style.fontWeight = '600';
    toast.style.zIndex = '1000';
    toast.style.boxShadow = 'var(--shadow-md)';
    toast.style.animation = 'slideIn 0.2s ease-out';
    
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s ease-out forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function renderMemories(memories) {
    sqliteMemoriesContainer.innerHTML = '';
    if (!memories || memories.length === 0) {
        sqliteMemoriesContainer.innerHTML = '<div class="empty-state">No long-term memories found. Try chatting!</div>';
        return;
    }

    memories.forEach(mem => {
        const card = document.createElement('div');
        card.className = 'memory-card';

        const info = document.createElement('div');
        info.className = 'memory-info';
        
        const content = document.createElement('div');
        content.className = 'memory-content';
        content.textContent = mem.content;

        const ts = document.createElement('div');
        ts.className = 'memory-timestamp';
        ts.textContent = `ID: ${mem.id} | Saved: ${mem.created_at}`;

        info.appendChild(content);
        info.appendChild(ts);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-mem-btn';
        deleteBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
        deleteBtn.title = 'Delete memory';
        deleteBtn.addEventListener('click', async () => {
            if (confirm('Delete this long-term memory fact?')) {
                const success = await apiDeleteMemory(mem.id);
                if (success) {
                    showToast('Memory deleted!');
                    syncSessionData();
                }
            }
        });

        card.appendChild(info);
        card.appendChild(deleteBtn);
        sqliteMemoriesContainer.appendChild(card);
    });
}

function renderActiveMessagesList(messages) {
    activeMessagesContainer.innerHTML = '';
    activeMsgsCount.textContent = messages.length;

    if (!messages || messages.length === 0) {
        activeMessagesContainer.innerHTML = '<div class="empty-state">Short-term chat history is empty.</div>';
        return;
    }

    messages.forEach((msg, idx) => {
        const card = document.createElement('div');
        card.className = `active-msg-card ${msg.role === 'user' ? 'user' : 'assistant'}`;

        const header = document.createElement('div');
        header.className = 'active-msg-header';
        
        const label = document.createElement('span');
        label.textContent = `[${idx}] ${msg.role === 'user' ? 'HumanMessage' : 'AIMessage'}`;

        const idSpan = document.createElement('span');
        idSpan.style.opacity = '0.6';
        idSpan.textContent = `ID: ${msg.id ? msg.id.substring(0, 8) : 'N/A'}`;

        header.appendChild(label);
        header.appendChild(idSpan);

        const content = document.createElement('div');
        content.className = 'active-msg-content';
        content.textContent = msg.content;

        card.appendChild(header);
        card.appendChild(content);
        activeMessagesContainer.appendChild(card);
    });
}

// ==========================================================================
// 4. API Request Calls
// ==========================================================================
async function checkBackendHealth() {
    try {
        const res = await fetch(`${BASE_URL}/healthz`, { method: 'GET' });
        if (res.status === 200) {
            updateBackendStatus(true);
            await syncConfigStatus();
        } else {
            updateBackendStatus(false);
        }
    } catch {
        updateBackendStatus(false);
    }
}

async function syncConfigStatus() {
    try {
        const res = await fetch(`${BASE_URL}/api/v1/config`);
        if (res.ok) {
            serverConfig = await res.json();
            checkApiKeyLoadStatus();
            updateApiKeyInputPlaceholder();
        }
    } catch (e) {
        console.error('Failed to sync config status:', e);
    }
}

async function syncSessionData() {
    if (!isBackendOnline) return;

    // 1. Fetch memories
    try {
        const res = await fetch(`${BASE_URL}/api/v1/memories/${activeSession}`);
        if (res.ok) {
            const data = await res.json();
            renderMemories(data);
        }
    } catch (e) {
        console.error('Failed to sync memories:', e);
    }

    // 2. Fetch short-term active messages
    try {
        const res = await fetch(`${BASE_URL}/api/v1/chat/${activeSession}`);
        if (res.ok) {
            const data = await res.json();
            renderActiveMessagesList(data);
            
            // Sync current chat log UI as well
            clearChatUI();
            data.forEach(msg => {
                renderMessage(msg.role, msg.content);
            });
        }
    } catch (e) {
        console.error('Failed to sync chat history:', e);
    }
}

async function sendMessage() {
    const text = chatTextarea.value.trim();
    const apiKey = apiKeyInput.value.trim();
    const isServerConfigured = serverConfig[activeProvider];
    if (!text || (!apiKey && !isServerConfigured) || !isBackendOnline) return;

    // Clear textbox & disable inputs during thinking state
    chatTextarea.value = '';
    chatTextarea.style.height = '48px';
    chatTextarea.disabled = true;
    sendMsgBtn.disabled = true;

    // Render User message instantly
    renderMessage('user', text);

    // Create spinner bubble for thinking state
    const spinnerWrapper = document.createElement('div');
    spinnerWrapper.className = 'message-wrapper assistant';
    spinnerWrapper.id = 'chat-thinking-spinner';
    spinnerWrapper.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="message-bubble" style="opacity: 0.6;">
            <i class="fa-solid fa-circle-notch fa-spin"></i> Agent is thinking...
        </div>
    `;
    chatMessagesContainer.appendChild(spinnerWrapper);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;

    // Request payload
    const payload = {
        user_id: activeSession,
        message: text,
        provider: activeProvider,
        model: activeModel,
        api_key: apiKey,
        limit: activeLimit
    };

    try {
        const res = await fetch(`${BASE_URL}/api/v1/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Token': 'streamlit-frontend-client'
            },
            body: JSON.stringify(payload)
        });

        // Remove spinner
        const spinner = document.getElementById('chat-thinking-spinner');
        if (spinner) spinner.remove();

        if (res.ok) {
            const data = await res.json();
            
            // Render assistant reply
            renderMessage('assistant', data.response);
            
            // Sync all lists from the response
            renderActiveMessagesList(data.active_messages);
            
            // Re-fetch SQLite memories since node analyzer might have extracted something new
            const memRes = await fetch(`${BASE_URL}/api/v1/memories/${activeSession}`);
            if (memRes.ok) {
                renderMemories(await memRes.json());
            }
        } else {
            const err = await res.json();
            alert(`Error: ${err.detail || 'Could not send message.'}`);
        }
    } catch (e) {
        console.error(e);
        const spinner = document.getElementById('chat-thinking-spinner');
        if (spinner) spinner.remove();
        alert('Failed to connect to backend server.');
    } finally {
        chatTextarea.disabled = false;
        sendMsgBtn.disabled = false;
        chatTextarea.focus();
    }
}

async function apiClearChatHistory() {
    try {
        const res = await fetch(`${BASE_URL}/api/v1/chat/clear/${activeSession}`, {
            method: 'POST'
        });
        if (res.ok) {
            const data = await res.json();
            return data.success;
        }
    } catch (e) {
        console.error('Failed to clear chat on backend:', e);
    }
    return false;
}

async function apiClearAllMemories() {
    try {
        const res = await fetch(`${BASE_URL}/api/v1/memories/${activeSession}`, {
            method: 'DELETE'
        });
        if (res.ok) {
            const data = await res.json();
            return data.success;
        }
    } catch (e) {
        console.error('Failed to clear memories on backend:', e);
    }
    return false;
}

async function apiDeleteMemory(memoryId) {
    try {
        const res = await fetch(`${BASE_URL}/api/v1/memories/${activeSession}/${memoryId}`, {
            method: 'DELETE'
        });
        if (res.ok) {
            const data = await res.json();
            return data.success;
        }
    } catch (e) {
        console.error('Failed to delete memory:', e);
    }
    return false;
}

async function apiSaveManualMemory(content) {
    try {
        const res = await fetch(`${BASE_URL}/api/v1/memories/${activeSession}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content })
        });
        if (res.ok) {
            const data = await res.json();
            return data.success;
        }
    } catch (e) {
        console.error('Failed to save manual memory:', e);
    }
    return false;
}
