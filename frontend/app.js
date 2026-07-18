// ==========================================================================
// 1. Constants & State
// ==========================================================================
const BASE_URL = window.location.origin;

const PROVIDER_MODELS = {
  google: [
    { value: 'gemini-2.0-flash', label: 'gemini-2.0-flash' },
    { value: 'gemini-1.5-flash', label: 'gemini-1.5-flash' },
    { value: 'gemini-2.5-pro',   label: 'gemini-2.5-pro' }
  ],
  groq: [
    { value: 'llama-3.3-70b-versatile', label: 'llama-3.3-70b-versatile' },
    { value: 'llama-3.1-8b-instant',    label: 'llama-3.1-8b-instant' },
    { value: 'gemma2-9b-it',            label: 'gemma2-9b-it' }
  ]
};

let activeSession  = 'default_user';
let activeConversationId = 'default_conversation';
let activeProvider  = 'google';
let activeModel     = 'gemini-2.0-flash';
let activeLimit     = 6;
let isBackendOnline = false;
let serverConfig    = { google: false, groq: false };

// DOM refs
const providerSelect     = document.getElementById('provider-select');
const modelSelect        = document.getElementById('model-select');
const apiKeyInput        = document.getElementById('api-key-input');
const apiKeyStatus       = document.getElementById('api-key-status');
const sessionIdInput     = document.getElementById('session-id-input');
const memoryLimitSlider  = document.getElementById('memory-limit-slider');
const memoryLimitVal     = document.getElementById('memory-limit-val');
const clearChatBtn       = document.getElementById('clear-chat-btn');
const clearDbBtn         = document.getElementById('clear-db-btn');
const backendStatus      = document.getElementById('backend-status');
const activeSessionName  = document.getElementById('active-session-name');
const chatContainer      = document.getElementById('chat-messages-container');
const welcomeState       = document.getElementById('welcome-state');
const chatTextarea       = document.getElementById('chat-textarea');
const sendMsgBtn         = document.getElementById('send-msg-btn');
const sqliteMemories     = document.getElementById('sqlite-memories-container');
const manualMemInput     = document.getElementById('manual-memory-input');
const saveMemBtn         = document.getElementById('save-manual-memory-btn');
const activeMsgsCount    = document.getElementById('active-msgs-count');
const activeMsgsList     = document.getElementById('active-messages-container');
const topbarModelName    = document.getElementById('topbar-model-name');
const featureCards       = document.getElementById('feature-cards');

// ==========================================================================
// 2. Init
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  updateModelOptions();
  initListeners();
  checkBackendHealth();
  syncSessionData();
});

// ==========================================================================
// 3. Event Listeners
// ==========================================================================
function initListeners() {
  // Provider change
  providerSelect.addEventListener('change', e => {
    activeProvider = e.target.value;
    updateModelOptions();
    saveSettings();
    updateApiKeyPlaceholder();
    checkApiKeyStatus();
  });

  // Model change
  modelSelect.addEventListener('change', e => {
    activeModel = e.target.value;
    topbarModelName.textContent = activeModel;
    saveSettings();
  });

  // API Key
  apiKeyInput.addEventListener('input', () => { saveSettings(); checkApiKeyStatus(); });

  // Session ID
  sessionIdInput.addEventListener('change', e => {
    const v = e.target.value.trim() || 'default_user';
    if (v !== activeSession) {
      activeSession = v;
      activeSessionName.textContent = activeSession;
      activeConversationId = 'default_conversation';
      saveSettings();
      syncSessionData();
      loadConversations();
      showToast(`Switched to user session: "${activeSession}"`);
    }
  });

  // Limit slider
  memoryLimitSlider.addEventListener('input', e => {
    activeLimit = parseInt(e.target.value);
    memoryLimitVal.textContent = activeLimit;
  });
  memoryLimitSlider.addEventListener('change', () => saveSettings());

  // Clear Chat
  clearChatBtn.addEventListener('click', async () => {
    if (!confirm('Clear the active conversation history?')) return;
    if (await apiClearChat()) {
      clearChatUI();
      showToast('Chat cleared!');
      syncSessionData();
    }
  });

  // Clear DB
  clearDbBtn.addEventListener('click', async () => {
    if (!confirm('Delete all long-term memories?')) return;
    if (await apiClearMemories()) {
      showToast('Memories cleared!');
      syncSessionData();
    }
  });

  // Textarea auto-resize & Enter submit
  chatTextarea.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
  });
  chatTextarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  // Send button
  sendMsgBtn.addEventListener('click', sendMessage);

  // Save manual memory
  saveMemBtn.addEventListener('click', async () => {
    const text = manualMemInput.value.trim();
    if (!text) return;
    if (await apiSaveMemory(text)) {
      manualMemInput.value = '';
      showToast('Memory saved!');
      syncSessionData();
    }
  });

  // Sidebar nav
  document.querySelectorAll('.nav-item[data-view]').forEach(item => {
    item.addEventListener('click', () => switchView(item.dataset.view));
  });

  // Feature cards
  document.querySelectorAll('.feature-card[data-view]').forEach(card => {
    card.addEventListener('click', () => switchView(card.dataset.view));
  });

  // Quick chips
  document.querySelectorAll('.quick-chip[data-prompt]').forEach(chip => {
    chip.addEventListener('click', () => {
      chatTextarea.value = chip.dataset.prompt;
      chatTextarea.focus();
    });
  });

  // Settings toggle
  const settingsToggle = document.getElementById('sidebar-settings-toggle');
  const settingsPanel  = document.getElementById('sidebar-settings-panel');
  settingsToggle.addEventListener('click', () => {
    settingsPanel.classList.toggle('open');
    settingsToggle.classList.toggle('active');
  });

  // Topbar config btn
  document.getElementById('topbar-config-btn').addEventListener('click', () => {
    settingsPanel.classList.toggle('open');
    settingsToggle.classList.toggle('active');
  });

  // New Chat btn
  document.getElementById('new-chat-btn').addEventListener('click', async () => {
    try {
      const res = await fetch(`${BASE_URL}/api/v1/conversations/${activeSession}?title=New+Conversation`, { method: 'POST' });
      if (res.ok) {
        const newConv = await res.json();
        activeConversationId = newConv.id;
        saveSettings();
        clearChatUI();
        switchView('chat');
        showToast('New conversation started!');
        await loadConversations();
        await syncSessionData();
      }
    } catch (err) {
      console.error(err);
    }
  });
}

// ==========================================================================
// 4. View Switching
// ==========================================================================
function switchView(view) {
  // Nav items
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const navTarget = document.querySelector(`.nav-item[data-view="${view}"]`);
  if (navTarget) navTarget.classList.add('active');

  // Panels
  document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
  const panelId = `view-${view}`;
  const panel = document.getElementById(panelId);
  if (panel) panel.classList.add('active');
}

// ==========================================================================
// 5. UI Helpers
// ==========================================================================
function updateModelOptions() {
  modelSelect.innerHTML = '';
  const models = PROVIDER_MODELS[activeProvider] || [];
  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.value;
    opt.textContent = m.label;
    if (m.value === activeModel) opt.selected = true;
    modelSelect.appendChild(opt);
  });
  if (!models.some(m => m.value === activeModel) && models.length) {
    activeModel = models[0].value;
    modelSelect.value = activeModel;
  }
  topbarModelName.textContent = activeModel;
}

function updateApiKeyPlaceholder() {
  const configured = serverConfig[activeProvider];
  apiKeyInput.placeholder = configured ? 'Configured in server .env' : `Enter ${activeProvider === 'google' ? 'Gemini' : 'Groq'} API Key...`;
}

function checkApiKeyStatus() {
  const key = apiKeyInput.value.trim();
  const configured = serverConfig[activeProvider];
  if (key || configured) {
    apiKeyStatus.className = 'status-card ok';
    apiKeyStatus.innerHTML = `<i class="fa-solid fa-circle-check"></i><span>${key ? 'Key loaded.' : 'Using server .env key.'}</span>`;
    chatTextarea.disabled = !isBackendOnline;
    sendMsgBtn.disabled = !isBackendOnline;
  } else {
    apiKeyStatus.className = 'status-card warn';
    apiKeyStatus.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i><span>API key required.</span>`;
    chatTextarea.disabled = true;
    sendMsgBtn.disabled = true;
  }
}

function updateBackendUI(online) {
  isBackendOnline = online;
  const dot = backendStatus.querySelector('.pulse-dot');
  const label = backendStatus.querySelector('span:last-child');
  if (online) {
    dot.className = 'pulse-dot online';
    label.textContent = 'Backend Online';
    checkApiKeyStatus();
  } else {
    dot.className = 'pulse-dot offline';
    label.textContent = 'Backend Offline';
    chatTextarea.disabled = true;
    sendMsgBtn.disabled = true;
  }
}

function loadSettings() {
  activeSession  = localStorage.getItem('s_session') || 'default_user';
  activeConversationId = localStorage.getItem('s_conv_id') || 'default_conversation';
  activeProvider  = localStorage.getItem('s_provider') || 'google';
  activeModel     = localStorage.getItem('s_model') || 'gemini-2.0-flash';
  activeLimit     = parseInt(localStorage.getItem('s_limit') || '6');
  const key       = localStorage.getItem(`s_key_${activeProvider}`) || '';

  sessionIdInput.value = activeSession;
  activeSessionName.textContent = activeSession;
  providerSelect.value = activeProvider;
  apiKeyInput.value = key;
  memoryLimitSlider.value = activeLimit;
  memoryLimitVal.textContent = activeLimit;
  updateApiKeyPlaceholder();
}

function saveSettings() {
  localStorage.setItem('s_session', activeSession);
  localStorage.setItem('s_conv_id', activeConversationId);
  localStorage.setItem('s_provider', activeProvider);
  localStorage.setItem('s_model', activeModel);
  localStorage.setItem('s_limit', activeLimit);
  localStorage.setItem(`s_key_${activeProvider}`, apiKeyInput.value.trim());
}

function addMessage(role, content) {
  welcomeState.style.display = 'none';
  featureCards.style.display = 'none';
  chatContainer.classList.add('visible');

  const msg = document.createElement('div');
  msg.className = `msg ${role}`;

  const av = document.createElement('div');
  av.className = 'msg-avatar';
  av.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

  const bub = document.createElement('div');
  bub.className = 'msg-bubble';
  bub.textContent = content;

  msg.appendChild(av);
  msg.appendChild(bub);
  chatContainer.appendChild(msg);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function clearChatUI() {
  chatContainer.innerHTML = '';
  chatContainer.classList.remove('visible');
  welcomeState.style.display = '';
  featureCards.style.display = '';
}

function showToast(text) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = text;
  document.body.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = '0.3s'; setTimeout(() => t.remove(), 300); }, 3000);
}

function renderMemories(mems) {
  sqliteMemories.innerHTML = '';
  if (!mems || !mems.length) {
    sqliteMemories.innerHTML = '<div class="empty-state"><i class="fa-regular fa-face-smile-wink"></i> No memories yet. Start chatting!</div>';
    return;
  }
  mems.forEach(m => {
    const card = document.createElement('div');
    card.className = 'mem-card';
    card.innerHTML = `
      <div>
        <div class="mem-text">${m.content}</div>
        <div class="mem-ts">ID: ${m.id} · ${m.created_at}</div>
      </div>
    `;
    const del = document.createElement('button');
    del.className = 'mem-del';
    del.innerHTML = '<i class="fa-solid fa-xmark"></i>';
    del.onclick = async () => {
      if (confirm('Delete this memory?') && await apiDeleteMemory(m.id)) {
        showToast('Deleted.');
        syncSessionData();
      }
    };
    card.appendChild(del);
    sqliteMemories.appendChild(card);
  });
}

function renderActiveMsgs(msgs) {
  activeMsgsList.innerHTML = '';
  activeMsgsCount.textContent = msgs.length;
  if (!msgs.length) {
    activeMsgsList.innerHTML = '<div class="empty-state"><i class="fa-regular fa-comments"></i> No active messages.</div>';
    return;
  }
  msgs.forEach((m, i) => {
    const card = document.createElement('div');
    card.className = `amsg ${m.role === 'user' ? 'user' : 'assistant'}`;
    card.innerHTML = `
      <div class="amsg-header">
        <span>[${i}] ${m.role === 'user' ? 'HumanMessage' : 'AIMessage'}</span>
        <span style="opacity:.5">${m.id ? m.id.substring(0,8) : ''}</span>
      </div>
      <div class="amsg-content">${m.content}</div>
    `;
    activeMsgsList.appendChild(card);
  });
}

async function loadConversations() {
  if (!isBackendOnline) return;
  try {
    const r = await fetch(`${BASE_URL}/api/v1/conversations/${activeSession}`);
    if (r.ok) {
      const convs = await r.json();
      renderConversationsList(convs);
    }
  } catch(e) { console.error(e); }
}

function renderConversationsList(convs) {
  const container = document.getElementById('conversations-list-container');
  container.innerHTML = '';
  if (!convs || !convs.length) {
    container.innerHTML = '<div style="font-size:0.72rem; color:var(--text-dim); text-align:center; padding:10px 0;">No conversations</div>';
    return;
  }
  convs.forEach(c => {
    const item = document.createElement('div');
    item.className = `conversation-item ${c.id === activeConversationId ? 'active' : ''}`;
    item.setAttribute('data-id', c.id);
    
    // Bubble icon
    const icon = document.createElement('i');
    icon.className = 'fa-regular fa-comment';
    item.appendChild(icon);
    
    // Title text
    const title = document.createElement('span');
    title.className = 'conversation-title';
    title.textContent = c.title;
    item.appendChild(title);
    
    // Delete button
    const del = document.createElement('button');
    del.className = 'conversation-delete';
    del.innerHTML = '<i class="fa-regular fa-trash-can"></i>';
    del.title = 'Delete Conversation';
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete conversation "${c.title}"?`)) return;
      try {
        const res = await fetch(`${BASE_URL}/api/v1/conversations/${activeSession}/${c.id}`, { method: 'DELETE' });
        if (res.ok) {
          showToast('Conversation deleted.');
          if (activeConversationId === c.id) {
            const remaining = convs.filter(x => x.id !== c.id);
            activeConversationId = remaining.length ? remaining[0].id : 'default_conversation';
            localStorage.setItem('s_conv_id', activeConversationId);
          }
          await loadConversations();
          await syncSessionData();
        }
      } catch (err) {
        console.error(err);
      }
    };
    item.appendChild(del);
    
    // Switch on item click
    item.onclick = async () => {
      if (activeConversationId !== c.id) {
        activeConversationId = c.id;
        localStorage.setItem('s_conv_id', activeConversationId);
        document.querySelectorAll('.conversation-item').forEach(el => el.classList.remove('active'));
        item.classList.add('active');
        switchView('chat');
        await syncSessionData();
      }
    };
    
    container.appendChild(item);
  });
}

// ==========================================================================
// 6. API Calls
// ==========================================================================
async function checkBackendHealth() {
  try {
    const r = await fetch(`${BASE_URL}/healthz`);
    if (r.ok) { 
      updateBackendUI(true); 
      await syncConfig(); 
      await loadConversations();
    }
    else updateBackendUI(false);
  } catch { updateBackendUI(false); }
}

async function syncConfig() {
  try {
    const r = await fetch(`${BASE_URL}/api/v1/config`);
    if (r.ok) { serverConfig = await r.json(); checkApiKeyStatus(); updateApiKeyPlaceholder(); }
  } catch(e) { console.error(e); }
}

async function syncSessionData() {
  if (!isBackendOnline) return;
  try {
    const r = await fetch(`${BASE_URL}/api/v1/memories/${activeSession}`);
    if (r.ok) renderMemories(await r.json());
  } catch(e) { console.error(e); }

  try {
    const r = await fetch(`${BASE_URL}/api/v1/chat/${activeSession}?conversation_id=${activeConversationId}`);
    if (r.ok) {
      const msgs = await r.json();
      renderActiveMsgs(msgs);
      clearChatUI();
      if (msgs.length) msgs.forEach(m => addMessage(m.role, m.content));
    }
  } catch(e) { console.error(e); }
}

async function sendMessage() {
  const text = chatTextarea.value.trim();
  const apiKey = apiKeyInput.value.trim();
  const configured = serverConfig[activeProvider];
  if (!text || (!apiKey && !configured) || !isBackendOnline) return;

  chatTextarea.value = '';
  chatTextarea.style.height = 'auto';
  chatTextarea.disabled = true;
  sendMsgBtn.disabled = true;

  addMessage('user', text);

  // Create streaming bubble
  const wrapper = document.createElement('div');
  wrapper.className = 'msg assistant';
  const av = document.createElement('div');
  av.className = 'msg-avatar';
  av.innerHTML = '<i class="fa-solid fa-robot"></i>';
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin" style="opacity:.4"></i>';
  wrapper.appendChild(av);
  wrapper.appendChild(bubble);
  chatContainer.appendChild(wrapper);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  const payload = {
    user_id: activeSession,
    conversation_id: activeConversationId,
    message: text,
    provider: activeProvider,
    model: activeModel,
    api_key: apiKey,
    limit: activeLimit
  };

  try {
    const res = await fetch(`${BASE_URL}/api/v1/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Token': 'streamlit-frontend-client' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      bubble.textContent = `Error: ${err.detail || 'Request failed.'}`;
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '', streamed = '', spinnerGone = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const ds = line.slice(6).trim();
        if (!ds) continue;
        const ev = JSON.parse(ds);

        if (ev.type === 'chunk') {
          if (!spinnerGone) { bubble.innerHTML = ''; spinnerGone = true; }
          streamed += ev.content;
          bubble.textContent = streamed;
          chatContainer.scrollTop = chatContainer.scrollHeight;
        } else if (ev.type === 'done') {
          renderActiveMsgs(ev.active_messages);
          const memR = await fetch(`${BASE_URL}/api/v1/memories/${activeSession}`);
          if (memR.ok) renderMemories(await memR.json());
          await loadConversations();
        } else if (ev.type === 'error') {
          bubble.textContent = `Error: ${ev.content}`;
        }
      }
    }
  } catch(e) {
    console.error(e);
    bubble.textContent = bubble.innerHTML.includes('fa-spin') ? 'Connection failed.' : bubble.textContent + '\n[Connection lost]';
  } finally {
    chatTextarea.disabled = false;
    sendMsgBtn.disabled = false;
    chatTextarea.focus();
  }
}

async function apiClearChat() {
  try { const r = await fetch(`${BASE_URL}/api/v1/chat/clear/${activeSession}?conversation_id=${activeConversationId}`, { method:'POST' }); return r.ok && (await r.json()).success; }
  catch { return false; }
}
async function apiClearMemories() {
  try { const r = await fetch(`${BASE_URL}/api/v1/memories/${activeSession}`, { method:'DELETE' }); return r.ok && (await r.json()).success; }
  catch { return false; }
}
async function apiDeleteMemory(id) {
  try { const r = await fetch(`${BASE_URL}/api/v1/memories/${activeSession}/${id}`, { method:'DELETE' }); return r.ok && (await r.json()).success; }
  catch { return false; }
}
async function apiSaveMemory(content) {
  try { const r = await fetch(`${BASE_URL}/api/v1/memories/${activeSession}`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({content}) }); return r.ok && (await r.json()).success; }
  catch { return false; }
}
