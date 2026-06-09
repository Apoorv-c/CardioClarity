/* ══════════════════════════════════════════════════════════════════════════
   chatbot.js — CardioClarity Context-Aware Health Chatbot
   - Floating chat widget (bottom-right)
   - Sends messages to /api/chat with current prediction context
   - Stores conversation history in memory
══════════════════════════════════════════════════════════════════════════ */

const MAX_MESSAGES = 50;
let chatOpen = false;
let chatMessages = []; // { role: 'bot'|'user', text, time }

// ── Suggested quick questions ─────────────────────────────────────────────
const QUICK_QUESTIONS = [
  'What does my risk score mean?',
  'How can I lower my blood pressure?',
  'What is cholesterol and why does it matter?',
  'What is SHAP and why are features highlighted?',
  'Should I see a doctor?',
  'What lifestyle changes can help my heart?',
];

// ── Initialize Chatbot UI ─────────────────────────────────────────────────

function initChatbot() {
  // Only inject once
  if (document.getElementById('chatbot-widget')) return;

  const widget = document.createElement('div');
  widget.id = 'chatbot-widget';
  widget.innerHTML = `
    <!-- FAB Trigger -->
    <button class="chat-fab" id="chat-fab" onclick="toggleChat()" aria-label="Open health assistant">
      <span class="chat-fab-icon" id="chat-fab-icon">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round">
          <path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402 0-3.791 3.068-5.191 5.281-5.191 1.312 0 4.151.501 5.719 4.457 1.59-3.968 4.464-4.447 5.726-4.447 2.54 0 5.274 1.621 5.274 5.181 0 4.069-5.136 8.625-11 14.402z" fill="white" stroke="none"/>
        </svg>
      </span>
      <span class="chat-fab-close hidden" id="chat-fab-close">✕</span>
      <span class="chat-unread" id="chat-unread" style="display:none">1</span>
    </button>

    <!-- Chat Panel -->
    <div class="chat-panel hidden" id="chat-panel">
      <!-- Header -->
      <div class="chat-header">
        <div class="chat-header-left">
          <div class="chat-avatar-bot">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
              <path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402 0-3.791 3.068-5.191 5.281-5.191 1.312 0 4.151.501 5.719 4.457 1.59-3.968 4.464-4.447 5.726-4.447 2.54 0 5.274 1.621 5.274 5.181 0 4.069-5.136 8.625-11 14.402z"/>
            </svg>
          </div>
          <div>
            <div class="chat-bot-name">CardioBot</div>
            <div class="chat-bot-status">
              <span class="status-dot"></span> Heart Health Assistant
            </div>
          </div>
        </div>
        <div class="chat-header-right">
          <button class="chat-clear-btn" onclick="clearChat()" title="Clear conversation">🗑</button>
          <button class="chat-close-btn" onclick="toggleChat()">✕</button>
        </div>
      </div>

      <!-- Messages -->
      <div class="chat-messages" id="chat-messages"></div>

      <!-- Quick Suggestions -->
      <div class="chat-suggestions" id="chat-suggestions"></div>

      <!-- Input -->
      <div class="chat-input-area">
        <textarea
          id="chat-input"
          class="chat-textarea"
          placeholder="Ask me about your heart health…"
          rows="1"
          onkeydown="chatKeyDown(event)"
          oninput="autoResizeTextarea(this)"
        ></textarea>
        <button class="chat-send-btn" id="chat-send-btn" onclick="sendChatMessage()">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
      <div class="chat-disclaimer">⚠ For educational purposes only. Not medical advice.</div>
    </div>
  `;
  document.body.appendChild(widget);

  // Show welcome message
  addBotMessage(
    "👋 Hi! I'm **CardioBot**, your heart health assistant.\n\n" +
    "I can explain your risk results, answer questions about heart health, " +
    "and help you understand what the AI is telling you.\n\n" +
    "What would you like to know?",
    false
  );
  renderSuggestions(QUICK_QUESTIONS.slice(0, 4));

  // Show unread badge after 2s if chat is closed
  setTimeout(() => {
    if (!chatOpen) {
      const badge = document.getElementById('chat-unread');
      if (badge) badge.style.display = 'flex';
    }
  }, 2000);
}

// ── Toggle Chat ───────────────────────────────────────────────────────────

function toggleChat() {
  chatOpen = !chatOpen;
  const panel  = document.getElementById('chat-panel');
  const fabIcon  = document.getElementById('chat-fab-icon');
  const fabClose = document.getElementById('chat-fab-close');
  const badge    = document.getElementById('chat-unread');

  panel.classList.toggle('hidden', !chatOpen);
  fabIcon.classList.toggle('hidden', chatOpen);
  fabClose.classList.toggle('hidden', !chatOpen);
  if (badge) badge.style.display = 'none';

  if (chatOpen) {
    setTimeout(() => {
      const msgs = document.getElementById('chat-messages');
      if (msgs) msgs.scrollTop = msgs.scrollHeight;
      document.getElementById('chat-input')?.focus();
    }, 120);
  }
}

// ── Message Rendering ─────────────────────────────────────────────────────

function addBotMessage(text, showTyping = true) {
  const msgs = document.getElementById('chat-messages');
  if (!msgs) return;

  const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  chatMessages.push({ role: 'bot', text, time });

  const el = document.createElement('div');
  el.className = 'chat-msg chat-msg-bot';

  if (showTyping) {
    el.innerHTML = `
      <div class="chat-bubble chat-bubble-bot typing-indicator">
        <span></span><span></span><span></span>
      </div>`;
    msgs.appendChild(el);
    msgs.scrollTop = msgs.scrollHeight;

    setTimeout(() => {
      el.innerHTML = `
        <div class="chat-bubble chat-bubble-bot">${markdownToHtml(text)}</div>
        <div class="chat-time">${time}</div>`;
      msgs.scrollTop = msgs.scrollHeight;
    }, 700 + Math.min(text.length * 10, 1200));
  } else {
    el.innerHTML = `
      <div class="chat-bubble chat-bubble-bot">${markdownToHtml(text)}</div>
      <div class="chat-time">${time}</div>`;
    msgs.appendChild(el);
    msgs.scrollTop = msgs.scrollHeight;
  }
}

function addUserMessage(text) {
  const msgs = document.getElementById('chat-messages');
  if (!msgs) return;
  const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  chatMessages.push({ role: 'user', text, time });

  const el = document.createElement('div');
  el.className = 'chat-msg chat-msg-user';
  el.innerHTML = `
    <div class="chat-bubble chat-bubble-user">${escapeHtml(text)}</div>
    <div class="chat-time chat-time-user">${time}</div>`;
  msgs.appendChild(el);
  msgs.scrollTop = msgs.scrollHeight;
}

// ── Suggestions ───────────────────────────────────────────────────────────

function renderSuggestions(questions) {
  const el = document.getElementById('chat-suggestions');
  if (!el) return;
  el.innerHTML = questions.map(q =>
    `<button class="chat-suggestion-chip" onclick="sendQuick('${q.replace(/'/g, "\\'")}')">${q}</button>`
  ).join('');
}

function sendQuick(q) {
  document.getElementById('chat-suggestions').innerHTML = '';
  handleSend(q);
}

// ── Send Message ──────────────────────────────────────────────────────────

function chatKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
}

function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const text  = (input?.value || '').trim();
  if (!text) return;
  input.value = '';
  autoResizeTextarea(input);
  document.getElementById('chat-suggestions').innerHTML = '';
  handleSend(text);
}

async function handleSend(text) {
  addUserMessage(text);
  setSendDisabled(true);

  // Build context from current prediction
  const pred = window.lastPrediction || null;
  const ctx  = pred ? {
    tier_key:     pred.tier_key,
    tier_label:   pred.tier_label,
    probability:  pred.probability,
    patient_mode: pred.patient_mode,
    inputs:       pred.inputs,
    derived:      pred.derived,
    recommendations: pred.recommendations,
    shap_top: (pred.shap_contributions || []).slice(0, 5).map(c => c.feature),
  } : null;

  try {
    const res  = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Auth-Token': getToken() },
      body: JSON.stringify({ message: text, context: ctx }),
    });
    const data = await res.json();
    addBotMessage(data.reply || "I'm sorry, I couldn't process that. Please try again.");
    if (data.suggestions && data.suggestions.length > 0) {
      setTimeout(() => renderSuggestions(data.suggestions), 1500);
    }
  } catch (e) {
    addBotMessage("⚠️ I'm having trouble connecting right now. Make sure the server is running.");
  } finally {
    setSendDisabled(false);
    document.getElementById('chat-input')?.focus();
  }
}

function setSendDisabled(val) {
  const btn = document.getElementById('chat-send-btn');
  if (btn) btn.disabled = val;
}

// ── Clear Chat ────────────────────────────────────────────────────────────

function clearChat() {
  chatMessages = [];
  const msgs = document.getElementById('chat-messages');
  if (msgs) msgs.innerHTML = '';
  addBotMessage("Chat cleared! How can I help you?", false);
  renderSuggestions(QUICK_QUESTIONS.slice(0, 4));
}

// ── Utilities ─────────────────────────────────────────────────────────────

function autoResizeTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function escapeHtml(text) {
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function markdownToHtml(text) {
  // Bold (**text**)
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic (*text*)
  text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Bullet lists
  text = text.replace(/^- (.+)$/gm, '<li>$1</li>');
  text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  // Line breaks
  text = text.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
  return `<p>${text}</p>`;
}
