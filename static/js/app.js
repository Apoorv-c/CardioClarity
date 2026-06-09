/* ══════════════════════════════════════════════════════════
   app.js — Main SPA: routing, tab switching, toast, init
══════════════════════════════════════════════════════════ */

// ── Tab Management ────────────────────────────────────────

function switchTab(tabId) {
  // Update nav
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  // Show/hide content
  document.querySelectorAll('.tab-content').forEach(tc => {
    tc.classList.remove('active');
    tc.classList.add('hidden');
  });
  const active = document.getElementById('tab-' + tabId);
  if (active) {
    active.classList.remove('hidden');
    active.classList.add('active');
  }

  // Lazy-load tab data
  if (tabId === 'history') loadHistory();
  if (tabId === 'insights') loadModelMetrics();
  if (tabId === 'whatif') updateWhatIfUI();
}

// ── Toast ─────────────────────────────────────────────────

let _toastTimer = null;
function showToast(msg, type = '') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className   = 'toast' + (type ? ' ' + type : '');
  el.classList.remove('hidden');
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.add('hidden'), 3500);
}

// ── Loading Overlay ───────────────────────────────────────

function showLoading(text = 'Please wait…') {
  document.getElementById('loading-text').textContent = text;
  document.getElementById('loading-overlay').classList.remove('hidden');
}
function hideLoading() {
  document.getElementById('loading-overlay').classList.add('hidden');
}

// ── User Menu ─────────────────────────────────────────────

function toggleUserMenu() {
  document.getElementById('user-menu').classList.toggle('hidden');
}
document.addEventListener('click', e => {
  const navUser = document.querySelector('.nav-user');
  const menu    = document.getElementById('user-menu');
  if (navUser && !navUser.contains(e.target)) {
    menu?.classList.add('hidden');
  }
});

// ── Init on Auth Success ──────────────────────────────────

function onAuthSuccess() {
  const user = getUser();

  // Update nav
  document.getElementById('nav-username').textContent = user.name || user.username || 'User';
  document.getElementById('nav-avatar').textContent   = (user.name || user.username || 'U')[0].toUpperCase();

  // Show dashboard
  document.getElementById('auth-screen').classList.add('hidden');
  document.getElementById('dashboard').classList.remove('hidden');

  // Force predict tab
  switchTab('predict');

  // Set default role + attach live-update listeners
  setRole('doctor');        // setRole() calls _attachLiveListeners() internally

  // Init chatbot
  initChatbot();

  showToast(`Welcome back, ${user.name || user.username}! 👋`, 'success');
}

// ── On Page Load ──────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  if (isLoggedIn()) {
    // Verify token is still valid
    fetch('/api/auth/profile', { headers: authHeaders() })
      .then(r => {
        if (r.ok) return r.json();
        throw new Error('Token expired');
      })
      .then(data => {
        // Update stored user from server (may have refreshed name)
        setSession(getToken(), { name: data.name, username: data.username });
        onAuthSuccess();
      })
      .catch(() => {
        clearSession();
        // Show auth screen (default state)
      });
  }

  // Patient age → auto-update default HR estimate
  document.getElementById('pat-age')?.addEventListener('input', e => {
    const age = parseInt(e.target.value) || 45;
    const hrEl = document.getElementById('pat-thalach');
    if (hrEl && !hrEl._userEdited) {
      hrEl.value = Math.round((220 - age) * 0.85);
    }
  });
  document.getElementById('pat-thalach')?.addEventListener('change', e => {
    e.target._userEdited = true;
  });

  // Initialize what-if UI state
  updateWhatIfUI();
});

// ── Mobile nav tab shortcut ───────────────────────────────
// Expose switchTab globally (called inline from HTML)
window.switchTab = switchTab;
