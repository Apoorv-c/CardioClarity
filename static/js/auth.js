/* ══════════════════════════════════════════════════════════
   auth.js — Login / Register / Session Management
══════════════════════════════════════════════════════════ */

const AUTH_TOKEN_KEY = 'cc_token';
const AUTH_USER_KEY  = 'cc_user';

// ── Helpers ──────────────────────────────────────────────

function getToken()    { return localStorage.getItem(AUTH_TOKEN_KEY) || ''; }
function getUser()     { try { return JSON.parse(localStorage.getItem(AUTH_USER_KEY)) || {}; } catch { return {}; } }
function setSession(t, u) {
  localStorage.setItem(AUTH_TOKEN_KEY, t);
  localStorage.setItem(AUTH_USER_KEY, JSON.stringify(u));
}
function clearSession() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_USER_KEY);
}
function isLoggedIn() { return !!getToken(); }

// Attach token to all API calls
function authHeaders() {
  return { 'Content-Type': 'application/json', 'X-Auth-Token': getToken() };
}

// ── Tab Switch ───────────────────────────────────────────

function authSwitchTab(tab) {
  document.getElementById('login-form').classList.toggle('hidden', tab !== 'login');
  document.getElementById('register-form').classList.toggle('hidden', tab !== 'register');
  document.getElementById('tab-login').classList.toggle('active', tab === 'login');
  document.getElementById('tab-register').classList.toggle('active', tab === 'register');
  document.getElementById('login-error').classList.add('hidden');
  document.getElementById('register-error').classList.add('hidden');
}

// ── Login ─────────────────────────────────────────────────

async function handleLogin(event) {
  event.preventDefault();
  const btn = document.getElementById('login-btn');
  const errEl = document.getElementById('login-error');
  errEl.classList.add('hidden');

  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value;

  setButtonLoading(btn, true);
  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();
    if (!res.ok) {
      showFormError(errEl, data.error || 'Login failed');
      return;
    }
    setSession(data.token, { name: data.name, username: data.username });
    onAuthSuccess();
  } catch (e) {
    showFormError(errEl, 'Network error — is the server running?');
  } finally {
    setButtonLoading(btn, false);
  }
}

// ── Register ──────────────────────────────────────────────

async function handleRegister(event) {
  event.preventDefault();
  const btn = document.getElementById('register-btn');
  const errEl = document.getElementById('register-error');
  errEl.classList.add('hidden');

  const name     = document.getElementById('reg-name').value.trim();
  const username = document.getElementById('reg-username').value.trim();
  const password = document.getElementById('reg-password').value;

  setButtonLoading(btn, true);
  try {
    const res = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, username, password }),
    });
    const data = await res.json();
    if (!res.ok) {
      showFormError(errEl, data.error || 'Registration failed');
      return;
    }
    setSession(data.token, { name: data.name, username: data.username });
    onAuthSuccess();
  } catch (e) {
    showFormError(errEl, 'Network error — is the server running?');
  } finally {
    setButtonLoading(btn, false);
  }
}

// ── Logout ────────────────────────────────────────────────

async function handleLogout() {
  try {
    await fetch('/api/auth/logout', {
      method: 'POST', headers: authHeaders(),
    });
  } catch (_) {}
  clearSession();
  window.lastPrediction = null;
  window.lastDoctorInputs = null;
  document.getElementById('dashboard').classList.add('hidden');
  document.getElementById('auth-screen').classList.remove('hidden');
  showToast('Signed out successfully.');
}

// ── UI Helpers ────────────────────────────────────────────

function showFormError(el, msg) {
  el.textContent = '⚠ ' + msg;
  el.classList.remove('hidden');
}

function setButtonLoading(btn, loading) {
  const text   = btn.querySelector('.btn-text');
  const loader = btn.querySelector('.btn-loader');
  btn.disabled = loading;
  if (text)   text.classList.toggle('hidden', loading);
  if (loader) loader.classList.toggle('hidden', !loading);
}

function togglePw(inputId) {
  const input = document.getElementById(inputId);
  input.type = input.type === 'password' ? 'text' : 'password';
}
