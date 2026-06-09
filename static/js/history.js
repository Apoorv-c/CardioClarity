/* ══════════════════════════════════════════════════════════
   history.js — Patient History Tracking & Progress Compare
══════════════════════════════════════════════════════════ */

// ── Load & Render History ─────────────────────────────────

async function loadHistory() {
  try {
    const res  = await fetch('/api/history', { headers: authHeaders() });
    if (!res.ok) return;
    const data = await res.json();
    renderHistory(data.history || []);
  } catch (e) {
    console.warn('Could not load history:', e);
  }
}

function renderHistory(history) {
  const list       = document.getElementById('history-list');
  const progressEl = document.getElementById('history-progress');

  if (!history || history.length === 0) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📊</div>
        <p>No predictions yet. Run your first assessment!</p>
        <button class="btn-primary" onclick="switchTab('predict')">Start Assessment</button>
      </div>`;
    progressEl.classList.add('hidden');
    return;
  }

  // ── Progress Banner: compare first & last doctor prediction ──
  const doctorEntries = history.filter(h => h.mode === 'doctor' && h.probability != null);
  if (doctorEntries.length >= 2) {
    const first    = doctorEntries[0];
    const last     = doctorEntries[doctorEntries.length - 1];
    const firstPct = Math.round(first.probability * 100);
    const lastPct  = Math.round(last.probability  * 100);
    const diff     = lastPct - firstPct;

    if (diff < 0) {
      progressEl.innerHTML = `
        🎉 <strong>Great progress!</strong>
        Your risk reduced from <strong>${firstPct}%</strong> → <strong>${lastPct}%</strong>
        (${Math.abs(diff)} percentage points lower since your first assessment).`;
      progressEl.style.background  = 'linear-gradient(135deg, #dcfce7, #bbf7d0)';
      progressEl.style.borderColor = '#86efac';
      progressEl.style.color       = '#14532d';
    } else if (diff > 0) {
      progressEl.innerHTML = `
        📈 Your risk increased from <strong>${firstPct}%</strong> → <strong>${lastPct}%</strong>.
        Consider reviewing lifestyle factors and consulting your doctor.`;
      progressEl.style.background  = '#fff7ed';
      progressEl.style.borderColor = '#fed7aa';
      progressEl.style.color       = '#92400e';
    } else {
      progressEl.innerHTML = `
        📊 Your risk has stayed consistent at <strong>${lastPct}%</strong> across ${doctorEntries.length} assessments.`;
      progressEl.style.background  = '#eff6ff';
      progressEl.style.borderColor = '#bfdbfe';
      progressEl.style.color       = '#1e40af';
    }
    progressEl.classList.remove('hidden');
  } else {
    progressEl.classList.add('hidden');
  }

  // ── History List ──────────────────────────────────────
  const tierEmoji = { low: '🟢', medium: '🟡', high: '🔴' };
  const tierCls   = { low: 'badge-green', medium: 'badge-amber', high: 'badge-red' };

  list.innerHTML = [...history].reverse().map(h => {
    const date   = new Date(h.timestamp);
    const tstamp = isNaN(date) ? h.timestamp : date.toLocaleString();
    const pctStr = h.probability != null ? `${(h.probability * 100).toFixed(1)}%` : 'Screening (no %)';
    const emoji  = tierEmoji[h.tier_key] || '⚪';
    const cls    = tierCls[h.tier_key]   || 'badge-gray';

    return `
      <div class="history-item" id="hist-${h.id}">
        <div class="history-item-time">📅 ${tstamp}</div>
        <div class="history-item-mode">${h.mode === 'doctor' ? '🧑‍⚕️ Doctor' : '👤 Patient'}</div>
        <div class="history-item-stats">
          <span class="history-stat">Age: <strong>${h.age}</strong></span>
          <span class="history-stat">Sex: <strong>${h.sex}</strong></span>
          <span class="history-stat">BP: <strong>${h.trestbps} mm Hg</strong></span>
          <span class="history-stat">Chol: <strong>${h.chol} mg/dl</strong></span>
          <span class="history-stat">Risk: <strong>${pctStr}</strong></span>
        </div>
        <span class="metric-badge ${cls}">${emoji} ${h.tier_label}</span>
        <button class="history-item-delete" onclick="deleteHistoryItem('${h.id}')" title="Delete">🗑</button>
      </div>`;
  }).join('');
}

// ── Delete Entry ──────────────────────────────────────────

async function deleteHistoryItem(id) {
  try {
    await fetch(`/api/history/${id}`, { method: 'DELETE', headers: authHeaders() });
    await loadHistory();
    showToast('Entry deleted.', 'success');
  } catch (e) {
    showToast('Could not delete entry.', 'error');
  }
}
