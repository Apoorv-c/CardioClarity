/* ══════════════════════════════════════════════════════════
   dashboard.js — Prediction forms, results, gauge, charts
   Live real-time update: debounced 800 ms after any input
══════════════════════════════════════════════════════════ */

// ── Global State ─────────────────────────────────────────
window.lastPrediction   = null;
window.lastDoctorInputs = null;
window.currentRole      = 'doctor';
window.shapChart        = null;

// ── Debounce / Live-Update ────────────────────────────────
let _liveTimer  = null;
const LIVE_DELAY = 800;   // ms after last keystroke before auto-predicting

function scheduleLiveUpdate() {
  clearTimeout(_liveTimer);
  showLivePill('Updating…');
  _liveTimer = setTimeout(() => {
    if (window.currentRole === 'doctor') {
      _collectAndRunDoctor(false);   // false = don't save to history
    } else {
      _collectAndRunPatient(false);
    }
  }, LIVE_DELAY);
}

// ── Live-update pill indicator ────────────────────────────
function showLivePill(text) {
  let pill = document.getElementById('live-pill');
  if (!pill) {
    pill = document.createElement('div');
    pill.id = 'live-pill';
    pill.style.cssText = [
      'position:fixed', 'top:68px', 'left:50%', 'transform:translateX(-50%)',
      'background:rgba(29,53,87,.92)', 'color:#fff', 'padding:6px 16px',
      'border-radius:100px', 'font-size:.78rem', 'font-weight:600',
      'letter-spacing:.02em', 'z-index:800', 'pointer-events:none',
      'display:flex', 'align-items:center', 'gap:6px',
      'box-shadow:0 4px 16px rgba(0,0,0,.25)',
      'transition:opacity .25s',
    ].join(';');
    document.body.appendChild(pill);
  }
  pill.innerHTML = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;
    background:#4ade80;animation:livePulse 1s infinite"></span>${text}`;
  pill.style.opacity = '1';

  // Inject keyframe once
  if (!document.getElementById('live-pill-style')) {
    const s = document.createElement('style');
    s.id = 'live-pill-style';
    s.textContent = `@keyframes livePulse{0%,100%{opacity:1}50%{opacity:.3}}`;
    document.head.appendChild(s);
  }
}

function hideLivePill() {
  const pill = document.getElementById('live-pill');
  if (pill) { pill.style.opacity = '0'; }
}

// ── Role Management ──────────────────────────────────────

function setRole(role) {
  window.currentRole = role;

  document.getElementById('role-doctor').classList.toggle('active', role === 'doctor');
  document.getElementById('role-patient').classList.toggle('active', role === 'patient');
  document.getElementById('doctor-form-section').classList.toggle('hidden', role !== 'doctor');
  document.getElementById('patient-form-section').classList.toggle('hidden', role !== 'patient');
  document.getElementById('role-banner-doctor').classList.toggle('hidden', role !== 'doctor');
  document.getElementById('role-banner-patient').classList.toggle('hidden', role !== 'patient');

  clearResults();
  updateWhatIfUI();

  // Attach live-update listeners for the newly active form
  _attachLiveListeners();
}

// ── Attach live listeners to all form fields ──────────────

function _attachLiveListeners() {
  const targets = window.currentRole === 'doctor'
    ? ['doc-age','doc-sex','doc-cp','doc-trestbps','doc-chol','doc-fbs',
       'doc-restecg','doc-thalach','doc-exang','doc-oldpeak','doc-slope',
       'doc-ca','doc-thal']
    : ['pat-age','pat-chest','pat-thalach','pat-trestbps','pat-chol'];

  targets.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    // remove old listener to avoid duplicates
    el.removeEventListener('change', scheduleLiveUpdate);
    el.removeEventListener('input',  scheduleLiveUpdate);
    el.addEventListener('change', scheduleLiveUpdate);
    el.addEventListener('input',  scheduleLiveUpdate);
  });

  // Radio buttons in patient form
  ['pat-sex','pat-fbs','pat-exang','pat-bp-known','pat-chol-known'].forEach(name => {
    document.querySelectorAll(`[name="${name}"]`).forEach(el => {
      el.removeEventListener('change', scheduleLiveUpdate);
      el.addEventListener('change', scheduleLiveUpdate);
    });
  });
}

// ── Input validation warnings ─────────────────────────────

function checkInputWarnings(age, trestbps, chol, thalach) {
  const area = document.getElementById('input-warnings');
  const maxHR = 220 - age;
  const warns = [];
  if (trestbps < 90)   warns.push(`⚠ BP of ${trestbps} mm Hg is unusually low — please verify.`);
  if (trestbps > 180)  warns.push(`⚠ BP of ${trestbps} mm Hg is critically elevated.`);
  if (chol < 120)      warns.push(`⚠ Cholesterol of ${chol} mg/dl is extremely low.`);
  if (chol > 500)      warns.push(`⚠ Cholesterol of ${chol} mg/dl is very high.`);
  if (thalach > maxHR) warns.push(`⚠ Heart rate ${thalach} bpm exceeds age-adjusted max (${maxHR} bpm).`);

  if (warns.length > 0) {
    area.innerHTML = warns.map(w => `<div class="warning-item">${w}</div>`).join('');
    area.classList.remove('hidden');
  } else {
    area.classList.add('hidden');
  }
}

// ── Doctor — collect & run ────────────────────────────────

function _collectDoctor() {
  const age      = parseInt(document.getElementById('doc-age').value);
  const sex      = document.getElementById('doc-sex').value;
  const cp       = document.getElementById('doc-cp').value;
  const trestbps = parseInt(document.getElementById('doc-trestbps').value);
  const chol     = parseInt(document.getElementById('doc-chol').value);
  const fbs      = document.getElementById('doc-fbs').value;
  const restecg  = document.getElementById('doc-restecg').value;
  const thalach  = parseInt(document.getElementById('doc-thalach').value);
  const exang    = document.getElementById('doc-exang').value;
  const oldpeak  = parseFloat(document.getElementById('doc-oldpeak').value);
  const slope    = document.getElementById('doc-slope').value;
  const ca       = parseInt(document.getElementById('doc-ca').value);
  const thal     = document.getElementById('doc-thal').value;
  return { age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal };
}

async function _collectAndRunDoctor(saveHistory = true) {
  const d = _collectDoctor();
  if (isNaN(d.age) || isNaN(d.trestbps) || isNaN(d.chol) || isNaN(d.thalach)) return;

  checkInputWarnings(d.age, d.trestbps, d.chol, d.thalach);
  window.lastDoctorInputs = d;

  // Sync what-if sliders
  const wiEl = document.getElementById('wi-bp');
  if (wiEl) { wiEl.value = d.trestbps; document.getElementById('wi-bp-val').textContent = d.trestbps; }
  const wcEl = document.getElementById('wi-chol');
  if (wcEl) { wcEl.value = d.chol; document.getElementById('wi-chol-val').textContent = d.chol; }

  await runPredict({ mode: 'doctor', ...d }, saveHistory);
}

// Form submit (button press) — always saves to history
async function handleDoctorPredict(event) {
  event.preventDefault();
  clearTimeout(_liveTimer);
  await _collectAndRunDoctor(true);
}

// ── Patient — collect & run ───────────────────────────────

function _collectPatient() {
  const age     = parseInt(document.getElementById('pat-age').value);
  const sexEl   = document.querySelector('[name="pat-sex"]:checked');
  const sex     = sexEl ? sexEl.value : 'Male';
  const cp      = document.getElementById('pat-chest').value;
  const exangEl = document.querySelector('[name="pat-exang"]:checked');
  const exang   = exangEl ? exangEl.value : 'No';
  const fbsEl   = document.querySelector('[name="pat-fbs"]:checked');
  const fbs     = fbsEl ? fbsEl.value : 'No';
  const thalach = parseInt(document.getElementById('pat-thalach').value);

  const bpKnownEl   = document.querySelector('[name="pat-bp-known"]:checked');
  const cholKnownEl = document.querySelector('[name="pat-chol-known"]:checked');
  const bpKnown     = bpKnownEl ? bpKnownEl.value === 'yes' : false;
  const cholKnown   = cholKnownEl ? cholKnownEl.value === 'yes' : false;
  const trestbps    = bpKnown   ? parseInt(document.getElementById('pat-trestbps').value) : 130;
  const chol        = cholKnown ? parseInt(document.getElementById('pat-chol').value)      : 220;

  return { age, sex, cp, trestbps, chol, fbs, thalach, exang };
}

async function _collectAndRunPatient(saveHistory = true) {
  const d = _collectPatient();
  if (isNaN(d.age) || isNaN(d.thalach)) return;
  await runPredict({ mode: 'patient', ...d }, saveHistory);
}

async function handlePatientPredict(event) {
  event.preventDefault();
  clearTimeout(_liveTimer);
  await _collectAndRunPatient(true);
}

// ── Core Predict Call ─────────────────────────────────────

async function runPredict(payload, saveHistory = true) {
  const isLive = !saveHistory;

  if (!isLive) showLoading('Analyzing clinical data…');

  try {
    const headers = { ...authHeaders() };
    if (isLive) headers['X-No-History'] = '1';   // tell server not to persist

    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers,
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      if (!isLive) showToast('Prediction error: ' + (data.error || 'Unknown'), 'error');
      return;
    }
    window.lastPrediction = data;
    renderResults(data, isLive);
    updateWhatIfUI();
    if (!isLive && typeof loadHistory === 'function') loadHistory();
  } catch (e) {
    if (!isLive) showToast('Network error: ' + e.message, 'error');
  } finally {
    if (!isLive) hideLoading();
    hideLivePill();
  }
}

// ── Render Results ────────────────────────────────────────

function renderResults(data, quiet = false) {
  const section = document.getElementById('results-section');
  section.classList.remove('hidden');
  if (!quiet) section.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const { tier_key, tier_label, probability, patient_mode,
          derived, recommendations, warnings, inputs, shap_contributions } = data;

  // ── Metric Cards ──────────────────────────────────────
  const bp   = inputs.trestbps;
  const chol = inputs.chol;
  const hr   = inputs.thalach;
  const cvr  = derived?.cardiovascular_risk ?? 0;

  setMetricCard('bp',     `${bp} mm Hg`,   bpBadge(bp));
  setMetricCard('chol',   `${chol} mg/dl`, cholBadge(chol));
  setMetricCard('hr',     `${hr} bpm`,     hrBadge(hr, inputs.age));
  setMetricCard('cvrisk', cvr.toString(),  cvrBadge(cvr));

  // ── Gauge ──────────────────────────────────────────────
  animateGauge(patient_mode ? null : probability, tier_key, tier_label);

  // ── Risk Badge + Advice ───────────────────────────────
  const badgeEl = document.getElementById('risk-badge-large');
  badgeEl.textContent = tier_label;
  badgeEl.className   = `risk-badge-large risk-badge-${tier_key}`;

  const binaryEl = document.getElementById('result-binary-label');
  binaryEl.textContent = patient_mode
    ? 'Exact percentage not shown for patient privacy and screening accuracy.'
    : `Binary label (0.5 cutoff): ${data.prediction === 1 ? 'Disease detected' : 'No disease detected'}`;

  const adviceMap = {
    high:   '⚠️ Please consult a cardiologist or primary care physician promptly.',
    medium: '📌 Consider discussing these results with your doctor at next visit.',
    low:    '✅ Keep up your healthy habits and attend regular check-ups.',
  };
  document.getElementById('result-advice').textContent = adviceMap[tier_key] || '';

  const recsEl = document.getElementById('result-recommendations');
  recsEl.innerHTML = recommendations?.length
    ? '<h4>📋 Recommendations</h4><ul>' +
      recommendations.slice(0, 5).map(r => `<li>${r}</li>`).join('') + '</ul>'
    : '';

  const warnEl = document.getElementById('result-warnings');
  if (warnings?.length) {
    warnEl.innerHTML = warnings.map(w => `<p>⚠️ ${w}</p>`).join('');
    warnEl.classList.remove('hidden');
  } else {
    warnEl.classList.add('hidden');
  }

  // ── SHAP / Bands ──────────────────────────────────────
  const shapSection  = document.getElementById('shap-section');
  const patientBands = document.getElementById('patient-bands');
  const whatifNavBtn = document.getElementById('whatif-nav-btn');

  if (!patient_mode) {
    shapSection.classList.remove('hidden');
    patientBands.classList.add('hidden');
    whatifNavBtn.classList.remove('hidden');
    renderShapChart(shap_contributions);
  } else {
    shapSection.classList.add('hidden');
    patientBands.classList.remove('hidden');
    whatifNavBtn.classList.add('hidden');
    renderPatientBands(tier_key);
  }
}

// ── Metric Card Helper ────────────────────────────────────

function setMetricCard(id, value, badge) {
  document.getElementById(`card-${id}-val`).textContent = value;
  const badgeEl = document.getElementById(`card-${id}-badge`);
  badgeEl.textContent = badge.text;
  badgeEl.className   = `metric-badge ${badge.cls}`;
}

function bpBadge(bp) {
  if (bp > 140)  return { text: 'Hypertensive', cls: 'badge-red' };
  if (bp > 120)  return { text: 'Elevated',     cls: 'badge-amber' };
  return               { text: 'Normal',        cls: 'badge-green' };
}
function cholBadge(chol) {
  if (chol > 240) return { text: 'High',       cls: 'badge-red' };
  if (chol > 200) return { text: 'Borderline', cls: 'badge-amber' };
  return                 { text: 'Desirable',  cls: 'badge-green' };
}
function hrBadge(hr, age) {
  const target = (220 - (age || 50)) * 0.85;
  return hr >= target ? { text: 'Good', cls: 'badge-green' } : { text: 'Below target', cls: 'badge-amber' };
}
function cvrBadge(cvr) {
  if (cvr >= 3) return { text: 'High (3)',      cls: 'badge-red' };
  if (cvr >= 1) return { text: `Score: ${cvr}`, cls: 'badge-amber' };
  return               { text: 'Low (0)',        cls: 'badge-green' };
}

// ── Gauge Animation ───────────────────────────────────────

function animateGauge(probability, tierKey, tierLabel) {
  const pctEl   = document.getElementById('gauge-percent-text');
  const labelEl = document.getElementById('gauge-risk-label');
  const needle  = document.getElementById('gauge-needle');
  const arc     = document.getElementById('gauge-arc');

  const colorMap = { low: '#22c55e', medium: '#f59e0b', high: '#ef4444' };
  const color = colorMap[tierKey] || '#6b7280';

  if (probability === null || probability === undefined) {
    pctEl.textContent   = tierLabel;
    labelEl.textContent = 'Screening Band';
    return;
  }

  const target = Math.round(probability * 100);
  let current  = 0;
  const step   = Math.ceil(target / 30);
  pctEl.style.color   = color;
  labelEl.textContent = tierLabel;

  const timer = setInterval(() => {
    current = Math.min(current + step, target);
    pctEl.textContent = current + '%';
    if (current >= target) clearInterval(timer);
  }, 30);

  const angle = (probability * 180) - 90;
  needle.style.transition = 'transform 0.9s cubic-bezier(.25,.1,.25,1)';
  needle.setAttribute('transform', `rotate(${angle}, 150, 150)`);

  const startAngle = Math.PI;
  const endAngle   = Math.PI - (probability * Math.PI);
  const r = 120, cx = 150, cy = 150;
  const x1 = cx + r * Math.cos(startAngle);
  const y1 = cy + r * Math.sin(startAngle);
  const x2 = cx + r * Math.cos(endAngle);
  const y2 = cy + r * Math.sin(endAngle);
  arc.setAttribute('d', `M ${x1} ${y1} A ${r} ${r} 0 ${probability > 0.5 ? 1 : 0} 1 ${x2} ${y2}`);
  arc.setAttribute('stroke', color);
}

// ── SHAP Bar Chart (Chart.js) ─────────────────────────────

function renderShapChart(contributions) {
  const canvas = document.getElementById('shap-chart');
  if (!canvas) return;
  if (window.shapChart) { window.shapChart.destroy(); window.shapChart = null; }
  if (!contributions?.length) return;

  const labels = contributions.map(c => c.feature);
  const values = contributions.map(c => c.value);
  const colors = values.map(v => v > 0 ? 'rgba(239,68,68,.75)' : 'rgba(34,197,94,.75)');

  window.shapChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace(',.75)', ',1)')),
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.raw > 0 ? '+' : ''}${ctx.raw.toFixed(4)} SHAP  (${ctx.raw > 0 ? 'risk ↑' : 'risk ↓'})`,
          },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(0,0,0,.06)' }, ticks: { font: { size: 11 } } },
        y: { grid: { display: false }, ticks: { font: { size: 11 }, color: '#343a40' } },
      },
    },
  });
}

// ── Patient Bands ─────────────────────────────────────────

function renderPatientBands(tierKey) {
  ['low', 'medium', 'high'].forEach(k => {
    document.getElementById(`band-${k}`).classList.toggle('active', k === tierKey);
  });
}

// ── Patient form BP/Chol toggles ──────────────────────────

function toggleBPInput(radio) {
  const show = radio.value === 'yes';
  document.getElementById('pat-bp-input').style.display = show ? '' : 'none';
  document.getElementById('pat-bp-median').classList.toggle('hidden', show);
  scheduleLiveUpdate();
}
function toggleCholInput(radio) {
  const show = radio.value === 'yes';
  document.getElementById('pat-chol-input').style.display = show ? '' : 'none';
  document.getElementById('pat-chol-median').classList.toggle('hidden', show);
  scheduleLiveUpdate();
}

// ── Clear Results ─────────────────────────────────────────

function clearResults() {
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('input-warnings').classList.add('hidden');
  if (window.shapChart) { window.shapChart.destroy(); window.shapChart = null; }
  const needle = document.getElementById('gauge-needle');
  if (needle) needle.setAttribute('transform', 'rotate(-90, 150, 150)');
  const arc = document.getElementById('gauge-arc');
  if (arc) arc.setAttribute('d', '');
  document.getElementById('gauge-percent-text').textContent = '—';
  document.getElementById('gauge-risk-label').textContent   = '—';
}

// ── What-If UI State ──────────────────────────────────────

function updateWhatIfUI() {
  const hasPredict = !!window.lastPrediction;
  const isDoctor   = window.currentRole === 'doctor';
  document.getElementById('whatif-patient-notice').classList.toggle('hidden', isDoctor);
  document.getElementById('whatif-no-predict').classList.toggle('hidden', hasPredict || !isDoctor);
  document.getElementById('whatif-controls').classList.toggle('hidden', !hasPredict || !isDoctor);
}

// ── Load Model Metrics (Insights tab) ─────────────────────

async function loadModelMetrics() {
  try {
    const res  = await fetch('/api/model/metrics', { headers: authHeaders() });
    const data = await res.json();
    const grid = document.getElementById('metrics-grid');
    const fcEl = document.getElementById('feature-count');
    if (fcEl) fcEl.textContent = data.feature_count || 25;
    grid.innerHTML = Object.entries(data.metrics || {}).map(([name, val]) => `
      <div class="insight-metric-card">
        <div class="insight-metric-name">${name}</div>
        <div class="insight-metric-value">${(val * 100).toFixed(1)}%</div>
      </div>`).join('');
  } catch (e) {
    console.warn('Could not load metrics:', e);
  }
}

// ── What-If Runner ────────────────────────────────────────

async function runWhatIf() {
  if (!window.lastDoctorInputs) return;

  const wi_bp   = parseInt(document.getElementById('wi-bp').value);
  const wi_chol = parseInt(document.getElementById('wi-chol').value);

  try {
    const res  = await fetch('/api/whatif', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ ...window.lastDoctorInputs, wi_bp, wi_chol }),
    });
    const data = await res.json();
    if (!res.ok) return;

    document.getElementById('wi-base-prob').textContent = (data.base_prob * 100).toFixed(1) + '%';
    document.getElementById('wi-new-prob').textContent  = (data.wi_prob   * 100).toFixed(1) + '%';

    const delta   = data.delta;
    const deltaEl = document.getElementById('wi-delta');
    deltaEl.textContent = (delta >= 0 ? '+' : '') + (delta * 100).toFixed(1) + '%';
    deltaEl.style.color = delta > 0.01 ? '#ef4444' : delta < -0.01 ? '#22c55e' : '#343a40';

    document.getElementById('wi-label').textContent        = data.wi_label;
    document.getElementById('whatif-message').textContent  = data.message;
  } catch (_) {}
}

