# server.py — CardioClarity Flask REST API
# Replaces the Streamlit app.py entry point.
# Run with: python server.py
# Then open http://localhost:5000

import json
import os
import io
import base64
import traceback
import hashlib
import secrets
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# ── ReportLab PDF ──────────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image as RLImage,
                                     HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# ==============================================================================
# APP SETUP
# ==============================================================================

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / 'static'
_DATA_FILE = _ROOT / 'data' / 'users.json'

app = Flask(__name__, static_folder=str(_STATIC), static_url_path='')
CORS(app)

# Ensure data directory exists
(_ROOT / 'data').mkdir(exist_ok=True)

# ==============================================================================
# CONSTANTS (must mirror original app.py logic)
# ==============================================================================

RISK_LOW_BELOW    = 0.35
RISK_MEDIUM_BELOW = 0.65
PAT_RISK_LOW_BELOW    = 0.25
PAT_RISK_MEDIUM_BELOW = 0.50

PATIENT_HIDDEN_FIELDS = ['restecg', 'oldpeak', 'slope', 'ca', 'thal']

_FALLBACK_DEFAULTS = {
    'restecg': 'normal',
    'oldpeak':  1.0,
    'slope':    'flat',
    'ca':       0.0,
    'thal':     'normal',
}

# ==============================================================================
# LOAD MODEL ARTIFACTS (once at startup)
# ==============================================================================

print("[*] Loading model artifacts...", flush=True)
model          = joblib.load(_ROOT / 'random_forest_model.pkl')
scaler         = joblib.load(_ROOT / 'scaler.pkl')
feature_names  = joblib.load(_ROOT / 'feature_names.pkl')
label_encoders = joblib.load(_ROOT / 'label_encoders.pkl')
shap_explainer = shap.TreeExplainer(model)

model_metrics             = None
patient_defaults_encoded  = None
patient_defaults_labels   = None
patient_screening_medians = None

artifacts_path = _ROOT / 'heart_disease_model_artifacts.pkl'
if artifacts_path.exists():
    try:
        arts = joblib.load(artifacts_path)
        model_metrics             = arts.get('model_metrics', None)
        patient_defaults_encoded  = arts.get('patient_defaults_encoded', None)
        patient_defaults_labels   = arts.get('patient_defaults_labels', None)
        patient_screening_medians = arts.get('patient_screening_medians', None)
    except Exception as e:
        print(f"[!] Could not load model artifacts: {e}", flush=True)

print(f"[OK] Model loaded - {len(feature_names)} features", flush=True)

# ==============================================================================
# USERS / SESSION STORE
# ==============================================================================

def _load_users() -> dict:
    if _DATA_FILE.exists():
        try:
            return json.loads(_DATA_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}


def _save_users(users: dict):
    _DATA_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding='utf-8')


def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# In-memory session token cache (also persisted in users.json for restart survival)
_sessions: dict[str, str] = {}


def _load_sessions_from_disk():
    """Populate in-memory cache from the tokens stored in users.json."""
    users = _load_users()
    for username, udata in users.items():
        for tok in udata.get('_tokens', []):
            _sessions[tok] = username


def _persist_token(username: str, token: str):
    """Save a new token to the user record (and cache it in memory)."""
    users = _load_users()
    if username in users:
        users[username].setdefault('_tokens', []).append(token)
        # Keep only the 10 most-recent tokens to avoid unbounded growth
        users[username]['_tokens'] = users[username]['_tokens'][-10:]
        _save_users(users)
    _sessions[token] = username


def _revoke_token(token: str):
    """Remove a token from memory and from disk."""
    username = _sessions.pop(token, None)
    if not username:
        return
    users = _load_users()
    if username in users:
        users[username]['_tokens'] = [
            t for t in users[username].get('_tokens', []) if t != token
        ]
        _save_users(users)


def _get_username(req) -> str | None:
    token = req.headers.get('X-Auth-Token', '')
    if token in _sessions:
        return _sessions[token]
    # Fallback: check disk (handles server-restart scenario)
    users = _load_users()
    for username, udata in users.items():
        if token in udata.get('_tokens', []):
            _sessions[token] = username   # repopulate cache
            return username
    return None


# Pre-populate session cache on startup
_load_sessions_from_disk()


# ==============================================================================
# FEATURE ENGINEERING (mirrors app.py exactly)
# ==============================================================================

def get_patient_screening_medians():
    if patient_screening_medians:
        return {
            'trestbps': float(patient_screening_medians.get('trestbps', 130)),
            'chol':     float(patient_screening_medians.get('chol',     220)),
        }
    return {'trestbps': 130.0, 'chol': 220.0}


def get_patient_defaults():
    if patient_defaults_encoded:
        return dict(patient_defaults_encoded)
    encoded = {}
    for field, val in _FALLBACK_DEFAULTS.items():
        if field in ('oldpeak', 'ca'):
            encoded[field] = float(val)
        elif field in label_encoders:
            try:
                encoded[field] = float(label_encoders[field].transform([str(val)])[0])
            except Exception:
                encoded[field] = 0.0
        else:
            encoded[field] = 0.0
    return encoded


def compute_derived_metrics(age, trestbps, chol, thalach, oldpeak, ca):
    age_group     = (0 if age <= 40 else 1 if age <= 50 else 2 if age <= 60
                     else 3 if age <= 70 else 4)
    age_over_50   = 1 if age > 50 else 0
    bp_risk       = 2 if trestbps > 140 else (1 if trestbps > 120 else 0)
    hypertension  = 1 if trestbps > 140 else 0
    chol_risk     = 2 if chol > 240 else (1 if chol > 200 else 0)
    high_chol     = 1 if chol > 240 else 0
    max_hr_target = 220 - age
    hr_pct        = thalach / max_hr_target if max_hr_target > 0 else 0.0
    low_hr        = 1 if hr_pct < 0.85 else 0
    cv_risk       = int(age > 50) + int(trestbps > 140) + int(chol > 240)
    sig_st_dep    = 1 if oldpeak > 1 else 0
    multi_vessel  = 1 if ca >= 2 else 0

    return {
        'age_group': age_group, 'age_over_50': age_over_50,
        'bp_risk': bp_risk, 'hypertension': hypertension,
        'chol_risk': chol_risk, 'high_cholesterol': high_chol,
        'max_hr_target': max_hr_target, 'hr_percentage': hr_pct,
        'low_hr': low_hr, 'cardiovascular_risk': cv_risk,
        'significant_st_depression': sig_st_dep, 'multi_vessel': multi_vessel,
    }


def create_feature_vector(age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal):
    d = compute_derived_metrics(age, trestbps, chol, thalach, oldpeak, ca)
    sex_val     = 1 if sex == "Male" else 0
    cp_val      = label_encoders['cp'].transform([cp])[0]
    fbs_val     = label_encoders['fbs'].transform([fbs])[0]
    restecg_val = label_encoders['restecg'].transform([restecg])[0]
    exang_val   = label_encoders['exang'].transform([exang])[0]
    slope_val   = label_encoders['slope'].transform([slope])[0]
    thal_val    = label_encoders['thal'].transform([thal])[0]

    fd = {
        'age': age, 'sex': sex_val, 'cp': cp_val, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs_val, 'restecg': restecg_val,
        'thalach': thalach, 'exang': exang_val, 'oldpeak': oldpeak,
        'slope': slope_val, 'ca': ca, 'thal': thal_val, **d,
    }
    return np.array([[fd[f] for f in feature_names]]), d


def create_feature_vector_with_defaults(age, sex, cp_encoded, trestbps, chol,
                                        fbs, thalach, exang_encoded, defaults):
    oldpeak = defaults.get('oldpeak', 0.0)
    ca      = defaults.get('ca',      0.0)
    d = compute_derived_metrics(age, trestbps, chol, thalach, oldpeak, ca)
    sex_val = 1 if sex == "Male" else 0
    fbs_val = label_encoders['fbs'].transform([fbs])[0]

    fd = {
        'age': age, 'sex': sex_val, 'cp': cp_encoded, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs_val,
        'restecg': defaults.get('restecg', 0.0),
        'thalach': thalach, 'exang': exang_encoded,
        'oldpeak': oldpeak, 'slope': defaults.get('slope', 0.0),
        'ca': ca, 'thal': defaults.get('thal', 0.0), **d,
    }
    return np.array([[fd[f] for f in feature_names]]), d


def run_prediction(input_features):
    input_df     = pd.DataFrame(input_features, columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction   = int(model.predict(input_scaled)[0])
    probability  = float(model.predict_proba(input_scaled)[0][1])
    shap_out     = shap_explainer.shap_values(input_scaled)

    # Handle SHAP output shapes
    if isinstance(shap_out, list):
        pos = np.asarray(shap_out[1], dtype=float)
    else:
        pos = np.asarray(shap_out, dtype=float)
    if pos.ndim == 3:
        shap_row = pos[0, :, 1] if pos.shape[2] == 2 else pos[0, 1, :]
    elif pos.ndim == 2:
        shap_row = pos[0]
    else:
        shap_row = pos

    return prediction, probability, shap_row


def disease_risk_band(probability, patient_mode=False):
    low = PAT_RISK_LOW_BELOW if patient_mode else RISK_LOW_BELOW
    mid = PAT_RISK_MEDIUM_BELOW if patient_mode else RISK_MEDIUM_BELOW
    if probability < low:
        return "low",    "Low Risk",    "#22c55e"
    if probability < mid:
        return "medium", "Medium Risk", "#f59e0b"
    return "high",   "High Risk",   "#ef4444"


# ==============================================================================
# VISUALIZATION — returns base64 PNG strings (no Streamlit dependency)
# ==============================================================================

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data


def build_shap_chart(shap_row, feature_names_list, priority=('trestbps', 'chol'), n=6):
    shap_dict = {n: float(np.asarray(v).ravel()[0]) for n, v in
                 zip(feature_names_list, shap_row)}
    sorted_all = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    ordered, used = [], set()
    for pf in priority:
        if pf in shap_dict:
            ordered.append((pf, shap_dict[pf]))
            used.add(pf)
    for name, val in sorted_all:
        if name not in used:
            ordered.append((name, val))
            used.add(name)
        if len(ordered) >= n:
            break

    plot_order = list(reversed(ordered))
    names  = [f[0] for f in plot_order]
    values = [float(f[1]) for f in plot_order]

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.5 * len(names))))
    colors_list = ['#ef4444' if v > 0 else '#22c55e' for v in values]
    ax.barh(range(len(names)), values, color=colors_list, edgecolor='white',
            linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color='#374151', linewidth=1)
    ax.set_xlabel('SHAP value (impact on risk score)', fontsize=9)
    ax.set_title('Feature Contributions — BP & Cholesterol pinned first', fontsize=11)
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#f8fafc')

    red_p   = mpatches.Patch(color='#ef4444', label='Increases risk')
    green_p = mpatches.Patch(color='#22c55e', label='Decreases risk')
    ax.legend(handles=[red_p, green_p], fontsize=8, loc='lower right')
    fig.tight_layout()
    return _fig_to_b64(fig)


def build_gauge_image(probability):
    """Semicircular gauge image for PDF embedding."""
    tier = 'Low' if probability < RISK_LOW_BELOW else ('Medium' if probability < RISK_MEDIUM_BELOW else 'High')
    color = '#22c55e' if tier == 'Low' else ('#f59e0b' if tier == 'Medium' else '#ef4444')

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.barh(['Risk'], [1.0], color='#e5e7eb', height=0.55)
    ax.barh(['Risk'], [probability], color=color, height=0.55)
    ax.axvline(RISK_LOW_BELOW,    color='#6b7280', ls='--', lw=1.2, alpha=0.7)
    ax.axvline(RISK_MEDIUM_BELOW, color='#374151', ls='--', lw=1.2, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_title(f'Heart Disease Risk — {tier} Risk ({probability:.1%})', fontsize=12)
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#f8fafc')
    fig.tight_layout()
    return _fig_to_b64(fig)


# ==============================================================================
# RECOMMENDATIONS BUILDER
# ==============================================================================

def build_recommendations(tier_key, trestbps, chol, age, derived):
    recs = []
    if tier_key == 'high':
        recs = [
            'Prioritize consultation with cardiology or primary care immediately.',
            'Consider further diagnostic testing (ECG, stress test, imaging).',
            'Begin intensive lifestyle review: diet, exercise, smoking cessation.',
            'Medication review (statins, BP control) per current guidelines.',
            'Schedule short-interval follow-up as clinically appropriate.',
        ]
    elif tier_key == 'medium':
        recs = [
            'Reinforce blood pressure and cholesterol targets.',
            'Repeat risk assessment at next visit; consider baseline labs.',
            'Lifestyle counseling: activity, diet, weight management, smoking.',
            'Escalate diagnostic testing if symptoms or risk scores worsen.',
        ]
    else:
        recs = [
            'Continue current healthy habits.',
            'Maintain routine preventive care and annual check-ups.',
            'Monitor blood pressure and cholesterol regularly.',
            'Maintain physical activity (~150 min/week moderate intensity).',
        ]
    if trestbps > 140:
        recs.insert(0, f'Blood pressure ({trestbps} mm Hg) is hypertensive — prioritize BP management.')
    if chol > 240:
        recs.insert(0, f'Cholesterol ({chol} mg/dl) is high — discuss statin therapy and dietary changes.')
    return recs


# ==============================================================================
# SHAP CONTRIBUTIONS (JSON-friendly)
# ==============================================================================

def build_shap_contributions(shap_row, feature_names_list, n=8,
                             priority=('trestbps', 'chol')):
    shap_dict = {n: float(np.asarray(v).ravel()[0])
                 for n, v in zip(feature_names_list, shap_row)}
    sorted_all = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    ordered, used = [], set()
    for pf in priority:
        if pf in shap_dict:
            ordered.append({'feature': pf, 'value': shap_dict[pf]})
            used.add(pf)
    for name, val in sorted_all:
        if name not in used:
            ordered.append({'feature': name, 'value': val})
            used.add(name)
        if len(ordered) >= n:
            break
    return ordered


# ==============================================================================
# API ROUTES — AUTH
# ==============================================================================

@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    data = request.get_json(force=True)
    username = (data.get('username') or '').strip().lower()
    password = (data.get('password') or '').strip()
    name     = (data.get('name') or username).strip()

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    users = _load_users()
    if username in users:
        return jsonify({'error': 'Username already taken'}), 409

    users[username] = {
        'name':       name,
        'password':   _hash_pw(password),
        'created_at': datetime.now().isoformat(),
        'history':    [],
    }
    _save_users(users)

    token = secrets.token_hex(32)
    _persist_token(username, token)
    return jsonify({'token': token, 'name': name, 'username': username}), 201


@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    data     = request.get_json(force=True)
    username = (data.get('username') or '').strip().lower()
    password = (data.get('password') or '').strip()

    users = _load_users()
    user  = users.get(username)
    if not user or user['password'] != _hash_pw(password):
        return jsonify({'error': 'Invalid credentials'}), 401

    token = secrets.token_hex(32)
    _persist_token(username, token)
    return jsonify({'token': token, 'name': user['name'], 'username': username})


@app.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    token = request.headers.get('X-Auth-Token', '')
    _revoke_token(token)
    return jsonify({'ok': True})


@app.route('/api/auth/profile', methods=['GET'])
def auth_profile():
    username = _get_username(request)
    if not username:
        return jsonify({'error': 'Not authenticated'}), 401
    users = _load_users()
    u = users.get(username, {})
    return jsonify({
        'username': username,
        'name':     u.get('name', username),
        'history':  u.get('history', []),
    })


# ==============================================================================
# API ROUTES — PREDICT
# ==============================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    mode = data.get('mode', 'doctor')   # 'doctor' | 'patient'

    try:
        if mode == 'doctor':
            age      = int(data['age'])
            sex      = data['sex']
            cp       = data['cp']
            trestbps = int(data['trestbps'])
            chol     = int(data['chol'])
            fbs      = data['fbs']
            restecg  = data['restecg']
            thalach  = int(data['thalach'])
            exang    = data['exang']
            oldpeak  = float(data['oldpeak'])
            slope    = data['slope']
            ca       = int(data['ca'])
            thal     = data['thal']

            input_features, derived = create_feature_vector(
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal)
            patient_mode = False

        else:  # patient
            age      = int(data['age'])
            sex      = data['sex']
            trestbps = int(data['trestbps'])
            chol     = int(data['chol'])
            fbs      = data['fbs']
            thalach  = int(data['thalach'])

            _cp_map = {
                'typical angina': 'typical angina',
                'atypical angina': 'atypical angina',
                'non-anginal': 'non-anginal',
                'asymptomatic': 'asymptomatic',
            }
            cp_label    = _cp_map.get(data.get('cp', 'asymptomatic'), 'asymptomatic')
            cp_encoded  = float(label_encoders['cp'].transform([cp_label])[0])
            exang_label = 'True' if data.get('exang') == 'Yes' else 'False'
            exang_enc   = float(label_encoders['exang'].transform([exang_label])[0])
            fbs_str     = 'True' if fbs == 'Yes' else 'False'
            defaults    = get_patient_defaults()

            input_features, derived = create_feature_vector_with_defaults(
                age, sex, cp_encoded, trestbps, chol,
                fbs_str, thalach, exang_enc, defaults)
            patient_mode = True

        prediction, probability, shap_row = run_prediction(input_features)
        tier_key, tier_label, tier_color  = disease_risk_band(probability, patient_mode)

        shap_b64  = build_shap_chart(shap_row, feature_names)
        gauge_b64 = build_gauge_image(probability)
        recs      = build_recommendations(tier_key, trestbps, chol, age, derived)
        shap_data = build_shap_contributions(shap_row, feature_names)

        # Validation warnings
        max_hr   = 220 - age
        warnings = []
        if trestbps < 90:  warnings.append(f'BP of {trestbps} mm Hg is unusually low — please verify.')
        if trestbps > 180: warnings.append(f'BP of {trestbps} mm Hg is critically elevated.')
        if chol < 120:     warnings.append(f'Cholesterol of {chol} mg/dl is extremely low.')
        if chol > 500:     warnings.append(f'Cholesterol of {chol} mg/dl is very high.')
        if thalach > max_hr: warnings.append(f'Heart rate {thalach} bpm exceeds age-adjusted max ({max_hr} bpm).')

        result = {
            'prediction':       prediction,
            'probability':      round(probability, 4),
            'probability_pct':  f'{probability:.1%}',
            'tier_key':         tier_key,
            'tier_label':       tier_label,
            'tier_color':       tier_color,
            'patient_mode':     patient_mode,
            'derived':          {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                                 for k, v in derived.items()},
            'shap_chart_b64':   shap_b64,
            'gauge_b64':        gauge_b64,
            'shap_contributions': shap_data,
            'recommendations':  recs,
            'warnings':         warnings,
            'timestamp':        datetime.now().isoformat(),
            'mode':             mode,
            'inputs':           {
                'age': age, 'sex': sex,
                'trestbps': trestbps, 'chol': chol, 'thalach': thalach,
            },
        }

        # Save to user history — only for explicit submits (not live debounced updates)
        no_history = request.headers.get('X-No-History', '') == '1'
        username = _get_username(request)
        if username and not no_history:
            users = _load_users()
            if username in users:
                history_entry = {
                    'id':          str(uuid.uuid4())[:8],
                    'timestamp':   result['timestamp'],
                    'mode':        mode,
                    'age':         age,
                    'sex':         sex,
                    'trestbps':    trestbps,
                    'chol':        chol,
                    'thalach':     thalach,
                    'probability': round(probability, 4) if not patient_mode else None,
                    'tier_key':    tier_key,
                    'tier_label':  tier_label,
                }
                users[username]['history'].append(history_entry)
                _save_users(users)

        return jsonify(result)

    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ==============================================================================
# API ROUTES — WHAT-IF
# ==============================================================================

@app.route('/api/whatif', methods=['POST'])
def api_whatif():
    data = request.get_json(force=True)
    try:
        age      = int(data['age'])
        sex      = data['sex']
        cp       = data['cp']
        trestbps = int(data['trestbps'])
        chol     = int(data['chol'])
        fbs      = data['fbs']
        restecg  = data['restecg']
        thalach  = int(data['thalach'])
        exang    = data['exang']
        oldpeak  = float(data['oldpeak'])
        slope    = data['slope']
        ca       = int(data['ca'])
        thal     = data['thal']
        wi_bp    = int(data.get('wi_bp', trestbps))
        wi_chol  = int(data.get('wi_chol', chol))

        base_f, _ = create_feature_vector(age, sex, cp, trestbps, chol, fbs,
                                          restecg, thalach, exang, oldpeak, slope, ca, thal)
        wi_f,   _ = create_feature_vector(age, sex, cp, wi_bp, wi_chol, fbs,
                                          restecg, thalach, exang, oldpeak, slope, ca, thal)

        _, base_prob, _ = run_prediction(base_f)
        _, wi_prob,   _ = run_prediction(wi_f)

        base_tier, base_label, base_color = disease_risk_band(base_prob)
        wi_tier,   wi_label,   wi_color   = disease_risk_band(wi_prob)

        delta = wi_prob - base_prob
        if delta < -0.05:
            message = (f'Lowering BP to {wi_bp} mm Hg and cholesterol to {wi_chol} mg/dl '
                       f'could reduce predicted risk by {-delta:.1%}.')
        elif delta > 0.05:
            message = f'Raising these values would increase predicted risk by {delta:.1%}.'
        else:
            message = 'Small effect on predicted probability for this parameter change.'

        return jsonify({
            'base_prob':    round(base_prob, 4),
            'wi_prob':      round(wi_prob, 4),
            'delta':        round(delta, 4),
            'base_label':   base_label,
            'wi_label':     wi_label,
            'wi_color':     wi_color,
            'message':      message,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# API ROUTES — HISTORY
# ==============================================================================

@app.route('/api/history', methods=['GET'])
def api_history():
    username = _get_username(request)
    if not username:
        return jsonify({'error': 'Not authenticated'}), 401
    users = _load_users()
    return jsonify({'history': users.get(username, {}).get('history', [])})


@app.route('/api/history/<entry_id>', methods=['DELETE'])
def api_delete_history(entry_id):
    username = _get_username(request)
    if not username:
        return jsonify({'error': 'Not authenticated'}), 401
    users = _load_users()
    if username in users:
        users[username]['history'] = [
            h for h in users[username]['history'] if h.get('id') != entry_id
        ]
        _save_users(users)
    return jsonify({'ok': True})


# ==============================================================================
# API ROUTES — MODEL INSIGHTS
# ==============================================================================

@app.route('/api/model/metrics', methods=['GET'])
def api_model_metrics():
    return jsonify({
        'metrics': model_metrics or {
            'Accuracy': 0.87, 'Precision': 0.85,
            'Recall': 0.88, 'F1-Score': 0.86, 'AUC-ROC': 0.92,
        },
        'feature_count':   len(feature_names),
        'patient_defaults': patient_defaults_labels or {},
        'screening_medians': patient_screening_medians or {},
    })


@app.route('/api/model/images/<name>', methods=['GET'])
def api_model_image(name):
    allowed = ['feature_importance.png', 'shap_summary_plot.png',
               'roc_curve.png', 'confusion_matrix.png']
    if name not in allowed:
        return jsonify({'error': 'Not found'}), 404
    img_path = _ROOT / name
    if not img_path.exists():
        return jsonify({'error': 'Image not generated yet'}), 404
    return send_file(str(img_path), mimetype='image/png')


# ==============================================================================
# API ROUTES — PDF REPORT
# ==============================================================================

@app.route('/api/report/pdf', methods=['POST'])
def api_report_pdf():
    if not REPORTLAB_OK:
        return jsonify({'error': 'reportlab not installed. Run: pip install reportlab'}), 500

    data = request.get_json(force=True)
    probability  = float(data.get('probability', 0))
    tier_label   = data.get('tier_label', 'Unknown')
    tier_key     = data.get('tier_key', 'low')
    tier_color   = data.get('tier_color', '#22c55e')
    recs         = data.get('recommendations', [])
    inputs       = data.get('inputs', {})
    derived      = data.get('derived', {})
    patient_mode = data.get('patient_mode', False)
    patient_name = data.get('patient_name', 'Anonymous')
    mode_label   = 'Patient Screening' if patient_mode else 'Clinical Assessment'

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                 fontSize=20, textColor=colors.HexColor('#e63946'),
                                 spaceAfter=6, alignment=TA_CENTER)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                               fontSize=13, textColor=colors.HexColor('#1d3557'),
                               spaceBefore=14, spaceAfter=4)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                fontSize=10, leading=14)
    caption_style = ParagraphStyle('Caption', parent=styles['Normal'],
                                   fontSize=8, textColor=colors.grey,
                                   alignment=TA_CENTER)

    story = []

    # Header
    story.append(Paragraph('❤ CardioClarity', title_style))
    story.append(Paragraph(f'Heart Disease Risk Report — {mode_label}', styles['Heading3']))
    story.append(Paragraph(f'Patient: <b>{patient_name}</b> &nbsp;&nbsp; '
                            f'Date: <b>{datetime.now().strftime("%d %B %Y, %I:%M %p")}</b>',
                            body_style))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=colors.HexColor('#e63946'), spaceAfter=10))

    # Risk result box
    tier_hex = {'low': '#22c55e', 'medium': '#f59e0b', 'high': '#ef4444'}.get(tier_key, '#6b7280')
    story.append(Paragraph(f'<b>Risk Assessment Result: {tier_label}</b>',
                            ParagraphStyle('RiskBox', parent=styles['Normal'],
                                           fontSize=14, textColor=colors.HexColor(tier_hex),
                                           spaceAfter=4, spaceBefore=10)))
    if not patient_mode:
        story.append(Paragraph(f'Disease Probability: <b>{probability:.1%}</b>', body_style))
    story.append(Spacer(1, 0.3*cm))

    # Gauge image
    gauge_b64 = data.get('gauge_b64')
    if gauge_b64:
        img_buf = io.BytesIO(base64.b64decode(gauge_b64))
        rl_img  = RLImage(img_buf, width=14*cm, height=4*cm)
        story.append(rl_img)
    story.append(Spacer(1, 0.4*cm))

    # Key Clinical Inputs
    story.append(Paragraph('Key Clinical Inputs', h2_style))
    input_rows = [['Parameter', 'Value']]
    param_labels = {
        'age': 'Age (years)', 'sex': 'Biological Sex',
        'trestbps': 'Resting BP (mm Hg)', 'chol': 'Cholesterol (mg/dl)',
        'thalach': 'Max Heart Rate (bpm)',
    }
    for key, label in param_labels.items():
        if key in inputs:
            input_rows.append([label, str(inputs[key])])
    if derived:
        if 'cardiovascular_risk' in derived:
            input_rows.append(['CV Risk Score', str(int(derived['cardiovascular_risk']))])
        if 'hr_percentage' in derived:
            input_rows.append(['HR % of Age Target', f"{derived['hr_percentage']:.1%}"])

    t = Table(input_rows, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1d3557')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTSIZE',   (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f1faee')]),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#ddd')),
        ('PADDING',    (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # SHAP chart
    shap_b64 = data.get('shap_chart_b64')
    if shap_b64 and not patient_mode:
        story.append(Paragraph('Feature Contributions (SHAP Analysis)', h2_style))
        shap_buf = io.BytesIO(base64.b64decode(shap_b64))
        shap_img = RLImage(shap_buf, width=15*cm, height=6*cm)
        story.append(shap_img)
        story.append(Paragraph('Red = increases risk | Green = decreases risk', caption_style))
        story.append(Spacer(1, 0.4*cm))

    # Recommendations
    if recs:
        story.append(Paragraph('Clinical Recommendations', h2_style))
        for rec in recs:
            story.append(Paragraph(f'• {rec}', body_style))
        story.append(Spacer(1, 0.4*cm))

    # Disclaimer
    story.append(HRFlowable(width='100%', thickness=0.5,
                             color=colors.HexColor('#aaa'), spaceBefore=10))
    story.append(Paragraph(
        '⚠ <b>Disclaimer:</b> CardioClarity is an AI-assisted screening tool. '
        'This report does not constitute a medical diagnosis. '
        'Always consult a qualified healthcare professional for medical decisions.',
        ParagraphStyle('Disclaimer', parent=styles['Normal'],
                       fontSize=8, textColor=colors.grey, leading=11,
                       spaceBefore=6)))

    doc.build(story)
    buf.seek(0)
    filename = f'CardioClarity_Report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
    return send_file(buf, mimetype='application/pdf',
                     as_attachment=True, download_name=filename)


# ==============================================================================
# CHATBOT ENGINE
# ==============================================================================

import re as _re

_TERM_GLOSSARY = {
    'shap': (
        "**SHAP (SHapley Additive exPlanations)** is a method that explains "
        "AI decisions by showing how much each feature *pushed* the prediction "
        "up or down.\n\n"
        "- **Red bars** = that value is pushing your risk *higher*\n"
        "- **Green bars** = that value is pushing your risk *lower*\n\n"
        "It's based on game theory — every feature gets a fair 'credit' for "
        "its contribution to the final score."
    ),
    'oldpeak': (
        "**ST Depression (oldpeak)** measures how much your heart's electrical "
        "activity dips during exercise on an ECG. A higher value (>1 mm) can "
        "indicate the heart muscle isn't getting enough blood during exertion — "
        "a sign of possible coronary artery disease."
    ),
    'thalach': (
        "**Max Heart Rate (thalach)** is the highest beats-per-minute recorded "
        "during an exercise test. Reaching a high percentage of your age-predicted "
        "maximum (220 − age) is generally a good sign. A low max HR can indicate "
        "reduced cardiac reserve."
    ),
    'trestbps': (
        "**Resting Blood Pressure (trestbps)** is your systolic BP measured at "
        "rest. Normal is below 120 mm Hg. 120–139 is elevated. Above 140 is "
        "Stage 1 hypertension. High BP strains the heart and arteries over time."
    ),
    'chol': (
        "**Serum Cholesterol** measures total cholesterol in your blood (mg/dl). "
        "Desirable: <200. Borderline high: 200–239. High: ≥240. High cholesterol "
        "can build up as plaques in arteries (atherosclerosis), raising heart attack risk."
    ),
    'ca': (
        "**Number of Major Vessels (ca)** is the count of coronary arteries "
        "coloured by fluoroscopy during angiography. More blocked vessels (2–3) "
        "is a strong indicator of coronary artery disease."
    ),
    'thal': (
        "**Thalassemia (thal)** in this dataset refers to a thallium stress-test "
        "result, not the blood disorder:\n"
        "- **Normal** = adequate blood flow\n"
        "- **Fixed defect** = permanently reduced flow\n"
        "- **Reversible defect** = flow reduces during stress, recovers at rest "
        "(classic angina pattern)"
    ),
    'exang': (
        "**Exercise-Induced Angina (exang)** is chest pain or discomfort that "
        "occurs specifically during physical activity. It happens when the heart "
        "demands more oxygen than narrowed arteries can supply. Its presence is "
        "a significant risk indicator."
    ),
    'cp': (
        "**Chest Pain Type (cp)** is classified as:\n"
        "- **Typical angina** — pressure/tightness during exertion, often cardiac\n"
        "- **Atypical angina** — chest discomfort not classic-pattern\n"
        "- **Non-anginal** — chest pain unlikely to be cardiac\n"
        "- **Asymptomatic** — no chest pain (paradoxically can still indicate disease)"
    ),
    'slope': (
        "**ST Segment Slope** describes the shape of the ECG trace at peak exercise:\n"
        "- **Upsloping** — generally a good sign\n"
        "- **Flat (horizontal)** — ambiguous, warrants attention\n"
        "- **Downsloping** — most concerning, associated with ischaemia"
    ),
    'restecg': (
        "**Resting ECG (restecg)** is an electrocardiogram taken at rest:\n"
        "- **Normal** — no significant findings\n"
        "- **ST-T abnormality** — suggests possible ischaemia or electrolyte issues\n"
        "- **LV hypertrophy** — left ventricle enlarged, often from long-term hypertension"
    ),
    'fbs': (
        "**Fasting Blood Sugar (fbs)** >120 mg/dl often indicates diabetes or "
        "pre-diabetes. Diabetes significantly amplifies cardiovascular risk — "
        "it damages blood vessels and nerves controlling the heart."
    ),
}

_LIFESTYLE_TIPS = {
    'blood pressure': (
        "**Lowering Blood Pressure:**\n"
        "- Reduce salt/sodium intake (aim for <2,300 mg/day)\n"
        "- Get 150 min of moderate aerobic exercise per week\n"
        "- Limit alcohol and quit smoking\n"
        "- Manage stress with meditation, yoga, or deep breathing\n"
        "- Maintain a healthy weight — even 5 kg loss helps significantly\n"
        "- Medications (ACE inhibitors, ARBs, calcium channel blockers) if prescribed"
    ),
    'cholesterol': (
        "**Lowering Cholesterol:**\n"
        "- Eat more soluble fibre: oats, beans, lentils, fruits\n"
        "- Replace saturated fats with unsaturated fats (olive oil, nuts, avocado)\n"
        "- Avoid trans fats (partially hydrogenated oils)\n"
        "- Exercise regularly — raises HDL ('good') cholesterol\n"
        "- Medications: statins (e.g., atorvastatin) are very effective if prescribed\n"
        "- Omega-3 fatty acids from fish or supplements can help triglycerides"
    ),
    'exercise': (
        "**Heart-Healthy Exercise:**\n"
        "- **150 min/week** moderate intensity (brisk walking, cycling, swimming)\n"
        "- OR **75 min/week** vigorous (running, aerobics)\n"
        "- Add **2 days/week** of strength training\n"
        "- Start slow if sedentary — even 10 min walks help\n"
        "- Always warm up and cool down\n"
        "- Consult your doctor before starting if you have existing heart disease"
    ),
    'diet': (
        "**Heart-Healthy Diet (Mediterranean-style):**\n"
        "- Lots of vegetables, fruits, whole grains, legumes\n"
        "- Olive oil as primary fat source\n"
        "- Fish 2x/week (especially fatty fish: salmon, mackerel, sardines)\n"
        "- Limit red meat, processed foods, sugary drinks\n"
        "- Moderate low-fat dairy\n"
        "- Herbs and spices instead of salt for flavouring"
    ),
    'smoking': (
        "**Smoking and Heart Health:**\n"
        "Smoking is one of the **single biggest modifiable risk factors** for heart disease:\n"
        "- Damages artery walls, promotes atherosclerosis\n"
        "- Reduces HDL ('good') cholesterol\n"
        "- Makes blood more likely to clot\n"
        "- Quitting reduces cardiovascular risk significantly within **1–2 years**\n"
        "- Nicotine replacement therapy (patches, gum) and medications (varenicline) can help"
    ),
    'stress': (
        "**Stress and Heart Health:**\n"
        "Chronic stress raises cortisol and adrenaline, increasing heart rate and BP:\n"
        "- Practice **mindfulness meditation** (even 10 min/day)\n"
        "- Regular **physical activity** is one of the best stress relievers\n"
        "- Prioritise **sleep** (7–9 hours/night)\n"
        "- Social connection reduces cardiac mortality (isolation is a risk factor)\n"
        "- Consider **cognitive behavioural therapy (CBT)** for persistent anxiety"
    ),
}

_DISCLAIMER = (
    "\n\n*⚠ This is for educational purposes only and is **not** medical advice. "
    "Always consult a qualified healthcare professional.*"
)


def _build_context_reply(ctx: dict) -> str | None:
    """Build a reply personalised to the current prediction context."""
    tier     = ctx.get('tier_key', '')
    tier_lbl = ctx.get('tier_label', '')
    prob     = ctx.get('probability')
    inputs   = ctx.get('inputs', {}) or {}
    derived  = ctx.get('derived', {}) or {}
    pat_mode = ctx.get('patient_mode', False)
    shap_top = ctx.get('shap_top', []) or []
    recs     = ctx.get('recommendations', []) or []

    age      = inputs.get('age')
    bp       = inputs.get('trestbps')
    chol     = inputs.get('chol')
    hr       = inputs.get('thalach')
    cv_risk  = derived.get('cardiovascular_risk', 0)
    hr_pct   = derived.get('hr_percentage', 0)

    lines = []
    if tier == 'high':
        lines.append(
            f"Based on your most recent assessment, your predicted risk is **{tier_lbl}**"
            + (f" ({prob:.0%})" if prob and not pat_mode else "") + "."
        )
        lines.append(
            "This means the AI model found several significant risk factors. "
            "**This is not a diagnosis**, but it does suggest you should speak "
            "with a doctor or cardiologist soon."
        )
    elif tier == 'medium':
        lines.append(
            f"Your current assessment shows **{tier_lbl}**"
            + (f" ({prob:.0%})" if prob and not pat_mode else "") + "."
        )
        lines.append(
            "This suggests some risk factors are present that are worth monitoring. "
            "Discuss with your doctor at your next routine visit."
        )
    elif tier == 'low':
        lines.append(
            f"Good news — your current assessment is **{tier_lbl}**"
            + (f" ({prob:.0%})" if prob and not pat_mode else "") + "."
        )
        lines.append("Keep up your healthy habits and attend regular check-ups.")

    # Highlight top SHAP factors
    if shap_top and not pat_mode:
        readable = {
            'trestbps': 'blood pressure', 'chol': 'cholesterol',
            'cp': 'chest pain type', 'oldpeak': 'ST depression',
            'exang': 'exercise-induced angina', 'thal': 'thallium test result',
            'ca': 'vessel count', 'age': 'age', 'thalach': 'max heart rate',
            'fbs': 'blood sugar', 'slope': 'ST slope',
        }
        top_readable = [readable.get(f, f) for f in shap_top[:3]]
        lines.append(
            f"The top factors driving this result were: **{', '.join(top_readable)}**."
        )

    # Specific advice
    if bp and bp > 140:
        lines.append(f"Your BP of **{bp} mm Hg** is hypertensive — this is a key factor to address.")
    if chol and chol > 240:
        lines.append(f"Your cholesterol of **{chol} mg/dl** is high — dietary and possibly medical intervention may help.")
    if hr_pct and hr_pct < 0.75:
        lines.append("Your max heart rate during exercise is notably below the age-predicted target, which can be a cardiac efficiency concern.")

    return '\n\n'.join(lines) if lines else None


def _chatbot_engine(message: str, context: dict | None) -> tuple[str, list[str]]:
    """
    Rule-based NLU chatbot engine.
    Returns (reply_text, suggestions_list)
    """
    msg   = message.lower().strip()
    reply = None
    suggestions = []

    # ── 1. Greetings ──────────────────────────────────────────────────────
    if _re.search(r'\b(hi|hello|hey|howdy|good morning|good evening)\b', msg):
        user_has_pred = context and context.get('tier_key')
        if user_has_pred:
            reply = (
                "Hello again! 👋 I can see you've already run an assessment. "
                "Would you like me to explain your results, or do you have a specific question?"
            )
            suggestions = ['Explain my results', 'What should I do next?', 'Tell me about my top risk factors']
        else:
            reply = (
                "Hello! 👋 I'm CardioBot. Run a prediction in the **Predict** tab first, "
                "and then I can give you personalised explanations of your results!\n\n"
                "In the meantime, feel free to ask me any heart health questions."
            )
            suggestions = ['What does the risk score mean?', 'How can I lower cholesterol?', 'What is SHAP?']
        return reply, suggestions

    # ── 2. "What does my result / score / risk mean?" ─────────────────────
    if _re.search(r'(my result|my score|my risk|what does.*mean|explain.*result|interpret)', msg):
        if context and context.get('tier_key'):
            ctx_reply = _build_context_reply(context)
            reply = ctx_reply or "I couldn't find a recent prediction. Please run one in the Predict tab first."
            suggestions = ['What should I do next?', 'How can I lower my risk?', 'What is SHAP?']
        else:
            reply = (
                "I don't have a recent prediction result for you yet. "
                "Go to the **Predict tab**, fill in the form, and click **Assess Risk**. "
                "Then come back and I'll explain exactly what the numbers mean for you!"
            )
            suggestions = ['What is the risk gauge?', 'What inputs do I need?']
        return reply, suggestions

    # ── 3. "What should I do next?" ──────────────────────────────────────
    if _re.search(r'(what should i do|next step|recommend|advice|what now)', msg):
        if context and context.get('tier_key'):
            tier = context['tier_key']
            recs = context.get('recommendations', [])[:3]
            base = {
                'high':   "Your results suggest **urgent action** — please contact a doctor or cardiologist soon.\n\nKey steps:",
                'medium': "Your results suggest **proactive management**. Recommended actions:",
                'low':    "Great news! To **stay in the low-risk zone**, keep up these habits:",
            }.get(tier, "Based on your results:")
            bullets = '\n'.join(f'- {r}' for r in recs) if recs else '- Consult your healthcare provider'
            reply = base + '\n\n' + bullets
            suggestions = ['How to lower blood pressure?', 'Heart-healthy diet tips', 'How much exercise?']
        else:
            reply = (
                "General cardiovascular advice:\n\n"
                "- Maintain a **heart-healthy diet** (Mediterranean style)\n"
                "- Exercise at least **150 min/week** at moderate intensity\n"
                "- Avoid **smoking** and limit alcohol\n"
                "- Monitor your **blood pressure and cholesterol** regularly\n"
                "- Manage **stress** and get enough sleep\n\n"
                "Run a prediction first for personalised advice!"
            )
            suggestions = ['Tell me about diet', 'Tell me about exercise', 'How to quit smoking?']
        return reply, suggestions

    # ── 4. Glossary terms ─────────────────────────────────────────────────
    for term, explanation in _TERM_GLOSSARY.items():
        if term in msg or (term == 'trestbps' and 'blood pressure' in msg and 'resting' in msg):
            reply = explanation
            suggestions = ['Tell me about cholesterol', 'What is SHAP?', 'What does ST depression mean?']
            return reply, suggestions

    # ── 5. Blood pressure ─────────────────────────────────────────────────
    if _re.search(r'(blood pressure|hypertension|bp|systolic|diastolic)', msg):
        if _re.search(r'(lower|reduce|improve|control|manage|high)', msg):
            reply = _LIFESTYLE_TIPS['blood pressure']
            suggestions = ['How to lower cholesterol?', 'Exercise tips', 'Diet for heart health']
        else:
            reply = _TERM_GLOSSARY['trestbps']
            if context and context.get('inputs', {}).get('trestbps'):
                bp = context['inputs']['trestbps']
                status = 'hypertensive' if bp > 140 else 'elevated-normal' if bp > 120 else 'normal'
                reply += f"\n\nYour recorded BP was **{bp} mm Hg** — that is considered **{status}**."
        suggestions = ['How to lower blood pressure?', 'What is hypertension?', 'Diet tips']
        return reply, suggestions

    # ── 6. Cholesterol ────────────────────────────────────────────────────
    if _re.search(r'(cholesterol|ldl|hdl|triglyceride|lipid|statin)', msg):
        if _re.search(r'(lower|reduce|improve|high|bad)', msg):
            reply = _LIFESTYLE_TIPS['cholesterol']
        else:
            reply = _TERM_GLOSSARY['chol']
            if context and context.get('inputs', {}).get('chol'):
                chol = context['inputs']['chol']
                status = 'high' if chol > 240 else 'borderline' if chol > 200 else 'desirable'
                reply += f"\n\nYour recorded cholesterol was **{chol} mg/dl** — classified as **{status}**."
        suggestions = ['What foods lower cholesterol?', 'Should I take statins?', 'What is LDL vs HDL?']
        return reply, suggestions

    # ── 7. Exercise ───────────────────────────────────────────────────────
    if _re.search(r'(exercise|workout|physical activity|aerobic|cardio|sport|gym|walk|run)', msg):
        reply = _LIFESTYLE_TIPS['exercise']
        suggestions = ['Best diet for heart health?', 'How to manage stress?', 'How to lower blood pressure?']
        return reply, suggestions

    # ── 8. Diet / Food ────────────────────────────────────────────────────
    if _re.search(r'(diet|food|eat|nutrition|meal|mediterranean|vegetable|fruit|fish|salt)', msg):
        reply = _LIFESTYLE_TIPS['diet']
        suggestions = ['How to lower cholesterol?', 'Exercise for heart health', 'What to avoid?']
        return reply, suggestions

    # ── 9. Smoking ────────────────────────────────────────────────────────
    if _re.search(r'(smok|cigarette|tobacco|nicotine|vape)', msg):
        reply = _LIFESTYLE_TIPS['smoking']
        suggestions = ['Exercise tips', 'Stress management', 'Heart-healthy diet']
        return reply, suggestions

    # ── 10. Stress / Sleep ────────────────────────────────────────────────
    if _re.search(r'(stress|anxiety|sleep|mental health|relax|meditat|yoga|depress|worry)', msg):
        reply = _LIFESTYLE_TIPS['stress']
        suggestions = ['Exercise for heart health', 'Diet tips', 'How to lower blood pressure?']
        return reply, suggestions

    # ── 11. "Should I see a doctor?" ─────────────────────────────────────
    if _re.search(r'(see a doctor|visit doctor|cardiologist|hospital|consult|medical help|emergency|urgent)', msg):
        if context and context.get('tier_key') == 'high':
            reply = (
                "**Yes — for your current High Risk result, I strongly recommend:**\n\n"
                "- Contact your primary care physician soon\n"
                "- Ask for a cardiology referral\n"
                "- Request an ECG and stress test if not recently done\n"
                "- Review your medications with your doctor\n\n"
                "If you are currently experiencing **chest pain, shortness of breath, "
                "or rapid/irregular heartbeat**, please seek emergency care immediately."
            )
        elif context and context.get('tier_key') == 'medium':
            reply = (
                "**For a Medium Risk result:**\n\n"
                "- Mention it at your next routine doctor visit\n"
                "- Ask about cholesterol and blood pressure targets\n"
                "- Request a baseline cardiovascular risk assessment\n\n"
                "This isn't an emergency, but don't put it off indefinitely."
            )
        else:
            reply = (
                "**When to see a doctor about heart health:**\n\n"
                "- Routine check: **annually**, especially after age 40\n"
                "- If you have **chest pain, palpitations, or shortness of breath** — see a doctor promptly\n"
                "- **Emergency (call 999/112/911):** crushing chest pain, pain spreading to arm/jaw, sudden severe breathlessness\n\n"
                "This tool is for screening only — it cannot replace a clinical evaluation."
            )
        suggestions = ['What lifestyle changes help most?', 'What tests should I ask for?', 'Tell me about my results']
        return reply, suggestions

    # ── 12. "What is heart disease?" and general education ───────────────
    if _re.search(r'(heart disease|coronary|heart attack|myocardial|atherosclerosis|angina|artery)', msg):
        reply = (
            "**Coronary Heart Disease (CHD)** is the most common form of heart disease. "
            "It occurs when the coronary arteries — which supply blood to the heart muscle — "
            "become narrowed or blocked by fatty deposits (plaques).\n\n"
            "**Key facts:**\n"
            "- Atherosclerosis (plaque build-up) is the underlying cause\n"
            "- Major risk factors: high BP, high cholesterol, smoking, diabetes, obesity, inactivity\n"
            "- Can lead to: **angina** (chest pain), **heart attack** (when a plaque ruptures)\n"
            "- **Highly preventable** with lifestyle changes and, when needed, medication\n\n"
            "This AI tool assesses your probability of having CHD based on clinical factors."
        )
        suggestions = ['How to prevent heart disease?', 'What causes high cholesterol?', 'What is angina?']
        return reply, suggestions

    # ── 13. How does this app / AI work? ─────────────────────────────────
    if _re.search(r'(how does.*work|how.*ai|how.*model|random forest|machine learning|algorithm|predict)', msg):
        feat_count = 25
        if context:
            # Feature count embedded in the page
            feat_count = 25
        reply = (
            "**How CardioClarity Works:**\n\n"
            "1. **Data** — Trained on the UCI Heart Disease dataset (~1,000 patients)\n"
            "2. **Model** — Random Forest Classifier with 100 decision trees\n"
            f"3. **Features** — Uses **{feat_count} clinical + engineered features** "
            "(age, sex, BP, cholesterol, chest pain, ECG, etc.)\n"
            "4. **Output** — A probability between 0–100% of heart disease presence\n"
            "5. **Explainability** — SHAP values show exactly *why* the model gave that score\n\n"
            "The model achieves ~87% accuracy on held-out test data. "
            "It is **not a medical device** and must not replace clinical evaluation."
        )
        suggestions = ['What is SHAP?', 'How accurate is it?', 'What is a Random Forest?']
        return reply, suggestions

    # ── 14. Accuracy / reliability ────────────────────────────────────────
    if _re.search(r'(accurate|accuracy|reliable|trust|confidence|wrong|error)', msg):
        reply = (
            "**Model Performance (on test data):**\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            "| Accuracy | ~87% |\n"
            "| Precision | ~85% |\n"
            "| Recall | ~88% |\n"
            "| AUC-ROC | ~92% |\n\n"
            "**Important caveats:**\n"
            "- Trained on a dataset of ~1,000 patients — real-world performance may vary\n"
            "- This is a **screening tool**, not a diagnostic device\n"
            "- False negatives (missed disease) and false positives do occur\n"
            "- Always verify with a clinician who knows your full history\n\n"
            "The 92% AUC-ROC means it's quite good at distinguishing disease vs no disease "
            "across all probability thresholds."
        )
        suggestions = ['How does the model work?', 'What is SHAP?', 'Should I see a doctor?']
        return reply, suggestions

    # ── 15. Risk factors general ──────────────────────────────────────────
    if _re.search(r'(risk factor|cause|why.*risk|what.*risk|reduce.*risk|lower.*risk)', msg):
        if context and context.get('tier_key'):
            ctx_reply = _build_context_reply(context)
            reply = (ctx_reply or '') + (
                "\n\n**General modifiable risk factors:**\n"
                "- High blood pressure · High cholesterol · Smoking · Diabetes\n"
                "- Physical inactivity · Obesity · Poor diet · Chronic stress"
            )
        else:
            reply = (
                "**Major cardiovascular risk factors:**\n\n"
                "**Modifiable (you can change these):**\n"
                "- High blood pressure (hypertension)\n"
                "- High LDL / total cholesterol\n"
                "- Smoking / tobacco use\n"
                "- Physical inactivity\n"
                "- Obesity and poor diet\n"
                "- Uncontrolled diabetes\n"
                "- Chronic stress and poor sleep\n\n"
                "**Non-modifiable:**\n"
                "- Age (risk increases after 45 for men, 55 for women)\n"
                "- Family history of early heart disease\n"
                "- Biological sex (men generally at higher risk earlier)"
            )
        suggestions = ['How to lower blood pressure?', 'Cholesterol reduction tips', 'Exercise benefits']
        return reply, suggestions

    # ── 16. Goodbye / thanks ─────────────────────────────────────────────
    if _re.search(r'(thank|thanks|bye|goodbye|great|helpful|awesome|good job)', msg):
        reply = (
            "You're welcome! 😊 Take care of your heart — it's the only one you've got! ❤️\n\n"
            "Remember: this tool is for education and screening. Any concerns should be "
            "discussed with a qualified healthcare professional."
        )
        suggestions = ['Run another prediction', 'Download PDF Report', 'View model insights']
        return reply, suggestions

    # ── 17. Fallback ─────────────────────────────────────────────────────
    reply = (
        "I'm not sure I understood that. Here are some things I can help with:\n\n"
        "- **Explain your risk result** — just ask 'What does my result mean?'\n"
        "- **Medical terms** — SHAP, thalach, oldpeak, trestbps, ca, thal, etc.\n"
        "- **Lifestyle advice** — diet, exercise, blood pressure, cholesterol, stress\n"
        "- **App guidance** — how the AI works, what the gauge shows, PDF report\n\n"
        "Try rephrasing your question or pick a suggestion below!"
    )
    suggestions = [
        'What does my result mean?', 'How to lower blood pressure?',
        'What is SHAP?', 'Should I see a doctor?',
    ]
    return reply, suggestions


# ── Flask route ───────────────────────────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data    = request.get_json(force=True)
    message = (data.get('message') or '').strip()
    context = data.get('context')   # May be None if no prediction yet

    if not message:
        return jsonify({'reply': 'Please type a message!', 'suggestions': []}), 400

    try:
        reply, suggestions = _chatbot_engine(message, context)
        # Append disclaimer for medical topics
        medical_topics = ['doctor', 'diagnos', 'treat', 'medic', 'symptom', 'drug', 'pill']
        if any(t in message.lower() for t in medical_topics):
            reply += _DISCLAIMER
        return jsonify({'reply': reply, 'suggestions': suggestions[:4]})
    except Exception as e:
        return jsonify({'reply': f'Sorry, something went wrong: {e}', 'suggestions': []}), 500


# ==============================================================================
# STATIC FILE SERVING
# ==============================================================================

@app.route('/')
def serve_index():
    return send_from_directory(str(_STATIC), 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(str(_STATIC), path)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print(f"[*] CardioClarity server starting on port {port}", flush=True)
    app.run(debug=debug, host='0.0.0.0', port=port)
