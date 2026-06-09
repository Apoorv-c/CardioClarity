# app.py — CardioClarity
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path
from datetime import datetime

from utils.visualization import create_shap_force_plot, shap_positive_class_row
from views.doctor_views import show_doctor_extra_panel
from views.patient_view import show_patient_friendly_result

# ============================================================================
# CONSTANTS
# ============================================================================

# Doctor mode — full-precision thresholds
RISK_LOW_BELOW    = 0.35
RISK_MEDIUM_BELOW = 0.65

# Patient mode — conservative thresholds (flag risk earlier; safer for screening)
PAT_RISK_LOW_BELOW    = 0.25
PAT_RISK_MEDIUM_BELOW = 0.50

_ROOT = Path(__file__).resolve().parent

# Clinical fields a patient CANNOT answer — filled with training-data defaults
PATIENT_HIDDEN_FIELDS = ['restecg', 'oldpeak', 'slope', 'ca', 'thal']

# Hardcoded fallback defaults (used ONLY if artifact doesn't have saved defaults)
# Values chosen from UCI population medians/modes — slightly conservative
_FALLBACK_DEFAULTS = {
    'restecg': 'normal',          # mode across dataset
    'oldpeak':  1.0,              # slightly above median — conservative
    'slope':    'flat',           # common in disease-positive cases — conservative
    'ca':       0.0,              # mode (most common = 0 vessels blocked)
    'thal':     'normal',         # mode
}


# ============================================================================
# LOAD MODEL ARTIFACTS
# ============================================================================

@st.cache_resource
def load_model():
    """Load trained model and all supporting artifacts once."""
    model         = joblib.load(_ROOT / 'random_forest_model.pkl')
    scaler        = joblib.load(_ROOT / 'scaler.pkl')
    feature_names = joblib.load(_ROOT / 'feature_names.pkl')
    label_encoders= joblib.load(_ROOT / 'label_encoders.pkl')
    shap_explainer= shap.TreeExplainer(model)

    model_metrics            = None
    patient_defaults_encoded = None
    patient_defaults_labels  = None
    patient_screening_medians = None

    artifacts_path = _ROOT / 'heart_disease_model_artifacts.pkl'
    if artifacts_path.exists():
        try:
            arts = joblib.load(artifacts_path)
            model_metrics             = arts.get('model_metrics', None)
            patient_defaults_encoded  = arts.get('patient_defaults_encoded', None)
            patient_defaults_labels   = arts.get('patient_defaults_labels', None)
            patient_screening_medians = arts.get('patient_screening_medians', None)
        except Exception:
            pass

    return (model, scaler, feature_names, label_encoders, shap_explainer,
            model_metrics, patient_defaults_encoded, patient_defaults_labels,
            patient_screening_medians)


(model, scaler, feature_names, label_encoders, shap_explainer,
 model_metrics, patient_defaults_encoded, patient_defaults_labels,
 patient_screening_medians) = load_model()


# ============================================================================
# PATIENT DEFAULTS — resolve from artifact or fallback
# ============================================================================

def get_patient_screening_medians():
    """Training-data medians for BP and cholesterol when the patient selects 'I don't know'."""
    if patient_screening_medians:
        return {
            'trestbps': float(patient_screening_medians.get('trestbps', 130)),
            'chol': float(patient_screening_medians.get('chol', 220)),
        }
    return {'trestbps': 130.0, 'chol': 220.0}


def get_patient_defaults():
    """
    Return encoded numeric defaults for the 5 hidden clinical fields.
    Prefers values saved from training data; falls back to _FALLBACK_DEFAULTS.
    """
    if patient_defaults_encoded:
        return dict(patient_defaults_encoded)

    # Encode the fallback string values using label_encoders
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


# ============================================================================
# FEATURE ENGINEERING  (must mirror train_model.py exactly)
# ============================================================================

def compute_derived_metrics(age, trestbps, chol, thalach, oldpeak, ca):
    age_group = (0 if age <= 40 else 1 if age <= 50 else 2 if age <= 60 else 3 if age <= 70 else 4)
    age_over_50             = 1 if age > 50 else 0
    bp_risk                 = 2 if trestbps > 140 else (1 if trestbps > 120 else 0)
    hypertension            = 1 if trestbps > 140 else 0
    chol_risk               = 2 if chol > 240 else (1 if chol > 200 else 0)
    high_cholesterol        = 1 if chol > 240 else 0
    max_hr_target           = 220 - age
    hr_percentage           = thalach / max_hr_target if max_hr_target > 0 else 0.0
    low_hr                  = 1 if hr_percentage < 0.85 else 0
    cardiovascular_risk     = (int(age > 50) + int(trestbps > 140) + int(chol > 240))
    significant_st_depression = 1 if oldpeak > 1 else 0
    multi_vessel            = 1 if ca >= 2 else 0

    return {
        'age_group': age_group, 'age_over_50': age_over_50,
        'bp_risk': bp_risk, 'hypertension': hypertension,
        'chol_risk': chol_risk, 'high_cholesterol': high_cholesterol,
        'max_hr_target': max_hr_target, 'hr_percentage': hr_percentage,
        'low_hr': low_hr, 'cardiovascular_risk': cardiovascular_risk,
        'significant_st_depression': significant_st_depression,
        'multi_vessel': multi_vessel,
    }


def create_feature_vector(age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal):
    """Build 25-feature input array in model-expected order."""
    d = compute_derived_metrics(age, trestbps, chol, thalach, oldpeak, ca)

    sex_val     = 1 if sex == "Male" else 0
    cp_val      = label_encoders['cp'].transform([cp])[0]
    fbs_val     = label_encoders['fbs'].transform([fbs])[0]
    restecg_val = label_encoders['restecg'].transform([restecg])[0]
    exang_val   = label_encoders['exang'].transform([exang])[0]
    slope_val   = label_encoders['slope'].transform([slope])[0]
    thal_val    = label_encoders['thal'].transform([thal])[0]

    features_dict = {
        'age': age, 'sex': sex_val, 'cp': cp_val, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs_val, 'restecg': restecg_val, 'thalach': thalach,
        'exang': exang_val, 'oldpeak': oldpeak, 'slope': slope_val, 'ca': ca,
        'thal': thal_val,
        **d,
    }
    feature_values = [features_dict[f] for f in feature_names]
    return np.array([feature_values]), d


def create_feature_vector_with_defaults(age, sex, cp_encoded, trestbps, chol,
                                        fbs, thalach, exang_encoded, defaults):
    """
    Build 25-feature vector for patient mode.
    cp and exang are already encoded integers.
    Hidden clinical fields (restecg, oldpeak, slope, ca, thal) come from defaults.
    """
    oldpeak = defaults.get('oldpeak', 0.0)
    ca      = defaults.get('ca',      0.0)
    d = compute_derived_metrics(age, trestbps, chol, thalach, oldpeak, ca)

    sex_val = 1 if sex == "Male" else 0
    fbs_val = label_encoders['fbs'].transform([fbs])[0]

    features_dict = {
        'age': age, 'sex': sex_val, 'cp': cp_encoded, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs_val,
        'restecg': defaults.get('restecg', 0.0),
        'thalach': thalach,
        'exang': exang_encoded,
        'oldpeak': oldpeak,
        'slope': defaults.get('slope', 0.0),
        'ca': ca,
        'thal': defaults.get('thal', 0.0),
        **d,
    }
    feature_values = [features_dict[f] for f in feature_names]
    return np.array([feature_values]), d


def run_prediction(input_features):
    """Scale and predict from a raw feature array. Returns (prediction, probability, shap_row)."""
    input_df     = pd.DataFrame(input_features, columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction   = int(model.predict(input_scaled)[0])
    probability  = float(model.predict_proba(input_scaled)[0][1])
    shap_out     = shap_explainer.shap_values(input_scaled)
    shap_row     = shap_positive_class_row(shap_out, len(feature_names))
    return prediction, probability, shap_row


def disease_risk_band(probability, patient_mode=False):
    """Map probability → (tier_key, tier_label, bar_color). Patient mode uses tighter thresholds."""
    low  = PAT_RISK_LOW_BELOW    if patient_mode else RISK_LOW_BELOW
    mid  = PAT_RISK_MEDIUM_BELOW if patient_mode else RISK_MEDIUM_BELOW
    if probability < low:
        return "low",    "Low risk",    "#22c55e"
    if probability < mid:
        return "medium", "Medium risk", "#f59e0b"
    return "high",   "High risk",   "#ef4444"


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def show_input_warnings(age, trestbps, chol, thalach):
    max_hr = 220 - age
    for cond, msg in [
        (trestbps < 90,      f"Resting BP of **{trestbps} mm Hg** is unusually low — please verify."),
        (trestbps > 180,     f"Resting BP of **{trestbps} mm Hg** is critically elevated. Confirm the reading."),
        (chol < 120,         f"Cholesterol of **{chol} mg/dl** is extremely low — possible data-entry error."),
        (chol > 500,         f"Cholesterol of **{chol} mg/dl** is very high — confirm with patient / lab."),
        (thalach > max_hr,   f"Max heart rate **{thalach} bpm** exceeds the age-adjusted maximum "
                             f"(**{max_hr} bpm** = 220 − {age}). Please double-check."),
    ]:
        if cond:
            st.warning(f"⚠️ {msg}")


# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
    defaults = {
        "show_prediction": False,
        "prediction_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def log_prediction(age, sex, trestbps, chol, probability, tier_label, mode, *, hide_probability=False):
    st.session_state.prediction_history.append({
        "Time":         datetime.now().strftime("%H:%M:%S"),
        "Mode":         mode,
        "Age":          age,
        "Sex":          sex,
        "BP (mm Hg)":   trestbps,
        "Chol (mg/dl)": chol,
        "Probability":  "(not shown — screening mode)" if hide_probability else f"{probability:.1%}",
        "Risk Band":    tier_label,
    })


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="CardioClarity — Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
)
init_session_state()

st.title("❤️ CardioClarity — Heart Disease Risk Assessment")
st.markdown("*AI-Powered Explainable Heart Disease Prediction*")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("👤 Select Your Role")
role = st.sidebar.radio(
    "Role",
    ["🧑‍⚕️ Doctor", "👨‍👩 Patient"],
    label_visibility="collapsed",
    help="Doctor: full clinical form + SHAP details. Patient: simplified screening form.",
)
st.sidebar.markdown("---")
if role == "👨‍👩 Patient":
    st.sidebar.warning(
        "**Patient screening mode**\n\n"
        "• **No exact percentage** is shown — only **lower / moderate / higher** concern bands.\n"
        "• Cutoffs are **stricter** than clinician view so more people are encouraged to follow up.\n"
        "• **Seven** kinds of hospital-grade measurements are **not** collected here; "
        "typical values stand in for them.\n\n"
        "**Not a diagnosis.** See a doctor for a full assessment."
    )
else:
    st.sidebar.info(
        "Doctor mode uses all 25 features with full probability output and SHAP explanations."
    )
st.sidebar.markdown("---")
st.sidebar.success(f"✅ Model ready · {len(feature_names)} clinical features")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_predict, tab_whatif, tab_history, tab_global = st.tabs([
    "🔍 Predict",
    "🔬 What-If Explorer",
    "📋 History",
    "📊 Model Insights",
])


# ============================================================================
# TAB 1: PREDICTION
# ============================================================================

with tab_predict:

    # ── DOCTOR FORM ──────────────────────────────────────────────────────────
    if role == "🧑‍⚕️ Doctor":
        st.subheader("📋 Enter Clinical Information")

        col1, col2, col3 = st.columns(3)
        with col1:
            age      = st.number_input("Age", min_value=20, max_value=100, value=55, key="doc_age")
            sex      = st.selectbox("Sex", ["Male", "Female"], key="doc_sex")
            cp       = st.selectbox("Chest Pain Type",
                                    ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
                                    help="0: typical angina, 1: atypical, 2: non-anginal, 3: asymptomatic",
                                    key="doc_cp")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, key="doc_trestbps")

        with col2:
            chol    = st.number_input("Cholesterol (mg/dl)", 100, 600, 200, key="doc_chol")
            fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"], key="doc_fbs")
            restecg = st.selectbox("Resting ECG Results",
                                   ["normal", "st-t abnormality", "lv hypertrophy"],
                                   key="doc_restecg")
            thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150, key="doc_thalach")

        with col3:
            exang   = st.selectbox("Exercise Induced Angina", ["False", "True"], key="doc_exang")
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1, key="doc_oldpeak")
            slope   = st.selectbox("ST Segment Slope", ["upsloping", "flat", "downsloping"], key="doc_slope")
            ca      = st.slider("Number of Major Vessels (0–3)", 0, 3, 0,
                                help="Vessels coloured by fluoroscopy", key="doc_ca")
            thal    = st.selectbox("Thalassemia",
                                   ["normal", "fixed defect", "reversable defect"],
                                   key="doc_thal")

        show_input_warnings(age, trestbps, chol, thalach)

        b1, b2 = st.columns([4, 1])
        with b1:
            doc_predict = st.button("🔍 Predict Heart Disease Risk", type="primary",
                                    use_container_width=True, key="doc_btn")
        with b2:
            if st.button("Clear", key="doc_clear"):
                st.session_state.show_prediction = False

        if doc_predict:
            st.session_state.show_prediction = True

        if st.session_state.show_prediction and role == "🧑‍⚕️ Doctor":
            try:
                input_features, derived = create_feature_vector(
                    age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal)
                prediction, probability, shap_row = run_prediction(input_features)
                tier_key, tier_label, bar_color   = disease_risk_band(probability, patient_mode=False)
                log_prediction(age, sex, trestbps, chol, probability, tier_label, "Doctor")

                # ── Doctor results ────────────────────────────────────────────
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    if tier_key == "high":
                        st.error(f"**{tier_label}** of heart disease")
                        st.metric("Disease probability", f"{probability:.1%}")
                        st.warning("Discuss with a qualified healthcare provider.")
                    elif tier_key == "medium":
                        st.warning(f"**{tier_label}** of heart disease")
                        st.metric("Disease probability", f"{probability:.1%}")
                        st.info("Consider lifestyle review and routine follow-up.")
                    else:
                        st.success(f"**{tier_label}** of heart disease")
                        st.metric("Disease probability", f"{probability:.1%}")
                        st.info("Keep healthy habits; still attend regular check-ups.")
                    st.caption(
                        f"Bands: below **{RISK_LOW_BELOW:.0%}** = low · "
                        f"**{RISK_LOW_BELOW:.0%}**–**{RISK_MEDIUM_BELOW:.0%}** = medium · "
                        f"≥ **{RISK_MEDIUM_BELOW:.0%}** = high."
                    )
                with col2:
                    st.markdown(f"""
                    <div style="background:#f0f2f6;border-radius:10px;padding:20px;text-align:center;">
                        <h4>{tier_label}</h4>
                        <div style="background:#ddd;border-radius:10px;height:30px;">
                            <div style="background:{bar_color};width:{probability*100:.1f}%;
                                        border-radius:10px;height:30px;line-height:30px;color:white;">
                                {probability:.1%}
                            </div>
                        </div>
                        <p style="margin-top:8px;font-size:0.85em;color:#555;">
                            Binary label (0.5 cutoff): {'disease' if prediction==1 else 'no disease'}
                        </p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("##### Primary clinical measures")
                c1, c2 = st.columns(2)
                c1.metric("Resting blood pressure", f"{trestbps} mm Hg")
                c2.metric("Total cholesterol",       f"{chol} mg/dl")

                # Key factors
                st.markdown("---")
                st.subheader("💡 Key factors (BP & cholesterol first)")
                bp_note   = (f"elevated (hypertension, **{trestbps}** mm Hg)" if trestbps > 140
                             else f"elevated-normal (**{trestbps}** mm Hg)" if trestbps > 120
                             else f"favourable (**{trestbps}** mm Hg)")
                chol_note = (f"high (**{chol}** mg/dl)" if chol > 240
                             else f"borderline (**{chol}** mg/dl)" if chol > 200
                             else f"favourable (**{chol}** mg/dl)")
                st.markdown(f"**Resting BP** is {bp_note}. **Cholesterol** is {chol_note}.")

                factors = [
                    f"• **Resting BP:** {trestbps} mm Hg (tier {derived['bp_risk']}/2)",
                    f"• **Cholesterol:** {chol} mg/dl (tier {derived['chol_risk']}/2)",
                ]
                if age > 50:         factors.append("• Age > 50 years")
                if oldpeak > 1:      factors.append("• Significant ST depression (oldpeak)")
                if ca >= 2:          factors.append("• Multiple major vessels affected")
                if derived['hr_percentage'] < 0.85: factors.append("• HR below age-predicted target zone")

                if tier_key == "high":   st.warning("**High risk** — review all lines:")
                elif tier_key == "medium": st.info("**Medium risk** — key contributors:")
                else:                    st.success("**Low risk** — summary:")
                for f in factors[:6]: st.write(f)

                # SHAP
                st.markdown("#### SHAP contributions")
                st.caption(
                    "trestbps & chol always pinned. Red = pushes risk up, green = down."
                )
                with st.expander("Why can demographics (sex, age) appear in SHAP?"):
                    st.markdown("""
                    The model was trained with these as inputs. SHAP attributes output fairly
                    to each input the tree splits on. Statistical association ≠ individual causation.
                    For production deployment, fairness constraints or removal of protected
                    attributes should be considered.
                    """)
                create_shap_force_plot(
                    shap_row, input_features[0].tolist(), feature_names,
                    priority_features=("trestbps", "chol"), n_bars=6,
                )

                # Doctor panel
                show_doctor_extra_panel(
                    tier_key=tier_key, tier_label=tier_label, probability=probability,
                    trestbps=trestbps, chol=chol, age=age,
                    derived=derived, model_metrics=model_metrics,
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")
                with st.expander("Debug info"):
                    st.write(feature_names)
                    import traceback; st.code(traceback.format_exc())

    # ── PATIENT FORM ──────────────────────────────────────────────────────────
    else:
        st.subheader("🫀 Heart Health Screening")

        # Prominent disclaimer BEFORE the form
        st.info(
            "**📋 Before you start**\n\n"
            "You will answer **8 plain-language questions**. The app will **not** show "
            "a precise percentage — only a **lower**, **moderate**, or **higher** concern band, "
            "using **cautious** cutoffs for self-screening.\n\n"
            "**Seven** clinical details that normally require a clinic visit (ECG pattern, "
            "ST-segment measures, heart-vessel imaging, thalassemia-related findings, and "
            "similar) **are missing**. Those are filled with **median/mode values from the "
            "training dataset** — the same approach runs when you retrain the model.\n\n"
            "If you do not know your blood pressure or cholesterol, those are also filled "
            "with **dataset medians**, not your real measurements.\n\n"
            "⚠️ **This is not a diagnosis.** Any outcome here still requires a clinician for "
            "a complete evaluation."
        )

        st.markdown("---")
        st.markdown("### 📝 Answer these 8 questions about yourself")

        col1, col2 = st.columns(2)

        with col1:
            pat_age = st.number_input(
                "1️⃣ How old are you?",
                min_value=20, max_value=100, value=45,
                key="pat_age",
            )
            pat_sex = st.radio(
                "2️⃣ What is your biological sex?",
                ["Male", "Female"],
                key="pat_sex",
                help="Biological sex affects cardiovascular risk factors. Used for risk estimation only.",
            )
            pat_chest = st.selectbox(
                "3️⃣ Do you experience any chest discomfort?",
                [
                    "No chest discomfort",
                    "Pressure, tightness or squeezing (often during activity/stress)",
                    "Occasional discomfort — not always linked to activity",
                    "Sharp or stabbing chest pain",
                ],
                key="pat_chest",
                help="Choose the option closest to what you experience.",
            )
            pat_exercise_pain = st.radio(
                "4️⃣ Do you get chest pain or tightness specifically during physical activity or exercise?",
                ["No", "Yes"],
                key="pat_exang",
            )

        with col2:
            pat_bp_known = st.radio(
                "5️⃣ Do you know your resting blood pressure (top / systolic number)?",
                ["Yes, I know it", "No, I don't know"],
                key="pat_bp_known",
            )
            if pat_bp_known == "Yes, I know it":
                pat_trestbps = st.number_input(
                    "Enter your systolic blood pressure (mm Hg):",
                    min_value=80, max_value=200, value=120,
                    key="pat_trestbps",
                )
            else:
                _med = get_patient_screening_medians()
                pat_trestbps = int(round(_med['trestbps']))
                st.caption(
                    f"📌 Using the training dataset **median** systolic BP (**{pat_trestbps} mm Hg**) "
                    "because you chose 'I don't know.' This is **not** your real reading."
                )

            pat_chol_known = st.radio(
                "6️⃣ Do you know your most recent cholesterol level?",
                ["Yes, I know it", "No, I don't know"],
                key="pat_chol_known",
            )
            if pat_chol_known == "Yes, I know it":
                pat_chol = st.number_input(
                    "Enter your cholesterol (mg/dl):",
                    min_value=100, max_value=600, value=200,
                    key="pat_chol",
                )
            else:
                _med = get_patient_screening_medians()
                pat_chol = int(round(_med['chol']))
                st.caption(
                    f"📌 Using the training dataset **median** cholesterol (**{pat_chol} mg/dl**) "
                    "because you chose 'I don't know.' Ask your doctor for a real lab value."
                )

            pat_fbs = st.radio(
                "7️⃣ Has a doctor told you that you have diabetes or high blood sugar (pre-diabetes)?",
                ["No", "Yes"],
                key="pat_fbs",
            )
            pat_hr_default = int((220 - pat_age) * 0.85)
            pat_thalach = st.number_input(
                "8️⃣ What is the highest heart rate you reach during exercise? (beats per minute)\n"
                "*(Leave as is if you don't know — we've estimated it from your age)*",
                min_value=60, max_value=220, value=pat_hr_default,
                key="pat_thalach",
                help="Your heart rate at peak exertion — from a fitness tracker or exercise test.",
            )

        # Map patient answers to model values
        _cp_map = {
            "Pressure, tightness or squeezing (often during activity/stress)": "typical angina",
            "Occasional discomfort — not always linked to activity":           "atypical angina",
            "Sharp or stabbing chest pain":                                    "non-anginal",
            "No chest discomfort":                                             "asymptomatic",
        }
        _cp_label   = _cp_map[pat_chest]
        _cp_encoded = float(label_encoders['cp'].transform([_cp_label])[0])
        _exang_encoded = float(label_encoders['exang'].transform(
            ["True" if pat_exercise_pain == "Yes" else "False"])[0])
        _fbs_val = "True" if pat_fbs == "Yes" else "False"

        show_input_warnings(pat_age, pat_trestbps, pat_chol, pat_thalach)

        st.markdown("---")
        if st.button("🔍 Check My Heart Health Risk", type="primary", use_container_width=True, key="pat_btn"):
            st.session_state.show_prediction = True

        if st.button("Clear results", key="pat_clear"):
            st.session_state.show_prediction = False

        if st.session_state.show_prediction and role == "👨‍👩 Patient":
            try:
                defaults = get_patient_defaults()
                input_features, derived = create_feature_vector_with_defaults(
                    pat_age, pat_sex, _cp_encoded, pat_trestbps, pat_chol,
                    _fbs_val, pat_thalach, _exang_encoded, defaults,
                )
                prediction, probability, shap_row = run_prediction(input_features)
                tier_key, tier_label, bar_color   = disease_risk_band(probability, patient_mode=True)
                log_prediction(
                    pat_age, pat_sex, pat_trestbps, pat_chol,
                    probability, tier_label, "Patient",
                    hide_probability=True,
                )

                show_patient_friendly_result(
                    tier_key=tier_key,
                    tier_label=tier_label,
                    probability=probability,
                    shap_row=shap_row,
                    feature_names=feature_names,
                    trestbps=pat_trestbps,
                    chol=pat_chol,
                    age=pat_age,
                    bp_was_estimated=(pat_bp_known == "No, I don't know"),
                    chol_was_estimated=(pat_chol_known == "No, I don't know"),
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")
                with st.expander("Debug info"):
                    import traceback; st.code(traceback.format_exc())


# ============================================================================
# TAB 2: WHAT-IF EXPLORER  (Doctor mode only — needs full clinical data)
# ============================================================================

with tab_whatif:
    st.subheader("🔬 What-If Scenario Explorer")

    if role == "👨‍👩 Patient":
        st.info(
            "The What-If Explorer is available in **Doctor mode** only, "
            "as it explores the full set of clinical parameters.\n\n"
            "Switch to 🧑‍⚕️ Doctor in the sidebar to use this tool."
        )
    elif not st.session_state.show_prediction:
        st.info("Run a prediction in the **Predict** tab first, then return here.")
    else:
        try:
            wi_col1, wi_col2 = st.columns(2)
            with wi_col1:
                wi_bp = st.slider("What if resting BP were… (mm Hg)",
                                  80, 200, int(trestbps), key="wi_bp")
            with wi_col2:
                wi_chol = st.slider("What if cholesterol were… (mg/dl)",
                                    100, 600, int(chol), key="wi_chol")

            base_feats, _ = create_feature_vector(
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal)
            wi_feats, _ = create_feature_vector(
                age, sex, cp, wi_bp, wi_chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal)

            _, base_prob, _ = run_prediction(base_feats)
            _, wi_prob,   _ = run_prediction(wi_feats)
            _, base_label, _= disease_risk_band(base_prob)
            wi_key, wi_label, _ = disease_risk_band(wi_prob)
            base_key_only, *_ = disease_risk_band(base_prob)

            delta = wi_prob - base_prob
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline probability",  f"{base_prob:.1%}")
            c2.metric("What-If probability",   f"{wi_prob:.1%}",
                      delta=f"{delta:+.1%}", delta_color="inverse")
            c3.metric("Risk band",             wi_label,
                      delta=("same" if wi_key == base_key_only else f"was {base_label}"),
                      delta_color="off")

            if delta < -0.05:
                st.success(f"Lowering BP to **{wi_bp} mm Hg** and cholesterol to **{wi_chol} mg/dl** "
                           f"could reduce predicted risk by **{-delta:.1%}**.")
            elif delta > 0.05:
                st.warning(f"Raising these values would increase predicted risk by **{delta:.1%}**.")
            else:
                st.info("Small effect on predicted probability for this patient.")

        except Exception as e:
            st.error(f"What-If error: {e}")


# ============================================================================
# TAB 3: HISTORY
# ============================================================================

with tab_history:
    st.subheader("📋 Session Prediction History")
    st.caption("Logged in-memory for this browser session only — not saved to disk.")

    history = st.session_state.prediction_history
    if not history:
        st.info("No predictions yet. Run one in the Predict tab.")
    else:
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist, use_container_width=True)
        if st.button("🗑️ Clear history"):
            st.session_state.prediction_history = []
            st.rerun()


# ============================================================================
# TAB 4: MODEL INSIGHTS
# ============================================================================

with tab_global:
    st.subheader("📊 Model Insights")
    st.markdown(
        "Global views of what the model learned across the training dataset "
        "(not patient-specific)."
    )

    for img_path, title, caption in [
        (_ROOT / "feature_importance.png",
         "🏆 Top 15 Feature Importances", None),
        (_ROOT / "shap_summary_plot.png",
         "🔮 SHAP Summary Plot",
         "Each dot = one test sample. Red = high feature value. Positive SHAP = increases risk."),
        (_ROOT / "roc_curve.png",  "📈 ROC Curve",       None),
        (_ROOT / "confusion_matrix.png", "🔢 Confusion Matrix", None),
    ]:
        if img_path.exists():
            st.markdown(f"### {title}")
            if caption: st.caption(caption)
            st.image(str(img_path), use_container_width=True)
        else:
            st.warning(f"{img_path.name} not found — run `models/train_model.py` to generate it.")

    if model_metrics:
        st.markdown("### 📐 Test-Set Metrics")
        cols = st.columns(len(model_metrics))
        for col, (name, val) in zip(cols, model_metrics.items()):
            col.metric(name, f"{val:.1%}")

    if patient_defaults_labels:
        st.markdown("### 🔧 Patient screening — imputed clinical test fields")
        st.caption(
            "Training-data median (numeric) or mode (categorical) for **five** fields "
            "patients do not self-report (ECG, ST depression, ST slope, vessels, thalassemia)."
        )
        st.table(pd.DataFrame(
            [{"Field": k, "Default value": v, "Source": "Training data median/mode"}
             for k, v in patient_defaults_labels.items()]
        ))

    if patient_screening_medians:
        st.markdown("### 🔧 Patient screening — unknown BP / cholesterol")
        st.caption('Dataset medians used when the patient selects "I don\'t know".')
        st.table(pd.DataFrame(
            [{"Measure": k, "Median (training data)": v} for k, v in patient_screening_medians.items()]
        ))


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<center><small>⚠️ CardioClarity is an AI-assisted screening tool. "
    "Always consult a qualified healthcare professional for medical decisions.</small></center>",
    unsafe_allow_html=True,
)