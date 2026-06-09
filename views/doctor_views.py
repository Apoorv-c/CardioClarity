"""
views/doctor_views.py
=====================
Provides show_doctor_extra_panel() — called by app.py when role == Doctor.
Renders the extended clinical decision-support section below the shared
prediction output.

Previously this file:
  • Referenced an 'origin' feature that does not exist in the 25-feature model.
  • Was never imported by app.py (dead code).
  • Used raw int values for categorical features (incompatible with the
    label-encoder pipeline in app.py).

All of those issues are now resolved: this module receives already-computed
values from app.py and only handles display logic.
"""

import streamlit as st
from utils.visualization import create_risk_gauge


def show_doctor_extra_panel(
    tier_key: str,
    tier_label: str,
    probability: float,
    trestbps: int,
    chol: int,
    age: int,
    derived: dict,
    model_metrics: dict | None = None,
):
    """
    Render the extended Doctor-only section.

    Parameters
    ----------
    tier_key       : "low" | "medium" | "high"
    tier_label     : Human-readable tier name
    probability    : Raw model probability (0-1)
    trestbps       : Resting blood pressure (mm Hg)
    chol           : Serum cholesterol (mg/dl)
    age            : Patient age (years)
    derived        : Dict of engineered feature values from compute_derived_metrics()
    model_metrics  : Optional dict loaded from heart_disease_model_artifacts.pkl
    """

    st.markdown("---")
    st.subheader("🔬 Clinical Decision Support")

    # ── Risk gauge ────────────────────────────────────────────────────────────
    create_risk_gauge(probability)

    st.metric("Risk band (probability-based)", tier_label)
    st.caption(f"Raw disease probability: **{probability:.1%}**")
    st.markdown("**Primary inputs:** resting BP and cholesterol (with engineered risk tiers).")

    # ── Clinical metrics grid ─────────────────────────────────────────────────
    col_bp, col_chol, col_rest = st.columns(3)
    with col_bp:
        st.metric("Resting BP (mm Hg)", trestbps)
        st.caption(f"Tier {derived['bp_risk']}/2 · hypertension flag: {derived['hypertension']}")
    with col_chol:
        st.metric("Cholesterol (mg/dl)", chol)
        st.caption(f"Tier {derived['chol_risk']}/2 · high chol flag: {derived['high_cholesterol']}")
    with col_rest:
        st.metric("Age (years)", age)
        st.metric("CV risk score (sum)", derived['cardiovascular_risk'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Age group (encoded)", derived['age_group'])
        st.metric("BP risk tier", derived['bp_risk'])
    with col2:
        st.metric("Chol risk tier", derived['chol_risk'])
        st.metric("Heart rate % of target", f"{derived['hr_percentage']:.1%}")
    with col3:
        st.metric("ST depression flag", derived['significant_st_depression'])
        st.metric("Multi-vessel", "Yes" if derived['multi_vessel'] else "No")

    # ── Clinical recommendations ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**📋 Clinical Recommendations:**")
    if tier_key == "high":
        st.markdown("""
        - ✅ Prioritize discussion with cardiology / primary care
        - ✅ Consider further diagnostic testing (ECG, stress test, imaging)
        - ✅ Intensive lifestyle review (diet, exercise, smoking cessation)
        - ✅ Medication review (statins, BP control) per guidelines
        - ✅ Short-interval follow-up as clinically appropriate
        """)
    elif tier_key == "medium":
        st.markdown("""
        - ✅ Reinforce BP and cholesterol targets with your patient
        - ✅ Repeat risk assessment at next visit; consider baseline labs
        - ✅ Lifestyle counseling (activity, diet, weight, smoking)
        - ✅ Escalate testing if symptoms or risk scores worsen
        """)
    else:
        st.markdown("""
        - ✅ Continue current healthy habits where appropriate
        - ✅ Routine preventive care and annual check-ups
        - ✅ Monitor blood pressure and cholesterol regularly
        - ✅ Maintain physical activity (~150 min/week moderate intensity)
        """)

    # ── Model performance metrics ─────────────────────────────────────────────
    with st.expander("📊 Model Performance Metrics"):
        if model_metrics:
            # Show real metrics loaded from the saved artifact
            st.caption("Metrics computed on the held-out test set during training.")
            cols = st.columns(len(model_metrics))
            for col, (name, val) in zip(cols, model_metrics.items()):
                col.metric(name, f"{val:.1%}")
        else:
            # Fallback static values (shown only if artifact not available)
            st.warning("Live metrics unavailable — showing approximate values from training run.")
        st.markdown("""
        - **Model type:** Random Forest Classifier
        - **Features used:** 25 clinical + engineered features
        - **Scaler:** RobustScaler
        - **Class imbalance:** handled with `class_weight='balanced'`
        """)