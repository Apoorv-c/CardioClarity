"""
Patient-facing results: no exact probability, no SHAP jargon, conservative messaging.
"""

import streamlit as st
import numpy as np

# Features we explain in plain language (exclude engineered / technical columns)
_PATIENT_EXPLAIN_FEATURES = frozenset({
    "age", "sex", "cp", "trestbps", "chol", "fbs", "thalach", "exang",
})

_READABLE = {
    "age": "your age",
    "sex": "sex (as recorded)",
    "cp": "how you described chest discomfort",
    "trestbps": "resting blood pressure",
    "chol": "cholesterol",
    "fbs": "blood sugar / diabetes history",
    "thalach": "heart rate with exercise",
    "exang": "chest pain with exercise",
}


def _qualitative_bullets(feature_names: list, shap_row, max_bullets: int = 4) -> list[str]:
    """Turn ranked SHAP contributions into plain-language bullets (no numbers)."""
    pairs = [
        (name, float(np.asarray(v, dtype=float).ravel()[0]))
        for name, v in zip(feature_names, shap_row)
        if name in _PATIENT_EXPLAIN_FEATURES
    ]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    out = []
    for name, impact in pairs:
        if abs(impact) < 0.02:
            continue
        label = _READABLE.get(name, name)
        if impact > 0:
            out.append(f"In this screening, **{label}** tended to move the estimate toward **more** concern.")
        else:
            out.append(f"In this screening, **{label}** tended to move the estimate toward **less** concern.")
        if len(out) >= max_bullets:
            break
    return out


def show_patient_friendly_result(
    tier_key: str,
    tier_label: str,
    probability: float,
    shap_row,
    feature_names: list,
    trestbps: int,
    chol: int,
    age: int,
    bp_was_estimated: bool = False,
    chol_was_estimated: bool = False,
):
    """
    Patient mode only. Never displays `probability`. Doctor mode must not call this.
    """
    _ = probability  # used only inside the model path; do not leak to UI

    # ── Before-results disclaimer (repeated for visibility) ─────────────────
    st.warning(
        "**Preliminary screen only**\n\n"
        "**Seven** kinds of clinical information that usually come from a clinic "
        "(for example resting ECG, detailed stress-test measures, artery imaging, "
        "and related specialist findings) **were not collected** here. Those were "
        "filled in using **typical population values**, so the result is **rough** "
        "and **not** a diagnosis.\n\n"
        "If you told us you did not know your blood pressure or cholesterol, those "
        "numbers were also **estimated** from typical values — not measured today."
    )

    st.markdown("---")
    st.subheader("Your screening result")

    # ── Three-band visual (no percentage) ──────────────────────────────────
    band_title = {
        "low": ("🟢", "Lower risk (screening band)", "This band does **not** mean you are fine or disease-free — many findings were guessed."),
        "medium": ("🟡", "Moderate risk (screening band)", "Worth discussing with a clinician at a routine visit."),
        "high": ("🔴", "Higher risk (screening band)", "Please contact a clinician soon — this screen is not a diagnosis."),
    }
    emoji, plain_title, body = band_title.get(tier_key, ("⚪", tier_label, ""))

    c1, c2, c3 = st.columns(3)
    for col, key, label, color in zip(
        (c1, c2, c3),
        ("low", "medium", "high"),
        ("🟢 Lower risk", "🟡 Moderate risk", "🔴 Higher risk"),
        ("#22c55e", "#f59e0b", "#ef4444"),
    ):
        active = tier_key == key
        with col:
            border = "3px solid #1f2937" if active else "1px solid #ccc"
            opacity = "1" if active else "0.45"
            _html = (
                f'<div style="text-align:center;padding:14px;border-radius:10px;'
                f'background:{color};opacity:{opacity};color:#fff;font-weight:600;'
                f'border:{border};margin-bottom:8px">{label}</div>'
            )
            st.markdown(_html, unsafe_allow_html=True)

    if tier_key == "high":
        st.error(f"{emoji} **{plain_title}**  \n{body}")
    elif tier_key == "medium":
        st.warning(f"{emoji} **{plain_title}**  \n{body}")
    else:
        st.info(f"{emoji} **{plain_title}**  \n{body}")

    st.caption(
        "You are **not** shown a percentage because it would look more exact than this screen can be. "
        "Bands use **stricter** cutoffs than the clinician view so more people are encouraged to follow up."
    )

    if bp_was_estimated or chol_was_estimated:
        bits = []
        if bp_was_estimated:
            bits.append("blood pressure")
        if chol_was_estimated:
            bits.append("cholesterol")
        st.info(
            f"**Note:** We estimated your **{' and '.join(bits)}** because you chose "
            '"I don\'t know." A nurse or doctor can measure the real values.'
        )

    # ── What influenced the estimate (no SHAP term, no numbers) ────────────
    st.markdown("---")
    st.subheader("What most influenced this estimate")
    st.caption(
        "Simple summary of how your answers (and a few typical stand-ins) affected "
        "the tool — not a medical explanation of disease."
    )
    bullets = _qualitative_bullets(feature_names, shap_row)
    if bullets:
        for b in bullets:
            st.markdown(f"- {b}")
    else:
        st.markdown(
            "- No single answer dominated this rough estimate; several factors combined."
        )

    st.markdown("---")
    st.subheader("Heart-healthy habits (general)")
    tips = []
    if trestbps > 130:
        tips.append("Discuss **blood pressure** with a clinician; salt reduction, activity, and sleep help many people.")
    if chol > 200:
        tips.append("Discuss **cholesterol** with a clinician; fibre-rich foods and regular activity are commonly recommended.")
    tips += [
        "Aim for regular **physical activity** as advised by your healthcare provider.",
        "Avoid **smoking** and limit alcohol as appropriate for you.",
        "Keep **routine check-ups** even when you feel well.",
    ]
    for t in tips:
        st.markdown(f"- {t}")

    # ── After-results disclaimer + always see a doctor ──────────────────────
    st.markdown("---")
    st.error(
        "**Please see a doctor or nurse for a full assessment.**  \n"
        "This tool cannot examine you, cannot see your ECG or labs, and cannot rule "
        "out heart disease. **Any** result here — lower, moderate, or higher concern — "
        "should be followed up with a qualified professional if you have symptoms, risk "
        "factors, or worries."
    )
    st.caption(
        "CardioClarity is not a medical device and does not replace clinical judgment."
    )
