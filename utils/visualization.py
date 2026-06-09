import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import pandas as pd
import numpy as np

# ── Risk band thresholds (must stay in sync with app.py) ─────────────────────
RISK_LOW_BELOW = 0.35
RISK_MEDIUM_BELOW = 0.65


def _shap_scalar(v):
    """SHAP may return 0-d or length>1 arrays; comparisons need a Python float."""
    a = np.asarray(v, dtype=float).ravel()
    return float(a[0]) if a.size else 0.0


def shap_positive_class_row(shap_out, n_features, sample_index=0):
    """
    Return SHAP vector (n_features,) for the positive class (index 1), first sample.
    Handles list [neg, pos], (n, feats), and some (n, feats, 2) layouts.
    """
    if isinstance(shap_out, list):
        pos = np.asarray(shap_out[1], dtype=float)
    else:
        pos = np.asarray(shap_out, dtype=float)

    if pos.ndim == 1:
        if pos.shape[0] != n_features:
            raise ValueError(f"Unexpected 1D SHAP shape {pos.shape}, expected ({n_features},)")
        return pos

    if pos.ndim == 2:
        return pos[sample_index]

    if pos.ndim == 3:
        # (samples, features, classes)
        if pos.shape[1] == n_features and pos.shape[2] == 2:
            return pos[sample_index, :, 1]
        # (samples, classes, features)
        if pos.shape[1] == 2 and pos.shape[2] == n_features:
            return pos[sample_index, 1, :]
        # (samples, features, 2) with features second
        if pos.shape[2] == 2 and pos.shape[1] == n_features:
            return pos[sample_index, :, 1]

    raise ValueError(f"Unhandled SHAP output shape {pos.shape}; n_features={n_features}")


def create_shap_force_plot(
    shap_values,
    feature_values,
    feature_names,
    priority_features=("trestbps", "chol"),
    n_bars=6,
):
    """
    Horizontal bar chart of SHAP contributions.
    `priority_features` are always included first (when present), then other
    strongest contributors, so BP and cholesterol stay visible for clinicians.
    """
    shap_dict = {n: _shap_scalar(sv) for n, sv in zip(feature_names, shap_values)}

    contributions = [(name, shap_dict[name]) for name in feature_names]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    ordered = []
    used = set()
    for pf in priority_features:
        if pf in shap_dict and pf not in used:
            ordered.append((pf, shap_dict[pf]))
            used.add(pf)
    for name, val in contributions:
        if name in used:
            continue
        ordered.append((name, val))
        used.add(name)
        if len(ordered) >= n_bars:
            break

    # Matplotlib barh: y=0 at bottom; reverse so priority rows sit at top of chart
    plot_order = list(reversed(ordered))
    names = [f[0] for f in plot_order]
    values = [float(f[1]) for f in plot_order]
    y_pos = range(len(names))

    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.45 * len(names))))
    colors = ['#ef4444' if v > 0 else '#22c55e' for v in values]
    ax.barh(list(y_pos), values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('SHAP value (impact on heart-disease risk score)')
    ax.set_title('Feature Contributions (resting BP & cholesterol pinned first)')

    red_patch = mpatches.Patch(color='#ef4444', label='Increases risk')
    green_patch = mpatches.Patch(color='#22c55e', label='Decreases risk')
    ax.legend(handles=[red_patch, green_patch], loc='lower right', fontsize=9)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


def create_risk_gauge(probability):
    """
    Create a risk gauge using the same 3-tier band as the rest of the app:
      < 35%  → Low   (green)
      35–65% → Medium (amber)
      ≥ 65%  → High  (red)

    Previously used a hardcoded 0.5 threshold (inconsistent with app logic).
    """
    if probability < RISK_LOW_BELOW:
        bar_color = '#22c55e'
        tier_label = 'Low Risk'
    elif probability < RISK_MEDIUM_BELOW:
        bar_color = '#f59e0b'
        tier_label = 'Medium Risk'
    else:
        bar_color = '#ef4444'
        tier_label = 'High Risk'

    fig, ax = plt.subplots(figsize=(8, 2.2))

    # Background track
    ax.barh(['Risk Level'], [1], color='#e5e7eb', height=0.5)
    # Filled portion
    ax.barh(['Risk Level'], [probability], color=bar_color, height=0.5)

    # Threshold markers
    ax.axvline(x=RISK_LOW_BELOW, color='#6b7280', linestyle='--', alpha=0.7, linewidth=1.2,
               label=f'Low/Medium ({RISK_LOW_BELOW:.0%})')
    ax.axvline(x=RISK_MEDIUM_BELOW, color='#374151', linestyle='--', alpha=0.7, linewidth=1.2,
               label=f'Medium/High ({RISK_MEDIUM_BELOW:.0%})')

    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title(f'Heart Disease Risk Gauge — {tier_label} ({probability:.1%})')
    ax.legend(fontsize=8, loc='upper left')

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_simple_explanation(shap_values, feature_names, feature_values):
    """Generate plain-English explanation for patients."""
    impacts = [(name, _shap_scalar(shap_values[i]), feature_values[i])
               for i, name in enumerate(feature_names)]
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    top_factors = impacts[:3]

    explanation = "Based on your information:\n\n"
    for name, impact, value in top_factors:
        imp = _shap_scalar(impact)
        if imp > 0.05:
            explanation += f"✓ Your **{name}** increases your risk\n"
        elif imp < -0.05:
            explanation += f"✓ Your **{name}** decreases your risk\n"

    return explanation