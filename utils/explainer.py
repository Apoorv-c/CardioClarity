from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from utils.visualization import shap_positive_class_row

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _ROOT / "random_forest_model.pkl"
_SCALER_PATH = _ROOT / "scaler.pkl"
_FEATURE_NAMES_PATH = _ROOT / "feature_names.pkl"
_ARTIFACTS_PATH = _ROOT / "heart_disease_model_artifacts.pkl"


@st.cache_resource
def load_explainer_resources():
    """
    Load model artifacts once and cache them.
    Previously these were loaded at module-level, which caused a double-load
    when app.py also loaded them. Now cached via Streamlit so both callers
    share the same in-memory objects.
    """
    model = joblib.load(_MODEL_PATH)
    scaler = joblib.load(_SCALER_PATH)
    feature_names = joblib.load(_FEATURE_NAMES_PATH)
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_names, explainer


def get_prediction_and_explanation(input_data):
    """
    Get prediction probability and SHAP explanation.

    Args:
        input_data: numpy array of shape (1, n_features), raw engineered features
            (will be RobustScaler-transformed here).

    Returns:
        prediction: 0 or 1
        probability: float between 0-1
        shap_values: array of SHAP values for the positive class (n_features,)
    """
    model, scaler, feature_names, explainer = load_explainer_resources()

    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)

    input_df = pd.DataFrame(np.asarray(input_data), columns=feature_names)
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(probability > 0.5)

    shap_raw = explainer.shap_values(input_scaled)
    shap_values = shap_positive_class_row(shap_raw, len(feature_names))

    return prediction, probability, shap_values


def get_feature_importance():
    """Return global feature importances as a sorted list of (name, importance) tuples."""
    model, _, feature_names, _ = load_explainer_resources()
    importance = model.feature_importances_
    return sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)


def get_decision_rules():
    """Return top-3 features as plain-English bullet points."""
    top_features = get_feature_importance()[:3]
    return [f"• {feat}: {imp:.2%} impact on prediction" for feat, imp in top_features]
