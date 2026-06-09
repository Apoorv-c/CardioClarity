# CardioClarity — Heart Disease Risk Assessment

An AI-powered, explainable heart disease prediction web app built with
**Streamlit**, **scikit-learn**, and **SHAP**.

---

## 📁 Project Structure

```
CardioClarity/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── data/
│   └── heart_disease_uci.csv     # UCI Heart Disease dataset (required for training)
├── models/
│   └── train_model.py            # Full training pipeline (run once)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py            # Dataset loading utilities
│   ├── explainer.py              # Prediction + SHAP helper (cached)
│   └── visualization.py          # SHAP plots, risk gauge
├── views/
│   ├── __init__.py
│   ├── doctor_views.py           # Extended clinical panel (Doctor role)
│   └── patient_view.py           # Patient-friendly result panel
└── *.pkl / *.png                 # Generated artifacts (see below)
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the **UCI Heart Disease** CSV in the `data/` folder:

```
data/heart_disease_uci.csv
```

The dataset can be downloaded from:
https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

### 3. Train the model (first time only)

```bash
python models/train_model.py
```

This produces the following files in the project root:
- `random_forest_model.pkl`
- `scaler.pkl`
- `feature_names.pkl`
- `label_encoders.pkl`
- `heart_disease_model_artifacts.pkl`
- `feature_importance.csv`
- `feature_importance.png`
- `shap_summary_plot.png`
- `roc_curve.png`
- `confusion_matrix.png`

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Dual role** | Doctor view (clinical metrics, real model performance, risk gauge) and Patient view (plain-language summary, lifestyle tips) |
| **25-feature model** | 13 raw UCI features + 12 engineered cardiovascular risk features |
| **SHAP explanations** | Per-prediction waterfall chart; trestbps & chol always pinned |
| **Input validation** | Warnings for clinically implausible values (BP, cholesterol, heart rate) |
| **What-If Explorer** | Adjust BP and cholesterol sliders to see probability change in real time |
| **Session history** | Log of all predictions made in the current browser session |
| **Model Insights** | Global feature importance, SHAP summary, ROC curve, confusion matrix |
| **Live metrics** | Real test-set metrics loaded from `heart_disease_model_artifacts.pkl` |

---

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier (`n_estimators=200`, `max_depth=12`)
- **Class imbalance:** `class_weight='balanced'` (no SMOTE)
- **Scaler:** `RobustScaler`
- **Performance (test set):**
  - Accuracy ~87% · Precision ~85% · Recall ~88% · AUC-ROC ~92%

---

## ⚠️ Disclaimer

CardioClarity is a research and educational tool. It is **not** a medical device
and must not be used as a substitute for professional clinical judgement.
Always consult a qualified healthcare professional for medical decisions.
