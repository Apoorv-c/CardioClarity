# ❤️ CardioClarity — AI-Powered Heart Disease Risk Assessment

<div align="center">

![CardioClarity Banner](https://img.shields.io/badge/CardioClarity-Heart%20Risk%20AI-e63946?style=for-the-badge&logo=heart&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Railway](https://img.shields.io/badge/Deployed%20on-Railway-0B0D0E?style=for-the-badge&logo=railway&logoColor=white)

**Live Demo → [web-production-a94f5.up.railway.app](https://web-production-a94f5.up.railway.app)**

*A clinically-informed, explainable AI tool that predicts heart disease risk — built for both doctors and patients.*

</div>

---

## 🌟 What is CardioClarity?

CardioClarity is a web application that uses machine learning to assess a person's risk of heart disease. It's not a replacement for a doctor — think of it as a smart second opinion that helps clinical professionals and patients alike understand cardiovascular risk in plain, actionable terms.

You enter clinical data (like blood pressure, cholesterol, age, and chest pain type), and CardioClarity:
- Predicts whether heart disease is likely
- Shows **exactly why** it made that prediction (using SHAP explainability)
- Gives tailored clinical recommendations
- Lets you explore "what if I lower my BP?" scenarios in real time
- Generates a downloadable PDF report

---

## ✨ Features at a Glance

### 👨‍⚕️ Doctor Mode
Full clinical input panel with all 13 UCI features + 12 engineered cardiovascular risk metrics. Designed for healthcare professionals who want the complete picture.

- Risk probability with confidence gauge
- SHAP waterfall chart (blood pressure & cholesterol always pinned as priority features)
- Clinical recommendations based on risk tier
- What-If Explorer — adjust BP/cholesterol sliders and watch the risk update live
- Model performance metrics (Accuracy, AUC-ROC, F1, etc.)
- Global feature importance, SHAP summary plot, ROC curve, confusion matrix

### 🧑 Patient Mode
Simplified, jargon-free interface that hides technical ECG/thalassemia fields and uses plain-language inputs.

- Friendly risk summary ("Low Risk / Medium Risk / High Risk")
- Lifestyle tips and next steps explained in everyday language
- No medical degree required to understand the results

### 🔐 User Accounts
- Secure registration and login (SHA-256 password hashing, token-based auth)
- Persistent prediction history per user
- Delete individual history entries

### 📄 PDF Reports
Generate and download a professional A4 clinical report including the risk gauge, SHAP chart, input summary, and recommendations — ready to share with a healthcare provider.

---

## 🧠 How the Model Works

The prediction engine is a **Random Forest Classifier** trained on the UCI Heart Disease dataset (Cleveland subset), enhanced with 12 custom-engineered cardiovascular risk features.

| Property | Value |
|---|---|
| Algorithm | Random Forest (`n_estimators=200`, `max_depth=12`) |
| Features | 25 total (13 raw UCI + 12 engineered) |
| Class Imbalance | Handled via `class_weight='balanced'` |
| Scaler | `RobustScaler` (robust to outliers) |
| Explainability | SHAP `TreeExplainer` |

### 📊 Model Performance (Test Set)

| Metric | Score |
|---|---|
| Accuracy | ~87% |
| Precision | ~85% |
| Recall | ~88% |
| F1-Score | ~86% |
| AUC-ROC | ~92% |

### 🔬 Engineered Features
Beyond the standard UCI fields, CardioClarity adds clinically meaningful derived features:

- `age_group`, `age_over_50` — age risk stratification
- `bp_risk`, `hypertension` — blood pressure severity tiers
- `chol_risk`, `high_cholesterol` — cholesterol severity tiers
- `max_hr_target`, `hr_percentage`, `low_hr` — age-adjusted heart rate analysis
- `cardiovascular_risk` — composite risk score (age + BP + cholesterol)
- `significant_st_depression` — ST segment depression flag
- `multi_vessel` — multi-vessel disease indicator

---

## 🚀 Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/Apoorv-c/CardioClarity.git
cd CardioClarity
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the server
```bash
python server.py
```

Then open **http://localhost:5000** in your browser.

> The pre-trained model (`.pkl` files) is included in the repo — no training step needed.

### Optional: Retrain the model
If you want to retrain on fresh data:
```bash
python models/train_model.py
```
This will regenerate all `.pkl` artifacts and model performance images.

---

## 🗂️ Project Structure

```
CardioClarity/
│
├── server.py                     # Flask REST API + app entry point
├── Procfile                      # Gunicorn deployment config (Railway/Heroku)
├── requirements.txt              # Python dependencies
├── .python-version               # Python 3.12.10 (for Railway/mise)
│
├── static/                       # Frontend (HTML + CSS + JS)
│   ├── index.html                # Single-page app shell
│   ├── css/
│   │   ├── style.css             # Main styles
│   │   ├── particles.css         # Background particle animation
│   │   └── chatbot.css           # Chatbot widget styles
│   └── js/
│       ├── app.js                # Core app logic & prediction flow
│       ├── auth.js               # Login / register / session
│       ├── history.js            # Prediction history tab
│       ├── pdf.js                # PDF report generation
│       ├── particles.js          # Animated particle background
│       ├── chatbot.js            # AI chatbot widget
│       └── dashboard.js          # Dashboard charts & metrics
│
├── models/
│   └── train_model.py            # Full training pipeline
│
├── utils/
│   ├── data_loader.py            # Dataset loading utilities
│   ├── explainer.py              # SHAP helper functions
│   └── visualization.py         # Plot generation utilities
│
├── views/
│   ├── doctor_views.py           # Doctor role view logic
│   └── patient_view.py          # Patient role view logic
│
├── data/
│   └── heart_disease_uci.csv    # UCI Heart Disease dataset
│
└── *.pkl / *.png                 # Pre-trained model artifacts & charts
```

---

## 🌐 API Reference

All endpoints are prefixed with `/api/`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Create a new account |
| `POST` | `/api/auth/login` | Log in and receive a session token |
| `POST` | `/api/auth/logout` | Invalidate the current session token |
| `GET` | `/api/auth/profile` | Get user profile and history |
| `POST` | `/api/predict` | Run a prediction (`mode: doctor` or `patient`) |
| `POST` | `/api/whatif` | What-If analysis for BP/cholesterol |
| `GET` | `/api/history` | Fetch saved prediction history |
| `DELETE` | `/api/history/<id>` | Delete a history entry |
| `GET` | `/api/model/metrics` | Model performance metrics |
| `GET` | `/api/model/images/<name>` | Serve model chart images |
| `POST` | `/api/report/pdf` | Generate a downloadable PDF report |

---

## ☁️ Deployment

CardioClarity is deployed on **Railway** using Gunicorn.

```
web: gunicorn server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

**Live URL:** https://web-production-a94f5.up.railway.app

To deploy your own instance:
1. Fork this repository
2. Connect it to [railway.app](https://railway.app)
3. Railway auto-detects Python and reads the `Procfile` — no extra config needed

---

## ⚠️ Disclaimer

CardioClarity is a **research and educational tool**. It is not a medical device and must not be used as a substitute for professional clinical judgment.

- Predictions are based on statistical patterns in training data
- The model has not been validated in a clinical setting
- Always consult a qualified healthcare professional for medical decisions

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, Flask, Gunicorn |
| ML / AI | scikit-learn (Random Forest), SHAP, NumPy, Pandas |
| Visualization | Matplotlib, Plotly, Seaborn |
| PDF Generation | ReportLab |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Deployment | Railway (Railpack builder) |
| Version Control | Git + GitHub |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ by [Apoorv-c](https://github.com/Apoorv-c)

*If this project helped you, please consider giving it a ⭐ on GitHub!*

</div>
