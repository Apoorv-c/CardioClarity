# complete_pipeline_fixed.py
"""
COMPLETE HEART DISEASE XAI PIPELINE - FIXED VERSION
No NaN errors, handles all edge cases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.impute import SimpleImputer
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🏥 HEART DISEASE XAI - COMPLETE PIPELINE (FIXED)")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n📂 Step 1: Loading dataset...")
df = pd.read_csv('data/heart_disease_uci.csv')
print(f"   ✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Columns: {df.columns.tolist()}")

# ============================================================================
# STEP 2: INITIAL CLEANING
# ============================================================================
print("\n🗑️ Step 2: Initial cleaning...")

# Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"   ✅ Removed {initial_rows - len(df)} duplicates")

# Drop id if exists
if 'id' in df.columns:
    df = df.drop('id', axis=1)
    print(f"   ✅ Dropped 'id' column")

# Fix column name if needed (thalch vs thalach)
if 'thalch' in df.columns and 'thalach' not in df.columns:
    df = df.rename(columns={'thalch': 'thalach'})
    print(f"   ✅ Renamed 'thalch' to 'thalach'")

# ============================================================================
# STEP 3: COMPREHENSIVE MISSING VALUE HANDLING
# ============================================================================
print("\n🔧 Step 3: Comprehensive missing value handling...")

print(f"   Missing values before:")
missing_before = df.isnull().sum()
print(f"   {missing_before[missing_before > 0]}")

# Handle numerical columns
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
for col in numerical_cols:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())
        print(f"   ✅ Filled {col} with median: {df[col].median()}")

# Handle categorical columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
for col in categorical_cols:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])
        print(f"   ✅ Filled {col} with mode: {df[col].mode()[0]}")

# Special handling for ca (major vessels)
if 'ca' in df.columns:
    df['ca'] = df['ca'].fillna(0)
    print(f"   ✅ Set ca missing to 0")

# Special handling for oldpeak
if 'oldpeak' in df.columns:
    df['oldpeak'] = df['oldpeak'].fillna(0)
    print(f"   ✅ Set oldpeak missing to 0")

print(f"\n   Missing values after: {df.isnull().sum().sum()}")

# ============================================================================
# STEP 4: REMOVE OUTLIERS (CAREFULLY)
# ============================================================================
print("\n📊 Step 4: Removing outliers...")

outlier_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
outlier_cols = [col for col in outlier_cols if col in df.columns]

initial_rows = len(df)
for col in outlier_cols:
    if col in df.columns and len(df[col].dropna()) > 0:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_before = len(df)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        print(f"   {col}: removed {outliers_before - len(df)} outliers")

print(f"   ✅ Total removed: {initial_rows - len(df)} outliers")

# ============================================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================================
print("\n✨ Step 5: Feature Engineering...")

# Age features
if 'age' in df.columns:
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 100], 
                              labels=[0, 1, 2, 3, 4]).astype(int)
    df['age_over_50'] = (df['age'] > 50).astype(int)
    print("   ✅ Created age features")

# Blood pressure features
if 'trestbps' in df.columns:
    df['bp_risk'] = df['trestbps'].apply(lambda x: 2 if x > 140 else (1 if x > 120 else 0))
    df['hypertension'] = (df['trestbps'] > 140).astype(int)
    print("   ✅ Created BP features")

# Cholesterol features
if 'chol' in df.columns:
    df['chol_risk'] = df['chol'].apply(lambda x: 2 if x > 240 else (1 if x > 200 else 0))
    df['high_cholesterol'] = (df['chol'] > 240).astype(int)
    print("   ✅ Created cholesterol features")

# Heart rate features
if 'thalach' in df.columns and 'age' in df.columns:
    df['max_hr_target'] = 220 - df['age']
    df['hr_percentage'] = df['thalach'] / df['max_hr_target']
    df['low_hr'] = (df['hr_percentage'] < 0.85).astype(int)
    print("   ✅ Created heart rate features")

# Combined risk score
if all(col in df.columns for col in ['age', 'trestbps', 'chol']):
    df['cardiovascular_risk'] = (
        (df['age'] > 50).astype(int) + 
        (df['trestbps'] > 140).astype(int) + 
        (df['chol'] > 240).astype(int)
    )
    print("   ✅ Created combined risk score")

# ST depression features
if 'oldpeak' in df.columns:
    df['significant_st_depression'] = (df['oldpeak'] > 1).astype(int)
    print("   ✅ Created ST depression features")

# Vessel disease feature
if 'ca' in df.columns:
    df['multi_vessel'] = (df['ca'] >= 2).astype(int)
    print("   ✅ Created vessel disease features")

print(f"   ✅ Total features now: {len(df.columns)}")

# ============================================================================
# STEP 6: ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n🏷️ Step 6: Encoding categorical variables...")

label_encoders = {}
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
categorical_cols = [col for col in categorical_cols if col in df.columns]

for col in categorical_cols:
    le = LabelEncoder()
    # Convert to string first to handle any mixed types
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   ✅ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ============================================================================
# STEP 7: PREPARE FEATURES AND TARGET
# ============================================================================
print("\n🎯 Step 7: Preparing features and target...")

# Create binary target
if 'num' in df.columns:
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop('num', axis=1)
elif 'target' not in df.columns:
    print("   ❌ No target column found!")
    exit()

# Select only numeric columns for features
X = df.drop('target', axis=1).select_dtypes(include=[np.number])
y = df['target']

# Ensure no NaN values remain
if X.isnull().any().any():
    print(f"   ⚠️ Still have NaN values! Filling with 0...")
    X = X.fillna(0)

feature_names = X.columns.tolist()
print(f"   ✅ Final features ({len(feature_names)}): {feature_names[:10]}...")
print(f"   ✅ Target distribution:\n      {y.value_counts().to_dict()}")
print(f"   ✅ Disease prevalence: {y.mean():.2%}")

# ============================================================================
# STEP 8: TRAIN-TEST SPLIT
# ============================================================================
print("\n📊 Step 8: Train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✅ Train set: {X_train.shape}")
print(f"   ✅ Test set: {X_test.shape}")

# ============================================================================
# STEP 9: SCALING (NO SMOTE TO AVOID ERRORS)
# ============================================================================
print("\n⚖️ Step 9: Scaling features...")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   ✅ Scaling complete")

# ============================================================================
# STEP 10: TRAIN RANDOM FOREST (WITH CLASS WEIGHT FOR IMBALANCE)
# ============================================================================
print("\n🌲 Step 10: Training Random Forest model...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',  # This handles imbalance without SMOTE
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
print(f"   ✅ Model trained successfully!")

# ============================================================================
# STEP 11: EVALUATION
# ============================================================================
print("\n📈 Step 11: Evaluating model...")

# Predictions
y_pred = rf_model.predict(X_test_scaled)
y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred),
    'AUC-ROC': roc_auc_score(y_test, y_proba)
}

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')

print("\n" + "="*70)
print("📊 MODEL PERFORMANCE")
print("="*70)
for metric, value in metrics.items():
    print(f"   {metric:10}: {value:.4f}")
print(f"   CV AUC    : {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)
print(f"                 Predicted")
print(f"                 No    Yes")
print(f"   Actual No    {cm[0,0]:3}    {cm[0,1]:3}")
print(f"          Yes   {cm[1,0]:3}    {cm[1,1]:3}")

# ============================================================================
# STEP 12: FEATURE IMPORTANCE
# ============================================================================
print("\n🔍 Step 12: Feature importance...")

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 Most Important Features:")
print("   " + "-"*40)
for idx, row in importance_df.head(10).iterrows():
    print(f"   {row['feature']:25}: {row['importance']:.4f} ({row['importance']*100:.2f}%)")

# ============================================================================
# STEP 13: SHAP EXPLANATIONS
# ============================================================================
print("\n🔮 Step 13: Generating SHAP explanations...")

try:
    # Use a smaller sample for SHAP to save time
    sample_size = min(50, len(X_test_scaled))
    X_sample = X_test_scaled[:sample_size]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ SHAP summary plot saved")
except Exception as e:
    print(f"   ⚠️ SHAP plot skipped: {e}")

# ============================================================================
# STEP 14: CREATE VISUALIZATIONS
# ============================================================================
print("\n📊 Step 14: Creating visualizations...")

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("   ✅ Confusion matrix saved")

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC = {metrics["AUC-ROC"]:.3f})')
plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()
print("   ✅ ROC curve saved")

# 3. Feature Importance Plot
plt.figure(figsize=(10, 8))
top_features = importance_df.head(15)
colors = plt.cm.RdYlGn_r(top_features['importance'].values)
plt.barh(range(len(top_features)), top_features['importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()
print("   ✅ Feature importance plot saved")

# ============================================================================
# STEP 15: SAVE ALL ARTIFACTS
# ============================================================================
print("\n💾 Step 15: Saving artifacts...")

# ── Compute patient-mode defaults for hidden clinical fields ──────────────────
# These are the median (numeric) or mode (categorical) values from the ENCODED
# training set, used whenever a patient cannot answer a clinical question.
# They are stored in the artifact so app.py uses training-data-derived values,
# not hardcoded guesses.
patient_hidden_fields = ['restecg', 'oldpeak', 'slope', 'ca', 'thal']
patient_defaults_encoded = {}
for col in patient_hidden_fields:
    if col in X.columns:
        col_data = X[col].dropna()
        if col in ['oldpeak', 'ca']:  # numeric — use median
            patient_defaults_encoded[col] = float(col_data.median())
        else:  # encoded categorical — use mode
            patient_defaults_encoded[col] = float(col_data.mode()[0])
print(f"   ✅ Patient defaults (encoded): {patient_defaults_encoded}")

# Also store the original (pre-encoded) mode labels for readable logging
patient_defaults_labels = {}
for col in ['restecg', 'slope', 'thal']:
    if col in label_encoders:
        enc_val = int(patient_defaults_encoded.get(col, 0))
        patient_defaults_labels[col] = label_encoders[col].inverse_transform([enc_val])[0]
patient_defaults_labels['oldpeak'] = patient_defaults_encoded.get('oldpeak', 0.0)
patient_defaults_labels['ca'] = patient_defaults_encoded.get('ca', 0)
print(f"   ✅ Patient defaults (labels): {patient_defaults_labels}")

# Median BP & cholesterol from training data — used when patient selects "I don't know"
patient_screening_medians = {}
for _col in ('trestbps', 'chol'):
    if _col in X.columns:
        patient_screening_medians[_col] = float(X[_col].median())
print(f"   ✅ Patient screening medians (BP/chol): {patient_screening_medians}")

# Save everything
artifacts = {
    'model': rf_model,
    'scaler': scaler,
    'feature_names': feature_names,
    'label_encoders': label_encoders,
    'feature_importance': importance_df,
    'model_metrics': metrics,
    'patient_defaults_encoded': patient_defaults_encoded,   # numeric encoded values
    'patient_defaults_labels': patient_defaults_labels,     # human-readable labels
    'patient_screening_medians': patient_screening_medians,
}

joblib.dump(artifacts, 'heart_disease_model_artifacts.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
importance_df.to_csv('feature_importance.csv', index=False)

print("   ✅ All artifacts saved!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print("\n📦 GENERATED FILES:")
print("   • random_forest_model.pkl")
print("   • scaler.pkl")
print("   • feature_names.pkl")
print("   • label_encoders.pkl")
print("   • feature_importance.csv")
print("   • shap_summary_plot.png")
print("   • confusion_matrix.png")
print("   • roc_curve.png")
print("   • feature_importance.png")

print("\n🚀 Your model is ready! Run: streamlit run app.py")
print("="*70)

# Test prediction
print("\n🎯 Quick Test:")
sample = X_test_scaled[0:1]
pred = rf_model.predict(sample)[0]
proba = rf_model.predict_proba(sample)[0][1]
print(f"   Sample prediction: {'High Risk' if pred == 1 else 'Low Risk'} ({proba:.2%})")
print(f"   Actual: {'Disease' if y_test.iloc[0] == 1 else 'No Disease'}")