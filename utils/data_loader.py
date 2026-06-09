import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_preprocess_data(filepath='data/heart_disease_uci.csv'):
    """Load the dataset and preprocess it"""
    df = pd.read_csv(filepath)
    
    # Drop id column
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Handle missing values
    df['ca'] = df['ca'].fillna(0)
    df['thal'] = df['thal'].fillna('normal')
    
    # Encode categorical features
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'origin']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Prepare features and target
    X = df.drop('num', axis=1)
    y = (df['num'] > 0).astype(int)  # Binary classification
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    return X, y, feature_names, label_encoders

def get_feature_labels():
    """Get human-readable feature names for display"""
    return {
        'age': 'Age (years)',
        'sex': 'Gender (0=Female, 1=Male)',
        'cp': 'Chest Pain Type',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Serum Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl',
        'restecg': 'Resting ECG Results',
        'thalach': 'Maximum Heart Rate Achieved',
        'exang': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression (exercise relative to rest)',
        'slope': 'Slope of Peak Exercise ST Segment',
        'ca': 'Number of Major Vessels (0-3)',
        'thal': 'Thalassemia',
        'origin': 'Study Origin'
    }