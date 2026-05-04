"""Train severity classification model"""

# TODO: Implement train_severity_model() to train RandomForest classifier for 4-class severity classification
# PURPOSE: Load cleaned training data, train RandomForest (100 estimators, max_depth=15), scale features with StandardScaler, encode target with LabelEncoder
# PARAMETERS: None (reads from data/processed/severity_cleaned.csv, saves to ml/models/)
# RETURNS: Boolean indicating success; saves severity_classifier.pkl and severity_scaler.pkl to models directory with evaluation metrics printed

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_severity_model():
    """Train RandomForest classifier for severity classification"""
    try:
        # Paths
        project_root = Path(__file__).parent.parent.parent
        input_path = project_root / "data" / "training_dataset" / "severity_training.csv"
        output_path = project_root / "data" / "processed" / "severity_processed.csv"
        models_dir = project_root / "ml" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training data
        df = pd.read_csv(input_path)
        
        # Select feature columns and target
        feature_cols = [
            'patient_age_years', 'symptom_duration_hours', 'fever_present', 'neck_stiffness',
            'body_temperature_celsius', 'heart_rate_bpm', 'blood_pressure_systolic_mmhg',
            'blood_pressure_diastolic_mmhg', 'respiratory_rate_breaths_per_minute',
            'oxygen_saturation_percent', 'comorbidities_count', 'severity_level'
        ]
        
        # Select only available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        df_processed = df[available_cols].copy()
        
        # Remove rows with NaN values
        df_processed = df_processed.dropna()
        
        # Save processed data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(output_path, index=False)
        
        # Extract features and target
        X = df_processed[available_cols[:-1]]  # All except last column (severity_level)
        y = df_processed['severity_level']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        model.fit(X_scaled, y)
        
        # Evaluate
        y_pred = model.predict(X_scaled)
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Save models
        joblib.dump(model, models_dir / "severity_model.pkl")
        joblib.dump(scaler, models_dir / "severity_scaler.pkl")
        
        print(f"✓ Severity model trained successfully")
        print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error training severity model: {str(e)}")
        return False
