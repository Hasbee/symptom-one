"""Train risk score regression model"""

# TODO: Implement train_risk_score_model() to train RandomForest regressor for continuous risk score prediction
# PURPOSE: Load cleaned training data, train RandomForest regressor (100 estimators, max_depth=15), scale features with StandardScaler, clip predictions to [0.0-1.0]
# PARAMETERS: None (reads from data/processed/risk_score_cleaned.csv, saves to ml/models/)
# RETURNS: Boolean indicating success; saves risk_score_regressor.pkl and risk_score_scaler.pkl with MAE, RMSE, R2 metrics printed

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_risk_score_model():
    """Train RandomForest regressor for risk score prediction"""
    try:
        # Paths
        project_root = Path(__file__).parent.parent.parent
        input_path = project_root / "data" / "training_dataset" / "risk_score_training.csv"
        output_path = project_root / "data" / "processed" / "risk_score_processed.csv"
        models_dir = project_root / "ml" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training data
        df = pd.read_csv(input_path)
        
        # Select feature columns and target
        feature_cols = [
            'patient_age_years', 'symptom_duration_hours', 'fever_present', 'neck_stiffness',
            'body_temperature_celsius', 'heart_rate_bpm', 'blood_pressure_systolic_mmhg',
            'blood_pressure_diastolic_mmhg', 'respiratory_rate_breaths_per_minute',
            'oxygen_saturation_percent', 'comorbidities_count', 'risk_score'
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
        X = df_processed[available_cols[:-1]]  # All except last column (risk_score)
        y = df_processed['risk_score']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        model.fit(X_scaled, y)
        
        # Evaluate
        y_pred = model.predict(X_scaled)
        y_pred = np.clip(y_pred, 0.0, 1.0)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Save models
        joblib.dump(model, models_dir / "risk_score_model.pkl")
        joblib.dump(scaler, models_dir / "risk_score_scaler.pkl")
        
        print(f"✓ Risk score model trained successfully")
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error training risk score model: {str(e)}")
        return False
