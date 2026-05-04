"""Risk Score ML Agent"""

# TODO: Implement RiskScoreMLAgent using trained RandomForest regressor to predict numerical risk score
# PURPOSE: Load pre-trained risk score regression model and scaler, predict continuous risk value for patient prioritization
# PARAMETERS: Extracted fields dictionary with 11 clinical features (same as severity assessment)
# RETURNS: Float risk score between 0.0 (low risk) and 1.0 (high risk), clipped to valid range, with 0.6 threshold for clinical routing

import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np


class RiskScoreMLAgent:
    """Predict clinical risk score from patient data using trained model"""
    
    def __init__(self):
        """Load trained risk score model"""
        self.model_path = Path(__file__).parent.parent / "ml" / "models" / "risk_score_model.pkl"
        self.scaler_path = Path(__file__).parent.parent / "ml" / "models" / "risk_score_scaler.pkl"
        
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        except:
            self.model = None
            self.scaler = None
    
    def predict_clinical_risk_score(self, patient_data: Dict[str, Any]) -> float:
        """Predict risk score from patient data"""
        features = [
            patient_data.get('patient_age_years', 0),
            patient_data.get('symptom_duration_hours', 0),
            patient_data.get('fever_present', 0),
            patient_data.get('neck_stiffness', 0),
            patient_data.get('body_temperature_celsius', 0),
            patient_data.get('heart_rate_bpm', 0),
            patient_data.get('blood_pressure_systolic_mmhg', 0),
            patient_data.get('blood_pressure_diastolic_mmhg', 0),
            patient_data.get('respiratory_rate_breaths_per_minute', 0),
            patient_data.get('oxygen_saturation_percent', 0),
            patient_data.get('comorbidities_count', 0),
        ]
        
        if self.model is None or self.scaler is None:
            return 0.5
        
        features_array = np.array([features])
        features_scaled = self.scaler.transform(features_array)
        prediction = self.model.predict(features_scaled)[0]
        
        return float(np.clip(prediction, 0.0, 1.0))
