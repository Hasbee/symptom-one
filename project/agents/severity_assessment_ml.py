"""Severity Assessment ML Agent"""

# TODO: Implement SeverityAssessmentMLAgent using trained RandomForest classifier to predict severity
# PURPOSE: Load pre-trained severity classification model and scaler, then predict severity level from vital signs
# PARAMETERS: Extracted fields dictionary with 11 clinical features (age, duration, fever, stiffness, temp, HR, BP, RR, O2, comorbidities)
# RETURNS: String severity level classification (Low, Moderate, High, or Critical)

import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np


class SeverityAssessmentMLAgent:
    """Predict severity level from patient data using trained model"""
    
    def __init__(self):
        """Load trained severity model"""
        self.model_path = Path(__file__).parent.parent / "ml" / "models" / "severity_model.pkl"
        self.scaler_path = Path(__file__).parent.parent / "ml" / "models" / "severity_scaler.pkl"
        
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        except:
            self.model = None
            self.scaler = None
    
    def predict_severity_classification(self, patient_data: Dict[str, Any]) -> str:
        """Predict severity level from patient data"""
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
            return 'Moderate'
        
        features_array = np.array([features])
        features_scaled = self.scaler.transform(features_array)
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction
        # severity_map = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Critical'}
        # return severity_map.get(int(prediction), 'Moderate')
