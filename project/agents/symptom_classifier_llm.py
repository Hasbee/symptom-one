"""Symptom Classifier LLM Agent"""

# TODO: Implement SymptomClassifierLLMAgent to classify symptom descriptions into medical categories
# PURPOSE: Use LLM to analyze free-text symptom descriptions with clinical context and extract structured classification
# PARAMETERS: Symptom description string, extracted clinical fields dictionary with vitals
# RETURNS: Dictionary containing primary_symptom, severity_descriptor, associated_symptoms, onset_pattern, character, location, red_flags, confidence

import json
from typing import Dict, Any, Optional
from agents.base_llm_agent import BaseLLMAgent
from utils.gemini_client import GeminiClient


class SymptomClassifierLLMAgent(BaseLLMAgent):
    """Classify symptoms from patient description"""
    
    def __init__(self, client: Optional[GeminiClient] = None):
        """Initialize symptom classifier agent"""
        super().__init__(client)
    
    def classify_symptoms(self, symptom_description: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify symptoms from description and vital signs"""
        prompt = f"""Classify the following symptom description into medical categories.
        
Symptom Description: {symptom_description}

Patient Data:
- Age: {patient_data.get('patient_age_years')}
- Temperature: {patient_data.get('body_temperature_celsius')}°C
- Heart Rate: {patient_data.get('heart_rate_bpm')} bpm
- Oxygen Saturation: {patient_data.get('oxygen_saturation_percent')}%

Return a JSON object with: primary_symptom, severity_descriptor, associated_symptoms, onset_pattern, character, location, red_flags, confidence"""
        
        response = self.client.generate_json(prompt)
        
        if not response:
            response = {
                'primary_symptom': 'Unspecified',
                'severity_descriptor': 'Moderate',
                'associated_symptoms': [],
                'onset_pattern': 'Unknown',
                'character': 'Unknown',
                'location': 'Unspecified',
                'red_flags': [],
                'confidence': 'medium'
            }
        
        return response
