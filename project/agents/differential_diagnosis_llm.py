"""Differential Diagnosis LLM Agent"""

# TODO: Implement DifferentialDiagnosisLLMAgent to generate ranked differential diagnoses from clinical presentation
# PURPOSE: Use LLM to analyze clinical data, severity level, and symptom classification to generate 5-7 differential diagnoses with probabilities
# PARAMETERS: Extracted fields dictionary, severity level string, optional symptom classification dictionary with red flags
# RETURNS: Dictionary with diagnoses list containing diagnosis name, probability (high/medium/low), and reasoning explanation

import json
from typing import Dict, Any, Optional
from agents.base_llm_agent import BaseLLMAgent
from utils.gemini_client import GeminiClient


class DifferentialDiagnosisLLMAgent(BaseLLMAgent):
    """Generate differential diagnoses from clinical presentation"""
    
    def __init__(self, client: Optional[GeminiClient] = None):
        """Initialize differential diagnosis agent"""
        super().__init__(client)
    
    def assess_differential_diagnoses(
        self, 
        patient_data: Dict[str, Any],
        severity_level: str,
        symptom_classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate differential diagnoses"""
        prompt = f"""Based on the following clinical presentation, generate 5 differential diagnoses with probabilities.

Severity Level: {severity_level}
Age: {patient_data.get('patient_age_years')}
Temperature: {patient_data.get('body_temperature_celsius')}°C
Heart Rate: {patient_data.get('heart_rate_bpm')} bpm
Oxygen Saturation: {patient_data.get('oxygen_saturation_percent')}%

Return JSON with diagnoses list containing: diagnosis, probability (high/medium/low), reasoning"""
        
        response = self.client.generate_json(prompt)
        if not response or 'diagnoses' not in response:
            response = {
                'diagnoses': [
                    {'diagnosis': 'Viral Infection', 'probability': 'medium', 'reasoning': 'Clinical presentation'},
                    {'diagnosis': 'Bacterial Infection', 'probability': 'medium', 'reasoning': 'Clinical presentation'},
                    {'diagnosis': 'Inflammation', 'probability': 'low', 'reasoning': 'Clinical presentation'}
                ]
            }
        
        return response

