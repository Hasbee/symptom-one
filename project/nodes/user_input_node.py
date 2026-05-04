"""Node : Validate JSON Input - Entry point for workflow"""

# TODO: Implement user_input_node to validate 11 required clinical fields in patient data
# PURPOSE: Entry point node to ensure all required clinical fields are present before proceeding with assessment
# PARAMETERS: Workflow state containing extracted_data with patient information
# RETURNS: Updated state with validation_errors appended if fields missing, validation non-blocking to allow workflow continuation

from typing import Any


REQUIRED_FIELDS = [
    'patient_age_years',
    'symptom_duration_hours',
    'fever_present',
    'neck_stiffness',
    'body_temperature_celsius',
    'heart_rate_bpm',
    'blood_pressure_systolic_mmhg',
    'blood_pressure_diastolic_mmhg',
    'respiratory_rate_breaths_per_minute',
    'oxygen_saturation_percent',
    'comorbidities_count'
]


def user_input_node(state: dict) -> dict:
    """Validate required clinical fields in patient data"""
    extracted_data = state.get('extracted_data', {})
    validation_errors = state.get('validation_errors', [])
    
    for field in REQUIRED_FIELDS:
        if field not in extracted_data:
            validation_errors.append(f"Missing required field: {field}")
    
    state['validation_errors'] = validation_errors
    state['validation_status'] = 'complete'
    
    return state
