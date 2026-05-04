"""Node : Output Formatting - Format final output for UI"""

# TODO: Implement output_formatting_node to format assessment results for Streamlit UI display
# PURPOSE: Final node to prepare all results in ui_output dictionary format with risk level calculation, completion percentage, and status markers
# PARAMETERS: Workflow state with session_id, severity_level, risk_score, health_advice, diagnoses, treatment, validation_status, errors
# RETURNS: State with ui_output dictionary formatted for 4-tab Streamlit display (Symptom Analysis, Clinical Data, Diagnosis & Treatment, Clinical Guidance)


def output_formatting_node(state: dict) -> dict:
    """Format assessment results for UI display"""
    risk_score = state.get('risk_score', 0.0)
    
    if risk_score >= 0.6:
        risk_level = 'HIGH RISK'
    else:
        risk_level = 'LOW RISK'
    
    ui_output = {
        'session_id': state.get('session_id'),
        'risk_level': risk_level,
        'risk_score': risk_score,
        'severity_level': state.get('severity_level'),
        'health_advice': state.get('health_advice'),
        'differential_diagnoses': state.get('differential_diagnoses', {}),
        'treatment_plan': state.get('treatment_plan', {}),
        'symptom_classification': state.get('symptom_classification', {}),
        'validation_status': state.get('validation_status'),
        'validation_errors': state.get('validation_errors', [])
    }
    
    state['ui_output'] = ui_output
    
    return state
