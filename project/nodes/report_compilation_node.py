"""Node : Report Compilation - Compile final assessment report"""

# TODO: Implement report_compilation_node to aggregate assessment results from both pathways into comprehensive report
# PURPOSE: Convergence point after high-risk and low-risk paths merge, aggregate all clinical analysis into report dictionary
# PARAMETERS: Workflow state with severity_level, risk_score, differential_diagnoses, treatment_plan, health_advice, extracted_data
# RETURNS: State with report dictionary containing aggregated assessment results from both clinical pathways


def report_compilation_node(state: dict) -> dict:
    """Compile final assessment report"""
    report = {
        'session_id': state.get('session_id'),
        'severity_level': state.get('severity_level'),
        'risk_score': state.get('risk_score'),
        'risk_path': state.get('risk_path'),
        'differential_diagnoses': state.get('differential_diagnoses', {}),
        'treatment_plan': state.get('treatment_plan', {}),
        'health_advice': state.get('health_advice'),
        'symptom_classification': state.get('symptom_classification', {}),
        'ml_results': state.get('ml_results', {}),
        'validation_status': state.get('validation_status'),
        'validation_errors': state.get('validation_errors', [])
    }
    
    state['report_json'] = report
    
    return state
