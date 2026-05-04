"""Node : Validation - Validate final output"""

# TODO: Implement validation_node to validate completeness and quality of final assessment output
# PURPOSE: Non-blocking validation to check that critical fields (severity_level, risk_score, health_advice, report) are present and non-empty
# PARAMETERS: Workflow state with all assessment results
# RETURNS: State with validation_status ("valid", "warning", or "error") and accumulated validation_errors list, non-blocking to allow workflow completion


def validation_node(state: dict) -> dict:
    """Validate final assessment output"""
    validation_errors = state.get('validation_errors', [])
    
    severity_level = state.get('severity_level')
    risk_score = state.get('risk_score')
    health_advice = state.get('health_advice')
    
    if not severity_level:
        validation_errors.append("Missing severity_level")
    if risk_score is None:
        validation_errors.append("Missing risk_score")
    if not health_advice:
        validation_errors.append("Missing health_advice")
    
    if validation_errors:
        state['validation_status'] = 'warning'
    else:
        state['validation_status'] = 'valid'
    
    state['validation_errors'] = validation_errors
    
    return state
