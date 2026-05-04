"""Node : Severity Assessment - Assess severity using ML"""

# TODO: Implement severity_assessment_node to invoke ML-based severity classification
# PURPOSE: Classify patient severity level (Low/Moderate/High/Critical) using pre-trained RandomForest classifier on vital signs
# PARAMETERS: Workflow state with extracted_data containing 11 clinical features
# RETURNS: State with severity_level populated and stored in ml_results dictionary

from agents.severity_assessment_ml import SeverityAssessmentMLAgent


def severity_assessment_node(state: dict) -> dict:
    """Assess patient severity level using ML model"""
    agent = SeverityAssessmentMLAgent()
    extracted_data = state.get('extracted_data', {})
    
    severity_level = agent.predict_severity_classification(extracted_data)
    
    state['severity_level'] = severity_level
    state['ml_results'] = state.get('ml_results', {})
    state['ml_results']['severity_level'] = severity_level
    
    return state
