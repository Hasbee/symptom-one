"""Node : Risk Score Assessment - Calculate risk score using ML"""

# TODO: Implement risk_score_assessment_node to invoke ML-based risk score prediction
# PURPOSE: Calculate numerical risk score (0.0-1.0) using pre-trained RandomForest regressor for clinical routing decision
# PARAMETERS: Workflow state with extracted_data containing 11 clinical features
# RETURNS: State with risk_score (float 0.0-1.0) populated and stored in ml_results dictionary

from agents.risk_score_ml import RiskScoreMLAgent


def risk_score_assessment_node(state: dict) -> dict:
    """Assess patient risk score using ML model"""
    agent = RiskScoreMLAgent()
    extracted_data = state.get('extracted_data', {})
    
    risk_score = agent.predict_clinical_risk_score(extracted_data)
    
    state['risk_score'] = risk_score
    state['ml_results'] = state.get('ml_results', {})
    state['ml_results']['risk_score'] = risk_score
    
    return state
