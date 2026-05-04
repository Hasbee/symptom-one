"""Node : Low Risk Advice - Generate reassuring health advice (low-risk path)"""

# TODO: Implement low_risk_advice_node to invoke LLM-based reassuring guidance for low-risk patients
# PURPOSE: Generate reassuring guidance with self-care recommendations, OTC options, warning signs for low-risk pathway only
# PARAMETERS: Workflow state with extracted_data, risk_score, severity_level, is_high_risk=False
# RETURNS: State with health_advice text plus explicitly marked differential_diagnoses and treatment_plan with status="low_risk_path" to distinguish intentional skip from error

from agents.health_advice_llm import HealthAdviceLLMAgent


def low_risk_advice_node(state: dict) -> dict:
    """Generate reassuring health advice for low-risk patients"""
    extracted_data = state.get('extracted_data', {})
    risk_score = state.get('risk_score', 0.0)
    severity_level = state.get('severity_level', 'Unknown')
    
    agent = HealthAdviceLLMAgent()
    advice = agent.generate_health_advice(
        extracted_data,
        risk_score,
        severity_level,
        is_high_risk=False
    )
    
    state['health_advice'] = advice
    state['differential_diagnoses'] = {'status': 'low_risk_path'}
    state['treatment_plan'] = {'status': 'low_risk_path'}
    
    return state
