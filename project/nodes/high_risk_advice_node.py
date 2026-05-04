"""Node : High Risk Advice - Generate urgent health advice (high-risk path)"""

# TODO: Implement high_risk_advice_node to invoke LLM-based urgent guidance generation for high-risk patients
# PURPOSE: Generate CRITICAL ALERT with emergency instructions, hospital expectations, and warning signs for high-risk pathway only
# PARAMETERS: Workflow state with extracted_data, risk_score, severity_level, is_high_risk=True, differential_diagnoses list
# RETURNS: State with health_advice text containing urgent, actionable clinical guidance appropriate for emergency situations

from agents.health_advice_llm import HealthAdviceLLMAgent


def high_risk_advice_node(state: dict) -> dict:
    """Generate urgent health advice for high-risk patients"""
    extracted_data = state.get('extracted_data', {})
    risk_score = state.get('risk_score', 0.0)
    severity_level = state.get('severity_level', 'Unknown')
    
    agent = HealthAdviceLLMAgent()
    advice = agent.generate_health_advice(
        extracted_data,
        risk_score,
        severity_level,
        is_high_risk=True
    )
    
    state['health_advice'] = advice
    
    return state
