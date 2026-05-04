"""Node : Treatment Plan - Generate treatment plan (high-risk path)"""

# TODO: Implement treatment_plan_node to invoke LLM-based treatment planning for high-risk cases
# PURPOSE: Generate comprehensive treatment recommendations including interventions, tests, medications, monitoring, follow-up for high-risk pathway only
# PARAMETERS: Workflow state with extracted_data, severity_level, differential_diagnoses list from previous node
# RETURNS: State with treatment_plan dictionary containing immediate_interventions, recommended_tests, medications, monitoring_plan, follow_up_timeline, specialist_referrals

from agents.treatment_plan_llm import TreatmentPlanLLMAgent


def treatment_plan_node(state: dict) -> dict:
    """Generate treatment plan for high-risk patients"""
    extracted_data = state.get('extracted_data', {})
    severity_level = state.get('severity_level', 'Unknown')
    differential_diagnoses = state.get('differential_diagnoses', {})
    
    agent = TreatmentPlanLLMAgent()
    plan = agent.generate_treatment_plan(
        extracted_data,
        severity_level,
        differential_diagnoses
    )
    
    state['treatment_plan'] = plan
    
    return state
