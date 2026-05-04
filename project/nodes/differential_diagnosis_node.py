"""Node : Differential Diagnosis - Generate diagnoses (high-risk path only)"""

# TODO: Implement differential_diagnosis_node to invoke LLM-based diagnosis generation for high-risk patients
# PURPOSE: Generate 5-7 differential diagnoses with probabilities for high-risk pathway only (skipped for low-risk)
# PARAMETERS: Workflow state with extracted_data, severity_level, optional symptom_classification with red flags
# RETURNS: State with differential_diagnoses dictionary containing diagnoses list with probability and reasoning

from agents.differential_diagnosis_llm import DifferentialDiagnosisLLMAgent


def differential_diagnosis_node(state: dict) -> dict:
    """Generate differential diagnoses for high-risk patients"""
    extracted_data = state.get('extracted_data', {})
    severity_level = state.get('severity_level', 'Unknown')
    symptom_classification = state.get('symptom_classification', {})
    
    agent = DifferentialDiagnosisLLMAgent()
    diagnoses = agent.assess_differential_diagnoses(
        extracted_data,
        severity_level,
        symptom_classification
    )
    
    state['differential_diagnoses'] = diagnoses
    
    return state
