"""Node : Symptom Classifier - Classify symptom descriptions"""

# TODO: Implement symptom_classifier_node to invoke LLM-based symptom classification if description provided
# PURPOSE: Optional node to classify free-text symptom description into structured medical categories with red flag detection
# PARAMETERS: Workflow state with optional symptom_description in extracted_data
# RETURNS: State with symptom_classification populated or status marked as skipped if description not provided

from agents.symptom_classifier_llm import SymptomClassifierLLMAgent


def symptom_classifier_node(state: dict) -> dict:
    """Classify symptoms from patient description"""
    extracted_data = state.get('extracted_data', {})
    symptom_description = extracted_data.get('symptom_description')
    
    if not symptom_description:
        state['symptom_classification'] = {}
        return state
    
    agent = SymptomClassifierLLMAgent()
    classification = agent.classify_symptoms(symptom_description, extracted_data)
    
    state['symptom_classification'] = classification
    
    return state
