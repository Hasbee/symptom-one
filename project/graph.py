"""SymptomOne LangGraph Workflow - Graph construction and orchestration"""

# TODO: Implement route_based_on_risk()
# PURPOSE: Route patients to appropriate clinical pathway based on risk score threshold (0.6)
# PARAMETERS: Workflow state containing risk_score
# RETURNS: Literal["diagnose", "advise_low_risk"] routing decision string

# TODO: Implement create_symptom_one_graph()
# PURPOSE: Construct 13-node LangGraph workflow with conditional routing based on ML risk assessment
# PARAMETERS: None
# RETURNS: Compiled StateGraph object configured with all nodes and edges

# TODO: Implement run_symptom_one_workflow()
# PURPOSE: Execute complete workflow pipeline through all 13 nodes with JSON patient data input
# PARAMETERS: Patient data dictionary (11 clinical fields), optional session ID
# RETURNS: Final workflow state with assessment results and ui_output formatted for display

from langgraph.graph import StateGraph, END
from typing import Literal
from state import SymptomOneState, create_initial_state

from nodes.user_input_node import user_input_node
from nodes.symptom_classifier_node import symptom_classifier_node
from nodes.severity_assessment_node import severity_assessment_node
from nodes.risk_score_assessment_node import risk_score_assessment_node
from nodes.risk_path_router_node import risk_path_router_node
from nodes.differential_diagnosis_node import differential_diagnosis_node
from nodes.treatment_plan_node import treatment_plan_node
from nodes.high_risk_advice_node import high_risk_advice_node
from nodes.low_risk_advice_node import low_risk_advice_node
from nodes.validation_node import validation_node
from nodes.output_formatting_node import output_formatting_node
from nodes.report_compilation_node import report_compilation_node
from nodes.save_report_node import save_report_node


def route_based_on_risk(state: SymptomOneState) -> Literal["diagnose", "advise_low_risk"]:
    """Route patients to appropriate clinical pathway based on risk score threshold (0.6)"""
    risk_path = state.get('risk_path')
    
    if risk_path == 'high_risk_path':
        return "diagnose"
    else:
        return "advise_low_risk"


def create_symptom_one_graph():
    """Construct 13-node LangGraph workflow with conditional routing based on ML risk assessment"""
    graph = StateGraph(SymptomOneState)
    
    # Add all nodes
    graph.add_node("user_input", user_input_node)
    graph.add_node("symptom_classifier", symptom_classifier_node)
    graph.add_node("severity_assessment", severity_assessment_node)
    graph.add_node("risk_score_assessment", risk_score_assessment_node)
    graph.add_node("risk_path_router", risk_path_router_node)
    graph.add_node("differential_diagnosis", differential_diagnosis_node)
    graph.add_node("treatment_plan_generation", treatment_plan_node)
    graph.add_node("high_risk_advice", high_risk_advice_node)
    graph.add_node("low_risk_advice", low_risk_advice_node)
    graph.add_node("validation", validation_node)
    graph.add_node("output_formatting", output_formatting_node)
    graph.add_node("report_compilation", report_compilation_node)
    graph.add_node("save_report", save_report_node)
    
    # Set entry point
    graph.set_entry_point("user_input")
    
    # Add edges for sequential processing
    graph.add_edge("user_input", "symptom_classifier")
    graph.add_edge("symptom_classifier", "severity_assessment")
    graph.add_edge("severity_assessment", "risk_score_assessment")
    graph.add_edge("risk_score_assessment", "risk_path_router")
    
    # Conditional routing based on risk path
    graph.add_conditional_edges(
        "risk_path_router",
        route_based_on_risk,
        {
            "diagnose": "differential_diagnosis",
            "advise_low_risk": "low_risk_advice"
        }
    )
    
    # High-risk path
    graph.add_edge("differential_diagnosis", "treatment_plan_generation")
    graph.add_edge("treatment_plan_generation", "high_risk_advice")
    graph.add_edge("high_risk_advice", "validation")
    
    # Low-risk path
    graph.add_edge("low_risk_advice", "validation")
    
    # Convergence and final steps
    graph.add_edge("validation", "output_formatting")
    graph.add_edge("output_formatting", "report_compilation")
    graph.add_edge("report_compilation", "save_report")
    graph.add_edge("save_report", END)
    
    return graph.compile()


def run_symptom_one_workflow(patient_data: dict, session_id: str = None) -> dict:
    """Execute complete workflow pipeline through all 13 nodes with JSON patient data input"""
    initial_state = create_initial_state(patient_data, session_id)
    graph = create_symptom_one_graph()
    
    final_state = graph.invoke(initial_state)
    
    return final_state
