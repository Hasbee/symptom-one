"""State schema for SymptomOne LangGraph workflow"""

# TODO: Implement SymptomOneState TypedDict with 15 state fields for complete workflow state management
# PURPOSE: Define immutable state schema using TypedDict (total=False) to track patient data, assessments, classifications, diagnoses, treatment, advice, validation, and UI output through all 13 workflow nodes
# INCLUDES: Session management, extracted_data, symptom_classification, severity_level, risk_score, ml_results, risk_path, differential_diagnoses, treatment_plan, health_advice, validation_status, validation_errors, ui_output, report_file_path, report_json
# RETURNS: TypedDict class definition with all 15 fields properly typed

# TODO: Implement create_initial_state() function to initialize workflow state from JSON patient data
# PURPOSE: Create fresh SymptomOneState instance with auto-generated session ID and patient data, setting all fields to proper initial values (None, empty dicts, zero values)
# PARAMETERS: Patient data dictionary with 11 clinical fields, optional session_id string (auto-generates UUID if not provided)
# RETURNS: Fully initialized SymptomOneState object ready for workflow invocation

from typing import TypedDict, Optional, Dict, Any, List
import uuid


class SymptomOneState(TypedDict, total=False):
    """State schema for SymptomOne workflow"""
    session_id: str
    extracted_data: Dict[str, Any]
    symptom_classification: Dict[str, Any]
    severity_level: str
    risk_score: float
    ml_results: Dict[str, Any]
    risk_path: str
    differential_diagnoses: Dict[str, Any]
    treatment_plan: Dict[str, Any]
    health_advice: str
    validation_status: str
    validation_errors: List[str]
    ui_output: Dict[str, Any]
    report_file_path: str
    report_json: Dict[str, Any]


def create_initial_state(patient_data: Dict[str, Any], session_id: Optional[str] = None) -> SymptomOneState:
    """Initialize workflow state from patient data"""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    return {
        'session_id': session_id,
        'extracted_data': patient_data,
        'symptom_classification': {},
        'severity_level': None,
        'risk_score': None,
        'ml_results': {},
        'risk_path': None,
        'differential_diagnoses': {},
        'treatment_plan': {},
        'health_advice': None,
        'validation_status': None,
        'validation_errors': [],
        'ui_output': {},
        'report_file_path': None,
        'report_json': {}
    }
