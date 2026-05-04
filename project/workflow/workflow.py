"""Workflow orchestration for JSON-based assessment execution (e.g. Streamlit)."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional


class SymptomOneWorkflow:
    """
    Runs the SymptomOne LangGraph pipeline with error handling and optional graph reuse.

    Caches the compiled graph after the first run so repeated calls (e.g. from Streamlit)
    avoid rebuilding the workflow.
    """

    def __init__(self) -> None:
        self._compiled_graph: Any = None

    def _get_compiled_graph(self) -> Any:
        if self._compiled_graph is None:
            from graph import create_symptom_one_graph

            self._compiled_graph = create_symptom_one_graph()
        return self._compiled_graph

    def run_workflow(
        self,
        patient_data: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full assessment workflow.

        Parameters
        ----------
        patient_data : dict
            Eleven clinical fields expected by nodes/agents (see user_input_node / ML agents).
        session_id : str, optional
            If omitted, a UUID is assigned when building initial state.

        Returns
        -------
        dict
            Final LangGraph state on success, or ``{"error": str, "session_id": str}`` on failure.
        """
        if not isinstance(patient_data, dict):
            return {
                "error": "patient_data must be a dictionary",
                "session_id": session_id or str(uuid.uuid4()),
            }

        resolved_session = session_id or ""
        try:
            from graph import run_symptom_one_workflow
            final_state = run_symptom_one_workflow(patient_data, session_id)
            
            if not isinstance(final_state, dict):
                return {
                    "error": "Workflow returned unexpected result type",
                    "session_id": resolved_session,
                }
            return final_state
        except Exception as exc:
            return {
                "error": str(exc),
                "session_id": resolved_session or str(uuid.uuid4()),
            }


workflow_instance = SymptomOneWorkflow()



"""Workflow Orchestration - JSON-based assessment execution"""

# TODO: Implement SymptomOneWorkflow class with run_workflow() method to orchestrate complete assessment execution
# PURPOSE: Orchestrate workflow execution with error handling, session management, and result return for Streamlit integration
# PARAMETERS: Patient data dictionary (11 clinical fields), optional session ID (auto-generated if not provided)
# RETURNS: Complete final state from run_symptom_one_workflow() or error dictionary with error message and session ID

# TODO: Implement global workflow_instance for Streamlit integration
# PURPOSE: Create singleton SymptomOneWorkflow instance used by main.py for workflow execution
