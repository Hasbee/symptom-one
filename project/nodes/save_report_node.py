"""Node : Save Report - Persist final report to output directory"""

# TODO: Implement save_report_node to persist comprehensive assessment report to JSON file in output directory
# PURPOSE: Save complete assessment with session ID, timestamp, patient data, clinical analysis, and validation results for audit trail and download
# PARAMETERS: Workflow state with all assessment results, session_id, timestamp, validation status and errors
# RETURNS: State with report_file_path (file path) and report_json (JSON string for download functionality) populated

import json
from pathlib import Path


def save_report_node(state: dict) -> dict:
    """Save assessment report to output directory"""
    report = state.get('report_json', {})
    session_id = state.get('session_id', 'unknown')
    
    output_dir = Path(__file__).parent.parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"{session_id}_report.json"
    
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        state['report_file_path'] = str(report_path)
    except Exception as e:
        state['report_file_path'] = None
    
    return state
