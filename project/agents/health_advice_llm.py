"""Health Advice LLM Agent"""

# TODO: Implement HealthAdviceLLMAgent to generate risk-appropriate personalized clinical guidance
# PURPOSE: Use LLM to create actionable health advice tailored to risk level (urgent for high-risk, reassuring for low-risk)
# PARAMETERS: Extracted fields dictionary, risk score float (0.0-1.0), severity level string, boolean is_high_risk flag, optional diagnoses list
# RETURNS: String containing personalized clinical guidance text (emergency instructions for high-risk, self-care for low-risk)

from typing import Dict, Any, Optional
from agents.base_llm_agent import BaseLLMAgent
from utils.gemini_client import GeminiClient


class HealthAdviceLLMAgent(BaseLLMAgent):
    """Generate health advice based on risk level"""
    
    def __init__(self, client: Optional[GeminiClient] = None):
        """Initialize health advice agent"""
        super().__init__(client)
    
    def generate_health_advice(
        self,
        patient_data: Dict[str, Any],
        risk_score: float,
        severity_level: str,
        is_high_risk: bool
    ) -> str:
        """Generate personalized health advice"""
        if is_high_risk:
            prompt = f"""Generate URGENT health advice for high-risk patient.
Risk Score: {risk_score}
Severity: {severity_level}
Return critical alert message."""
            advice = self.client.generate_text(prompt)
            if not advice:
                advice = 'CRITICAL ALERT: You require immediate medical attention. Call emergency services or visit the nearest hospital immediately.'
        else:
            prompt = f"""Generate reassuring health advice for low-risk patient.
Risk Score: {risk_score}
Severity: {severity_level}
Return supportive guidance."""
            advice = self.client.generate_text(prompt)
            if not advice:
                advice = 'Your symptoms appear manageable. Rest, hydrate well, and monitor your symptoms. Contact a healthcare provider if symptoms persist.'
        
        return advice

