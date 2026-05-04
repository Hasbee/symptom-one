"""Base LLM Agent class"""

# TODO: Implement BaseLLMAgent class as foundation for all LLM-based agents
# PURPOSE: Provide common LLM client functionality and initialization for derived agents
# PARAMETERS: Optional GeminiClient instance (creates new instance if not provided)
# RETURNS: None (Constructor initializes self.client attribute)

from typing import Optional
from utils.gemini_client import GeminiClient


class BaseLLMAgent:
    """Base class for all LLM-based agents"""
    
    def __init__(self, client: Optional[GeminiClient] = None):
        """Initialize agent with LLM client"""
        if client is None:
            self.client = GeminiClient()
        else:
            self.client = client

