"""Google Gemini LLM Client with API key rotation and robust error handling"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

genai = None
is_new_api = False

try:
    from google import genai as genai_new
    genai = genai_new
    is_new_api = True
except (ImportError, AttributeError):
    try:
        from google import generativeai as genai_old
        genai = genai_old
        is_new_api = False
    except ImportError:
        genai = None


class GeminiClient:
    """Client for Google Gemini LLM API with key rotation and robust error handling"""

    def __init__(self, model: str = None):
        """
        Initialize Gemini client

        Args:
            model: Model name (default: from GEMINI_MODEL env or gemini-2.5-flash-lite)
        """
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

        # Load API keys from environment
        keys = []
        for i in range(1, 5):
            env_key_i = os.getenv(f"GEMINI_API_KEY_{i}")
            if env_key_i and env_key_i not in keys:
                keys.append(env_key_i)
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key and env_key not in keys:
            keys.append(env_key)

        self.api_keys = keys
        self.current_key_index = 0
        self.client = None

        if not self.api_keys:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY_1..4 or GEMINI_API_KEY in .env file")

        if genai is None:
            raise ImportError("google-generativeai is not installed. Install with: pip install google-generativeai")

        # Initialize client with current API key
        if self.current_key_index >= len(self.api_keys):
            raise ValueError("No valid API keys available for Gemini")

        current_key = self.api_keys[self.current_key_index]
        if is_new_api:
            self.client = genai.Client(api_key=current_key)
        else:
            genai.configure(api_key=current_key)
            self.client = genai.GenerativeModel(self.model)

        logger.info(f"Initialized Gemini client with key {self.current_key_index + 1}/{len(self.api_keys)}")

    def _rotate_api_key(self):
        """Rotate to next available API key"""
        self.current_key_index += 1
        if self.current_key_index >= len(self.api_keys):
            raise ValueError(f"All {len(self.api_keys)} API keys exhausted")

        # Initialize client with next API key
        current_key = self.api_keys[self.current_key_index]
        if is_new_api:
            self.client = genai.Client(api_key=current_key)
        else:
            genai.configure(api_key=current_key)
            self.client = genai.GenerativeModel(self.model)

        logger.info(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")

    def _generate_content(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate content with automatic API key rotation on quota exceeded."""
        max_retries = len(self.api_keys)
        attempt = 0

        while attempt < max_retries:
            try:
                generation_config = {"temperature": temperature, "max_output_tokens": max_tokens}

                if is_new_api:
                    response = self.client.models.generate_content(
                        model=self.model, contents=prompt, config=generation_config
                    )
                    response_text = response.text if hasattr(response, "text") else str(response)
                else:
                    response = self.client.generate_content(prompt, generation_config=generation_config)
                    response_text = response.text if hasattr(response, "text") else str(response)

                if not response_text:
                    raise ValueError("Empty response from LLM")

                return response_text

            except Exception as e:
                error_str = str(e)

                if any(quota_indicator in error_str.lower()
                       for quota_indicator in ["429", "quota", "rate limit", "exceeded", "per_minute", "per_day"]):
                    logger.warning(f"Quota exceeded on key {self.current_key_index + 1}: {error_str}")
                    attempt += 1

                    if attempt < max_retries:
                        try:
                            self._rotate_api_key()
                            logger.info(f"Retrying with next API key ({self.current_key_index + 1}/{len(self.api_keys)})")
                            continue
                        except ValueError as rotate_error:
                            raise ValueError(f"All API keys exhausted or invalid: {rotate_error}")
                    else:
                        raise ValueError(f"All {len(self.api_keys)} API keys quota exceeded")
                else:
                    logger.error(f"LLM API error: {error_str}")
                    raise ValueError(f"Failed to generate content from LLM: {error_str}")

        raise ValueError("Failed to generate content after all retries")

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response with robust fallback strategies."""
        try:
            # Clean JSON text: remove markdown fences
            cleaned = response_text.strip()
            cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
            cleaned = re.sub(r"```", "", cleaned, flags=re.MULTILINE)
            cleaned = cleaned.strip()

            candidates = [cleaned]

            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                candidates.append(json_match.group(0))

            last_error = None
            for candidate in candidates:
                try:
                    # Strip trailing commas before closing braces/brackets
                    repaired = candidate
                    previous = None
                    while repaired != previous:
                        previous = repaired
                        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

                    parsed_json = json.loads(repaired)
                    if not isinstance(parsed_json, dict):
                        raise ValueError("LLM response is not a valid JSON object")
                    return parsed_json
                except json.JSONDecodeError as parse_error:
                    try:
                        nested_matches = re.findall(r"\{[^{}]*\}", candidate)
                        for nested in nested_matches:
                            try:
                                parsed_json = json.loads(nested)
                                if isinstance(parsed_json, dict):
                                    return parsed_json
                            except Exception:
                                continue
                    except Exception:
                        pass
                    last_error = parse_error
                    continue
                except Exception as parse_error:
                    last_error = parse_error
                    continue

            raise ValueError(f"Invalid JSON in LLM response: {last_error}")

        except Exception as e:
            logger.error(f"Response extraction error: {str(e)}")
            raise ValueError(f"Failed to extract JSON from LLM response: {str(e)}")

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate JSON response from Gemini.

        Args:
            prompt: Input prompt (should request JSON output)

        Returns:
            Parsed JSON dictionary
        """
        text_response = self._generate_content(prompt)
        return self._extract_json_from_response(text_response)

    def generate_text(self, prompt: str) -> str:
        """
        Generate text response from Gemini.

        Args:
            prompt: Input prompt text

        Returns:
            Generated text response
        """
        return self._generate_content(prompt)
