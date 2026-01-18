"""
Local inference adapter for MLX pipeline.

Connects to the local FastAPI model server for text generation
instead of using external APIs like Anthropic.
"""

import os
import time
import logging
import requests
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LocalModelConfig:
    """Configuration for local model server."""

    base_url: str = "http://localhost:8000"
    default_model: str = "Qwen3-32B"  # Good for creative writing tasks
    reasoning_model: str = "QwQ-32B"  # For complex analysis
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 2.0


class LocalInferenceClient:
    """
    Client for local MLX inference server.

    Connects to FastAPI server running at localhost:8000
    with endpoints for model loading and text generation.
    """

    def __init__(self, config: Optional[LocalModelConfig] = None):
        """
        Initialize the local inference client.

        Args:
            config: Configuration for the local server.
        """
        self.config = config or LocalModelConfig()
        self._session = requests.Session()
        self._loaded_models = set()

    def health_check(self) -> bool:
        """Check if the local server is running."""
        try:
            response = self._session.get(
                f"{self.config.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def load_model(self, model_name: str) -> bool:
        """
        Load a model into memory on the server.

        Args:
            model_name: Name of the model to load.

        Returns:
            True if loaded successfully.
        """
        if model_name in self._loaded_models:
            return True

        try:
            response = self._session.post(
                f"{self.config.base_url}/models/load",
                json={"model": model_name},
                timeout=300  # Models can take time to load
            )

            if response.status_code == 200:
                self._loaded_models.add(model_name)
                logger.info(f"Loaded model: {model_name}")
                return True
            else:
                logger.error(f"Failed to load model: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Error loading model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        top_p: float = 0.9,
        stop_sequences: Optional[list[str]] = None
    ) -> str:
        """
        Generate text using the local model.

        Args:
            prompt: The input prompt.
            model: Model to use (defaults to config default).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            stop_sequences: Sequences to stop generation.

        Returns:
            Generated text.
        """
        model = model or self.config.default_model

        # Ensure model is loaded
        if model not in self._loaded_models:
            self.load_model(model)

        request_data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": stop_sequences or []
        }

        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = self._session.post(
                    f"{self.config.base_url}/models/generate",
                    json=request_data,
                    timeout=self.config.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    last_error = f"Server error: {response.status_code} - {response.text}"
                    logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                logger.warning(f"Attempt {attempt + 1}: {last_error}")

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}: {last_error}")

            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay * (attempt + 1))

        raise RuntimeError(f"Failed after {self.config.max_retries} attempts: {last_error}")

    def get_model_status(self) -> dict:
        """Get status of loaded models."""
        try:
            response = self._session.get(
                f"{self.config.base_url}/models/status",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting status: {e}")

        return {"loaded_models": list(self._loaded_models)}

    def get_memory_status(self) -> dict:
        """Get memory usage statistics."""
        try:
            response = self._session.get(
                f"{self.config.base_url}/models/memory",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting memory: {e}")

        return {}

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        try:
            response = self._session.post(
                f"{self.config.base_url}/models/unload",
                json={"model": model_name},
                timeout=30
            )

            if response.status_code == 200:
                self._loaded_models.discard(model_name)
                logger.info(f"Unloaded model: {model_name}")
                return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Error unloading model: {e}")

        return False


# Recommended models for different tasks
RECOMMENDED_MODELS = {
    "style_analysis": "Qwen3-32B",      # Good at understanding writing patterns
    "content_replacement": "Qwen3-32B", # Creative writing with style matching
    "continuity_check": "QwQ-32B",      # Reasoning for logical consistency
    "general": "Qwen3-32B"
}
