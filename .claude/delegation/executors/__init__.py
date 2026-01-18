"""
Executor implementations for the Delegation Toolkit.

Each executor wraps a specific AI coding tool:
- JulesExecutor: Google Jules (multi-file, agentic)
- GeminiExecutor: Gemini CLI (fast code generation)
- QwenExecutor: QwenAgent with MLX (local, free)
- PerplexityExecutor: Research via Chrome extension
"""

from .jules import JulesExecutor
from .gemini import GeminiExecutor
from .qwen import QwenExecutor
from .perplexity import PerplexityExecutor

__all__ = [
    "JulesExecutor",
    "GeminiExecutor",
    "QwenExecutor",
    "PerplexityExecutor",
]
