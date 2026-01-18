"""
Delegation Toolkit - Simplified AI orchestration for Claude Code.

Tools:
- delegate_code: Route code generation to Jules/Gemini/QwenAgent
- delegate_research: Route research to Perplexity
- delegation_status: Check quotas and executor health
- run_tests: Execute tests locally

Key improvement over previous SDK:
- context_files is MANDATORY to prevent executor hallucination
- Smart routing based on task complexity and quota availability
- Simplified interface (4 tools vs 30+)
"""

from .server import mcp, delegate_code, delegate_research, delegation_status, run_tests
from .quota import QuotaManager

__all__ = [
    "mcp",
    "delegate_code",
    "delegate_research",
    "delegation_status",
    "run_tests",
    "QuotaManager",
]

__version__ = "1.0.0"
