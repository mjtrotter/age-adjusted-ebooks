"""
Perplexity Executor - Research via Chrome extension.

Uses Claude's Chrome extension (mcp__claude-in-chrome__*) for reliable
browser automation, replacing the previous Playwright-based approach.

This solves the asset download issue by using the Chrome extension's
native file handling capabilities.
"""

import asyncio
from pathlib import Path
from typing import Optional


class PerplexityExecutor:
    """Execute research queries via Perplexity using Chrome extension."""

    def __init__(self):
        self.name = "perplexity"
        self._tab_id: Optional[int] = None

    async def execute(
        self,
        query: str,
        mode: str = "deep",
        download_assets: bool = True,
        output_dir: Path = None
    ) -> dict:
        """
        Execute research query via Perplexity.

        This method is designed to be called from Claude Code, which has
        access to the Chrome extension tools. The actual browser automation
        should be done by the orchestrator using:

        1. mcp__claude-in-chrome__navigate to perplexity.ai
        2. mcp__claude-in-chrome__form_input for the query
        3. mcp__claude-in-chrome__computer for interactions
        4. mcp__claude-in-chrome__get_page_text for results

        Args:
            query: Research question
            mode: "quick" or "deep"
            download_assets: Whether to download generated files
            output_dir: Where to save downloaded files

        Returns:
            dict with success, result, sources, downloaded_files
        """
        # This executor returns instructions for the orchestrator
        # since it needs to use Claude's Chrome extension tools
        return {
            "success": True,
            "result": f"[ORCHESTRATOR: Execute Perplexity research using Chrome extension]\n\nQuery: {query}\n\nMode: {mode}",
            "instructions": self._get_chrome_instructions(query, mode, download_assets),
            "sources": [],
            "downloaded_files": []
        }

    def _get_chrome_instructions(self, query: str, mode: str, download_assets: bool) -> list:
        """
        Generate step-by-step instructions for Chrome extension automation.

        The orchestrator (Claude Code) should execute these using the
        mcp__claude-in-chrome__* tools.
        """
        steps = [
            {
                "step": 1,
                "action": "Get tab context",
                "tool": "mcp__claude-in-chrome__tabs_context_mcp",
                "params": {"createIfEmpty": True}
            },
            {
                "step": 2,
                "action": "Navigate to Perplexity",
                "tool": "mcp__claude-in-chrome__navigate",
                "params": {"url": "https://perplexity.ai"}
            },
            {
                "step": 3,
                "action": "Wait for page load",
                "tool": "mcp__claude-in-chrome__computer",
                "params": {"action": "wait", "duration": 2}
            },
            {
                "step": 4,
                "action": "Find search input",
                "tool": "mcp__claude-in-chrome__find",
                "params": {"query": "search input or textarea"}
            },
            {
                "step": 5,
                "action": "Enter query",
                "tool": "mcp__claude-in-chrome__form_input",
                "params": {"value": query}
            },
            {
                "step": 6,
                "action": "Submit search",
                "tool": "mcp__claude-in-chrome__computer",
                "params": {"action": "key", "text": "Enter"}
            },
            {
                "step": 7,
                "action": "Wait for results",
                "tool": "mcp__claude-in-chrome__computer",
                "params": {"action": "wait", "duration": 10 if mode == "deep" else 5}
            },
            {
                "step": 8,
                "action": "Extract results",
                "tool": "mcp__claude-in-chrome__get_page_text",
                "params": {}
            }
        ]

        if download_assets:
            steps.extend([
                {
                    "step": 9,
                    "action": "Check for downloadable assets",
                    "tool": "mcp__claude-in-chrome__find",
                    "params": {"query": "download button or code block with download"}
                },
                {
                    "step": 10,
                    "action": "Download assets if found",
                    "tool": "mcp__claude-in-chrome__computer",
                    "params": {"action": "left_click"}
                }
            ])

        return steps


# Alternative: Direct Perplexity MCP tool
# If perplexity-research MCP is available, use it instead

async def perplexity_via_mcp(query: str, mode: str = "deep") -> dict:
    """
    Alternative implementation using the Perplexity MCP server directly.

    This is used when mcp__perplexity-research__perplexity_deep_research
    is available in the tool set.
    """
    # This would be called via the MCP tool, not directly
    return {
        "success": True,
        "result": "[Use mcp__perplexity-research__perplexity_deep_research tool]",
        "query": query,
        "mode": mode
    }
