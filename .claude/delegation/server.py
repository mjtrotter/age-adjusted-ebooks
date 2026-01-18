"""
Delegation Toolkit MCP Server

A simplified FastMCP-based server that routes tasks to:
- Jules (complex multi-file, 15/day)
- Gemini CLI (primary code gen, 1500/day)
- QwenAgent (fallback, local MLX, unlimited)
- Perplexity (research only, via Chrome extension)

Key improvement: context_files is MANDATORY to prevent hallucination.
"""

from fastmcp import FastMCP, Context
from pathlib import Path
from typing import Optional
import json
import os
import sys

# Handle both package and standalone execution
_this_dir = Path(__file__).parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

try:
    from executors import JulesExecutor, GeminiExecutor, QwenExecutor, PerplexityExecutor
    from quota import QuotaManager
except ImportError:
    from .executors import JulesExecutor, GeminiExecutor, QwenExecutor, PerplexityExecutor
    from .quota import QuotaManager

mcp = FastMCP(
    name="delegation-toolkit",
    dependencies=["httpx", "aiofiles"],
)

# Initialize components
quota = QuotaManager()
jules = JulesExecutor()
gemini = GeminiExecutor()
qwen = QwenExecutor()
perplexity = PerplexityExecutor()


def _read_context_files(files: list[str]) -> str:
    """Read and concatenate context files into a single string."""
    context_parts = []
    for file_path in files:
        path = Path(file_path)
        if path.exists() and path.is_file():
            try:
                content = path.read_text()
                context_parts.append(f"### {file_path}\n```\n{content}\n```\n")
            except Exception as e:
                context_parts.append(f"### {file_path}\n[Error reading: {e}]\n")
        else:
            context_parts.append(f"### {file_path}\n[File not found]\n")
    return "\n".join(context_parts)


def _select_executor(task: str, context_size: int, executor: str) -> str:
    """Smart routing when executor='auto'."""
    if executor != "auto":
        return executor

    # Check quotas
    jules_available = quota.get_remaining("jules") > 0
    gemini_available = quota.get_remaining("gemini") > 0

    # Routing logic
    # 1. Multi-file indicators → Jules (if available)
    multi_file_keywords = ["refactor", "across files", "multiple files", "module", "service"]
    is_multi_file = any(kw in task.lower() for kw in multi_file_keywords)

    if is_multi_file and jules_available:
        return "jules"

    # 2. Default to Gemini (if available)
    if gemini_available:
        return "gemini"

    # 3. Fallback to QwenAgent (always available)
    return "qwen"


@mcp.tool()
async def delegate_code(
    task: str,
    context_files: list[str],
    output_path: str,
    executor: str = "auto",
    ctx: Context = None
) -> dict:
    """
    Delegate code generation to an executor.

    Args:
        task: What to implement (be specific)
        context_files: MANDATORY list of files the executor must read first.
                      This prevents hallucination by providing real context.
        output_path: Where to write the generated code
        executor: "jules" | "gemini" | "qwen" | "auto" (default: auto-route)

    Returns:
        dict with success, result, executor used, and remaining quota
    """
    if not context_files:
        return {
            "success": False,
            "error": "context_files is MANDATORY. Provide at least one file for the executor to read.",
            "hint": "Example: context_files=['src/module.py', 'tests/test_module.py']"
        }

    # Read context files
    context = _read_context_files(context_files)

    # Build enriched prompt
    enriched_task = f"""# Context Files
{context}

# Task
{task}

# Output Requirements
- Write complete, working code
- Follow patterns from the context files
- Include error handling
- Add type hints (Python) or TypeScript types (JS/TS)
"""

    # Select executor
    selected = _select_executor(task, len(context), executor)

    if ctx:
        ctx.info(f"Routing to {selected} executor")

    # Execute
    try:
        if selected == "jules":
            result = await jules.execute(enriched_task)
        elif selected == "gemini":
            result = await gemini.execute(enriched_task)
        else:  # qwen
            result = await qwen.execute(enriched_task)

        # Record usage
        if result.get("success"):
            quota.record_usage(selected)

            # Write output
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result.get("result", ""))

        return {
            "success": result.get("success", False),
            "result": result.get("result", "")[:500] + "..." if len(result.get("result", "")) > 500 else result.get("result", ""),
            "executor": selected,
            "output_path": output_path,
            "quota_remaining": {
                "jules": quota.get_remaining("jules"),
                "gemini": quota.get_remaining("gemini"),
                "qwen": "unlimited"
            },
            "error": result.get("error")
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "executor": selected
        }


@mcp.tool()
async def delegate_research(
    query: str,
    output_path: str,
    mode: str = "deep",
    download_assets: bool = True,
    ctx: Context = None
) -> dict:
    """
    Delegate research to Perplexity via Chrome extension.

    Args:
        query: Research question (be specific, include context)
        output_path: Where to save the research results
        mode: "quick" (30-60s) | "deep" (2-5min, more sources)
        download_assets: Whether to download generated files (md, py, etc.)

    Returns:
        dict with success, summary, sources, and downloaded files
    """
    if ctx:
        ctx.info(f"Research query: {query[:50]}...")

    try:
        result = await perplexity.execute(
            query=query,
            mode=mode,
            download_assets=download_assets
        )

        if result.get("success"):
            # Save results
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result.get("result", ""))

        return {
            "success": result.get("success", False),
            "summary": result.get("result", "")[:1000] + "..." if len(result.get("result", "")) > 1000 else result.get("result", ""),
            "sources": result.get("sources", [])[:5],
            "downloaded_files": result.get("downloaded_files", []),
            "output_path": output_path,
            "error": result.get("error")
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def delegation_status(ctx: Context = None) -> dict:
    """
    Get current delegation status: quotas, executor health, active tasks.

    Returns:
        dict with executor statuses and quota information
    """
    return {
        "quotas": {
            "jules": {
                "used": quota.get_used("jules"),
                "limit": 15,
                "remaining": quota.get_remaining("jules")
            },
            "gemini": {
                "used": quota.get_used("gemini"),
                "limit": 1500,
                "remaining": quota.get_remaining("gemini")
            },
            "qwen": {
                "status": "unlimited",
                "active": qwen.is_active()
            },
            "perplexity": {
                "status": "ready"
            }
        },
        "routing_priority": [
            "1. Jules (if multi-file + quota available)",
            "2. Gemini CLI (if quota available)",
            "3. QwenAgent (always available, one at a time)"
        ],
        "recommendation": _get_recommendation()
    }


def _get_recommendation() -> str:
    """Generate current routing recommendation."""
    jules_remaining = quota.get_remaining("jules")
    gemini_remaining = quota.get_remaining("gemini")

    if jules_remaining > 10:
        return "Jules available for complex tasks. Use for multi-file refactors."
    elif gemini_remaining > 1000:
        return "Gemini has plenty of quota. Good for most code generation."
    elif gemini_remaining > 100:
        return "Gemini quota getting low. Consider batching similar tasks."
    else:
        return "Low quotas. Routing to QwenAgent for most tasks."


@mcp.tool()
async def run_tests(
    test_command: str,
    working_dir: str = ".",
    timeout: int = 300,
    ctx: Context = None
) -> dict:
    """
    Run tests locally and return structured results.

    Args:
        test_command: The test command to run (e.g., "pytest tests/ -v")
        working_dir: Directory to run tests in
        timeout: Max execution time in seconds

    Returns:
        dict with success, output, failures (if any), and duration
    """
    import asyncio
    import time

    start = time.time()

    try:
        process = await asyncio.create_subprocess_shell(
            test_command,
            cwd=working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=float(timeout)
        )

        duration = time.time() - start
        output = stdout.decode() + stderr.decode()

        # Parse for failures
        failures = []
        for line in output.split("\n"):
            if "FAILED" in line or "ERROR" in line:
                failures.append(line.strip())

        return {
            "success": process.returncode == 0,
            "return_code": process.returncode,
            "output": output[-2000:] if len(output) > 2000 else output,  # Truncate
            "failures": failures[:10],  # Max 10 failures
            "duration_seconds": round(duration, 2),
            "command": test_command
        }

    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"Test timed out after {timeout}s",
            "command": test_command
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "command": test_command
        }


if __name__ == "__main__":
    mcp.run()
