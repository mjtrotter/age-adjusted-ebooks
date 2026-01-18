"""
Gemini CLI Executor - Primary code generation via Google's Gemini CLI.

Quota: 1500 requests/day (free tier)
Models: gemini-3-flash-preview (fast), gemini-3-pro-preview (quality)
"""

import asyncio
import shutil
from typing import Optional


class GeminiExecutor:
    """Execute code generation via Gemini CLI."""

    MODEL_MAP = {
        "flash": "gemini-3-flash-preview",
        "pro": "gemini-3-pro-preview",
    }

    def __init__(self):
        self.name = "gemini"
        self._cli_available: Optional[bool] = None

    async def check_available(self) -> bool:
        """Check if gemini CLI is installed."""
        if self._cli_available is not None:
            return self._cli_available

        cli_path = shutil.which("gemini")
        if not cli_path:
            self._cli_available = False
            return False

        try:
            process = await asyncio.create_subprocess_exec(
                "gemini", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(process.communicate(), timeout=10.0)
            self._cli_available = process.returncode == 0
        except Exception:
            self._cli_available = False

        return self._cli_available

    async def execute(
        self,
        task: str,
        mode: str = "flash",
        timeout: int = 120
    ) -> dict:
        """
        Execute task via Gemini CLI.

        Args:
            task: The prompt including context and instructions
            mode: "flash" (fast) or "pro" (quality)
            timeout: Execution timeout in seconds

        Returns:
            dict with success, result, error
        """
        if not await self.check_available():
            return {
                "success": False,
                "error": "Gemini CLI not installed. Run: pip install google-generativeai && gemini configure"
            }

        model_id = self.MODEL_MAP.get(mode, mode)

        try:
            process = await asyncio.create_subprocess_exec(
                "gemini",
                "--model", model_id,
                task,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=float(timeout)
            )

            if process.returncode == 0:
                return {
                    "success": True,
                    "result": stdout.decode().strip(),
                    "model": model_id
                }
            else:
                return {
                    "success": False,
                    "error": stderr.decode().strip() or "Unknown error",
                    "model": model_id
                }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Execution timed out after {timeout}s"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
