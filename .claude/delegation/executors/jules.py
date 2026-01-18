"""
Jules Executor - Google's agentic coding assistant.

Quota: 15 tasks/day (free tier)
Best for: Multi-file refactors, complex implementations
"""

import asyncio
import shutil
import re
from pathlib import Path
from typing import Optional


class JulesExecutor:
    """Execute tasks via Google Jules CLI."""

    def __init__(self, work_dir: Path = None):
        self.name = "jules"
        self.work_dir = work_dir or Path.cwd()
        self._cli_available: Optional[bool] = None

    async def check_available(self) -> bool:
        """Check if Jules CLI is installed."""
        if self._cli_available is not None:
            return self._cli_available

        cli_path = shutil.which("jules")
        if not cli_path:
            self._cli_available = False
            return False

        try:
            process = await asyncio.create_subprocess_exec(
                "jules", "--version",
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
        timeout: int = 600,  # Jules can take longer for complex tasks
        poll_interval: int = 30
    ) -> dict:
        """
        Execute task via Jules.

        Jules runs asynchronously - we submit the task and poll for results.

        Args:
            task: The prompt including context and instructions
            timeout: Maximum wait time for completion
            poll_interval: How often to check for results

        Returns:
            dict with success, result, session_id, error
        """
        if not await self.check_available():
            return {
                "success": False,
                "error": "Jules CLI not installed. Run: npm install -g @google/jules && jules login"
            }

        try:
            # Create new Jules session
            create_process = await asyncio.create_subprocess_exec(
                "jules", "new", task,
                cwd=str(self.work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                create_process.communicate(),
                timeout=60.0
            )

            if create_process.returncode != 0:
                return {
                    "success": False,
                    "error": stderr.decode().strip() or "Failed to create Jules session"
                }

            # Extract session ID from output
            output = stdout.decode()
            session_match = re.search(r'session[:\s]+([a-zA-Z0-9-]+)', output, re.IGNORECASE)
            session_id = session_match.group(1) if session_match else None

            if not session_id:
                # If no session ID, Jules may have completed synchronously
                return {
                    "success": True,
                    "result": output,
                    "session_id": None
                }

            # Poll for completion
            elapsed = 0
            while elapsed < timeout:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                # Check session status
                status_process = await asyncio.create_subprocess_exec(
                    "jules", "remote", "list", "--session",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                status_out, _ = await status_process.communicate()
                status_text = status_out.decode()

                # Check if our session is complete
                if session_id in status_text and "complete" in status_text.lower():
                    # Pull results
                    pull_process = await asyncio.create_subprocess_exec(
                        "jules", "remote", "pull", "--session", session_id,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )

                    pull_out, pull_err = await pull_process.communicate()

                    return {
                        "success": pull_process.returncode == 0,
                        "result": pull_out.decode(),
                        "session_id": session_id,
                        "error": pull_err.decode() if pull_process.returncode != 0 else None
                    }

            # Timeout reached
            return {
                "success": False,
                "error": f"Jules task timed out after {timeout}s. Session: {session_id}",
                "session_id": session_id
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Jules session creation timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
