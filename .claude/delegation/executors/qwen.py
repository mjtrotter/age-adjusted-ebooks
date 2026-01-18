"""
QwenAgent Executor - Local MLX-based code generation.

Quota: Unlimited (local compute)
Constraint: One instance at a time (memory limits)
Best for: Fallback, data processing, simple code generation
"""

import asyncio
from typing import Optional
from pathlib import Path


class QwenExecutor:
    """Execute tasks via QwenAgent with MLX backend."""

    # Model options (ordered by speed)
    MODELS = {
        "fast": "OpenCoder-8B",        # 86 tok/s
        "balanced": "Phi-4-reasoning",  # 51 tok/s (recommended)
        "quality": "QwQ-32B",           # 25 tok/s
    }

    def __init__(self, model: str = "balanced"):
        self.name = "qwen"
        self.model = self.MODELS.get(model, model)
        self._active = False
        self._available: Optional[bool] = None

    def is_active(self) -> bool:
        """Check if an instance is currently running."""
        return self._active

    async def check_available(self) -> bool:
        """Check if QwenAgent and MLX are available."""
        if self._available is not None:
            return self._available

        try:
            # Check for mlx and qwen_agent
            import importlib.util
            mlx_spec = importlib.util.find_spec("mlx")
            qwen_spec = importlib.util.find_spec("qwen_agent")

            self._available = mlx_spec is not None and qwen_spec is not None
        except Exception:
            self._available = False

        return self._available

    async def execute(
        self,
        task: str,
        mode: str = "balanced",
        timeout: int = 300
    ) -> dict:
        """
        Execute task via QwenAgent with MLX.

        Args:
            task: The prompt including context and instructions
            mode: "fast", "balanced", or "quality"
            timeout: Execution timeout in seconds

        Returns:
            dict with success, result, error
        """
        if self._active:
            return {
                "success": False,
                "error": "QwenAgent is already running. Wait for current task to complete."
            }

        if not await self.check_available():
            return {
                "success": False,
                "error": "QwenAgent/MLX not available. Install: pip install qwen-agent mlx mlx-lm"
            }

        self._active = True

        try:
            # Import here to avoid loading MLX if not needed
            from qwen_agent.agents import Assistant
            from qwen_agent.llm.schema import Message

            # Use subprocess to isolate memory
            result = await self._run_in_subprocess(task, mode, timeout)
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            self._active = False
            # Clear GPU cache
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except Exception:
                pass

    async def _run_in_subprocess(self, task: str, mode: str, timeout: int) -> dict:
        """Run QwenAgent in a subprocess for memory isolation."""
        import json
        import tempfile

        # Write task to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(task)
            task_file = f.name

        model = self.MODELS.get(mode, self.model)

        # Python script to run QwenAgent
        script = f'''
import json
from pathlib import Path

try:
    from qwen_agent.agents import Assistant

    # Read task
    task = Path("{task_file}").read_text()

    # Create assistant with MLX backend
    assistant = Assistant(
        llm={{"model": "{model}", "model_server": "mlx"}},
        name="CodeAssistant"
    )

    # Run task
    messages = [{{"role": "user", "content": task}}]
    response = ""
    for chunk in assistant.run(messages):
        if chunk and isinstance(chunk, list) and len(chunk) > 0:
            msg = chunk[-1]
            if hasattr(msg, "content"):
                response = msg.content

    print(json.dumps({{"success": True, "result": response}}))

except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
'''

        try:
            process = await asyncio.create_subprocess_exec(
                "python3", "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=float(timeout)
            )

            # Parse result
            output = stdout.decode().strip()
            if output:
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    return {"success": True, "result": output}
            else:
                return {
                    "success": False,
                    "error": stderr.decode() or "No output from QwenAgent"
                }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"QwenAgent timed out after {timeout}s"
            }
        finally:
            # Cleanup temp file
            try:
                Path(task_file).unlink()
            except Exception:
                pass
