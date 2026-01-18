"""
Quota Manager - Track daily usage for rate-limited executors.
"""

from datetime import date
from pathlib import Path
import json


class QuotaManager:
    """Track and enforce daily quotas for executors."""

    LIMITS = {
        "jules": 15,
        "gemini": 1500,
        "qwen": float("inf"),  # Unlimited
        "perplexity": float("inf"),  # Unlimited (research only)
    }

    def __init__(self, state_file: Path = None):
        self.state_file = state_file or Path.home() / ".claude" / "delegation_quota.json"
        self._load_state()

    def _load_state(self):
        """Load quota state from disk."""
        self.usage = {}
        self.last_reset = date.today().isoformat()

        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.last_reset = data.get("last_reset", self.last_reset)
                self.usage = data.get("usage", {})

                # Reset if new day
                if self.last_reset != date.today().isoformat():
                    self._reset()
            except Exception:
                self._reset()

    def _reset(self):
        """Reset daily quotas."""
        self.usage = {k: 0 for k in self.LIMITS}
        self.last_reset = date.today().isoformat()
        self._save_state()

    def _save_state(self):
        """Persist quota state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({
            "last_reset": self.last_reset,
            "usage": self.usage
        }, indent=2))

    def record_usage(self, executor: str, count: int = 1):
        """Record usage for an executor."""
        # Reset if new day
        if self.last_reset != date.today().isoformat():
            self._reset()

        self.usage[executor] = self.usage.get(executor, 0) + count
        self._save_state()

    def get_used(self, executor: str) -> int:
        """Get usage count for today."""
        if self.last_reset != date.today().isoformat():
            self._reset()
        return self.usage.get(executor, 0)

    def get_remaining(self, executor: str) -> int:
        """Get remaining quota for today."""
        limit = self.LIMITS.get(executor, 0)
        if limit == float("inf"):
            return float("inf")
        return max(0, int(limit - self.get_used(executor)))

    def is_available(self, executor: str) -> bool:
        """Check if executor has quota remaining."""
        return self.get_remaining(executor) > 0
