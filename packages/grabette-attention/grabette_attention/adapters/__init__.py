"""Policy adapters: every model-specific fact lives behind this boundary."""

from .base import PolicyAdapter, RunResult

__all__ = ["PolicyAdapter", "RunResult"]
