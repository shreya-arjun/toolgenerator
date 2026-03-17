"""
Lightweight key-value store for a single conversation.

Stores outputs from completed tool calls so later steps can reference them
(e.g. IDs, handles, selected items). No dependencies on other toolgenerator modules.
"""

from __future__ import annotations

from typing import Any


class SessionState:
    """
    Per-conversation store of tool call outputs.

    - store(endpoint_id, output): record one tool call result.
    - get(endpoint_id): return the most recent output for that endpoint, or None.
    - get_all(): return all prior outputs in call order, each as {"endpoint_id": ..., "output": ...}.
    """

    def __init__(self) -> None:
        self._by_endpoint: dict[str, Any] = {}  # endpoint_id -> latest output
        self._order: list[tuple[str, Any]] = []  # (endpoint_id, output) in call order

    def store(self, endpoint_id: str, output: Any) -> None:
        """Record the output of a tool call for this endpoint."""
        self._by_endpoint[endpoint_id] = output
        self._order.append((endpoint_id, output))

    def get(self, endpoint_id: str) -> Any | None:
        """Return the most recent output for this endpoint, or None."""
        return self._by_endpoint.get(endpoint_id)

    def get_all(self) -> list[dict[str, Any]]:
        """Return all prior outputs in call order. Each item is {"endpoint_id": str, "output": Any}."""
        return [
            {"endpoint_id": eid, "output": out}
            for eid, out in self._order
        ]
