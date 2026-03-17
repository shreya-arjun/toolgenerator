"""In-memory tool registry: lookup tools and endpoints by id."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolgenerator.registry.loader import load_toolbench_tools
from toolgenerator.registry.normalizer import Endpoint, Tool, normalize_tool, tool_to_dict


class ToolRegistry:
    """
    In-memory registry of normalised tools and endpoints.

    Supports lookup by tool_id and endpoint_id. Endpoint IDs are globally
    unique: "{tool_id}::{endpoint_name}".
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: list[Tool] = list(tools)
        self._by_tool_id: dict[str, Tool] = {t.tool_id: t for t in self._tools}
        self._by_endpoint_id: dict[str, Endpoint] = {}
        for t in self._tools:
            for ep in t.endpoints:
                self._by_endpoint_id[ep.endpoint_id] = ep

    def get_tool(self, tool_id: str) -> Tool | None:
        """Return the tool with the given tool_id, or None."""
        return self._by_tool_id.get(tool_id)

    def get_endpoint(self, endpoint_id: str) -> Endpoint | None:
        """Return the endpoint with the given endpoint_id, or None."""
        return self._by_endpoint_id.get(endpoint_id)

    def list_tools(self) -> list[Tool]:
        """Return all tools in registration order."""
        return list(self._tools)

    def list_endpoints(self, tool_id: str | None = None) -> list[Endpoint]:
        """Return all endpoints; if tool_id is set, only that tool's endpoints."""
        if tool_id is None:
            return list(self._by_endpoint_id.values())
        t = self._by_tool_id.get(tool_id)
        return list(t.endpoints) if t else []

    def __len__(self) -> int:
        return len(self._tools)

    @classmethod
    def from_toolbench_path(cls, tools_root: Path) -> ToolRegistry:
        """
        Load from ToolBench directory and return a ToolRegistry.

        tools_root should be the path to data/toolenv/tools (or data/sample
        for tests). Skips files that don't normalise to a valid tool.
        """
        raw = load_toolbench_tools(Path(tools_root))
        tools: list[Tool] = []
        for r in raw:
            category = r.get("category", "")
            tool = normalize_tool(r, category)
            if tool is not None:
                tools.append(tool)
        return cls(tools)

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry to a JSON-serializable dict (for artifacts/registry.json)."""
        return {
            "tools": [tool_to_dict(t) for t in self._tools],
        }
