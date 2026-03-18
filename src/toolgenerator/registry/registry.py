"""In-memory tool registry: lookup tools and endpoints by id."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolgenerator.registry.loader import load_toolbench_tools
from toolgenerator.registry.normalizer import Endpoint, Parameter, Tool, normalize_tool, tool_to_dict


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolRegistry:
        """Load a ToolRegistry from a dict like the output of to_dict()."""
        raw_tools = data.get("tools", [])
        tools: list[Tool] = []
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                continue
            endpoints: list[Endpoint] = []
            for raw_ep in raw_tool.get("endpoints", []):
                if not isinstance(raw_ep, dict):
                    continue
                required_parameters = [
                    Parameter(**p)
                    for p in raw_ep.get("required_parameters", [])
                    if isinstance(p, dict)
                ]
                optional_parameters = [
                    Parameter(**p)
                    for p in raw_ep.get("optional_parameters", [])
                    if isinstance(p, dict)
                ]
                endpoints.append(
                    Endpoint(
                        endpoint_id=raw_ep.get("endpoint_id", ""),
                        name=raw_ep.get("name", ""),
                        url=raw_ep.get("url", ""),
                        description=raw_ep.get("description", ""),
                        method=raw_ep.get("method", "GET"),
                        required_parameters=required_parameters,
                        optional_parameters=optional_parameters,
                        response_schema=raw_ep.get("response_schema"),
                    )
                )
            tools.append(
                Tool(
                    tool_id=raw_tool.get("tool_id", ""),
                    tool_name=raw_tool.get("tool_name", ""),
                    tool_description=raw_tool.get("tool_description", ""),
                    title=raw_tool.get("title", ""),
                    category=raw_tool.get("category", ""),
                    endpoints=endpoints,
                )
            )
        return cls(tools)

    def save_json(self, path: Path | str) -> None:
        """Write the registry artifact to disk as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> ToolRegistry:
        """Read the registry artifact from disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
