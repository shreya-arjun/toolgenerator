"""Normalise raw ToolBench tool JSON into typed Tool / Endpoint / Parameter structures."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Normalised model (JSON-serializable via asdict)
# ---------------------------------------------------------------------------


@dataclass
class Parameter:
    """Normalised parameter schema for an endpoint."""

    name: str
    type: str  # canonical lowercase: string, integer, number, boolean, array, object
    description: str
    default: Any = None  # str | int | float | bool | None; missing -> None


@dataclass
class Endpoint:
    """Normalised API endpoint (one callable unit)."""

    endpoint_id: str
    name: str
    url: str
    description: str
    method: str
    required_parameters: list[Parameter]
    optional_parameters: list[Parameter]
    response_schema: dict[str, Any] | None = None


@dataclass
class Tool:
    """Normalised tool: metadata + list of endpoints."""

    tool_id: str
    tool_name: str
    tool_description: str
    title: str
    category: str
    endpoints: list[Endpoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Type normalization
# ---------------------------------------------------------------------------

_TYPE_ALIASES = {
    "string": "string",
    "str": "string",
    "STRING": "string",
    "integer": "integer",
    "int": "integer",
    "INTEGER": "integer",
    "number": "number",
    "float": "number",
    "NUMBER": "number",
    "boolean": "boolean",
    "bool": "boolean",
    "BOOLEAN": "boolean",
    "array": "array",
    "ARRAY": "array",
    "object": "object",
    "obj": "object",
    "OBJECT": "object",
}


def _normalize_param_type(raw: Any) -> str:
    """Canonicalise parameter type to lowercase; unknown -> 'string'."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return "string"
    key = str(raw).strip()
    return _TYPE_ALIASES.get(key, "string")


def _slug(s: str) -> str:
    """Lowercase, replace non-alnum/space with nothing, spaces to underscores."""
    s = (s or "").strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s).strip("_").lower()
    return s or "unknown"


# ---------------------------------------------------------------------------
# Normalise one raw tool dict
# ---------------------------------------------------------------------------


def _normalize_parameter(raw: dict[str, Any]) -> Parameter:
    """Build a Parameter from a raw required_parameters / optional_parameters item."""
    name = (raw.get("name") or "").strip() or "param"
    raw_type = raw.get("type")
    param_type = _normalize_param_type(raw_type)
    description = (raw.get("description") or "").strip()
    default = raw.get("default")  # allow missing key -> None
    return Parameter(name=name, type=param_type, description=description, default=default)


def _normalize_endpoint(api: dict[str, Any], tool_id: str) -> Endpoint:
    """Build an Endpoint from one api_list item."""
    name = (api.get("name") or "").strip() or "endpoint"
    # Globally unique endpoint id
    endpoint_id = f"{tool_id}::{name}"
    url = (api.get("url") or "").strip() if api.get("url") is not None else ""
    description = (api.get("description") or "").strip()
    method = (api.get("method") or "GET").strip().upper() or "GET"

    raw_req = api.get("required_parameters")
    raw_opt = api.get("optional_parameters")
    if not isinstance(raw_req, list):
        raw_req = []
    if not isinstance(raw_opt, list):
        raw_opt = []

    required_parameters = [_normalize_parameter(p) for p in raw_req if isinstance(p, dict)]
    optional_parameters = [_normalize_parameter(p) for p in raw_opt if isinstance(p, dict)]

    schema = api.get("schema")
    if isinstance(schema, dict) and schema:
        response_schema = schema
    else:
        response_schema = None

    return Endpoint(
        endpoint_id=endpoint_id,
        name=name,
        url=url,
        description=description,
        method=method,
        required_parameters=required_parameters,
        optional_parameters=optional_parameters,
        response_schema=response_schema,
    )


def normalize_tool(raw: dict[str, Any], category: str) -> Tool | None:
    """
    Normalise a single raw ToolBench tool dict into a Tool.

    raw must contain at least tool name and api_list. Uses category (from path)
    to build tool_id. Returns None if api_list is missing or empty.
    """
    tool_name = (raw.get("tool_name") or raw.get("name") or "").strip()
    if not tool_name:
        return None

    tool_id = f"{category}__{_slug(tool_name)}"
    tool_description = (raw.get("tool_description") or "").strip()
    title = (raw.get("title") or tool_name).strip()

    api_list = raw.get("api_list")
    if not isinstance(api_list, list) or len(api_list) == 0:
        return None

    endpoints: list[Endpoint] = []
    for api in api_list:
        if not isinstance(api, dict):
            continue
        ep = _normalize_endpoint(api, tool_id)
        endpoints.append(ep)

    if not endpoints:
        return None

    return Tool(
        tool_id=tool_id,
        tool_name=tool_name,
        tool_description=tool_description,
        title=title,
        category=category,
        endpoints=endpoints,
    )


def tool_to_dict(tool: Tool) -> dict[str, Any]:
    """Serialize Tool to a JSON-serializable dict (for artifacts/registry.json)."""
    out = {
        "tool_id": tool.tool_id,
        "tool_name": tool.tool_name,
        "tool_description": tool.tool_description,
        "title": tool.title,
        "category": tool.category,
        "endpoints": [],
    }
    for ep in tool.endpoints:
        ep_dict = {
            "endpoint_id": ep.endpoint_id,
            "name": ep.name,
            "url": ep.url,
            "description": ep.description,
            "method": ep.method,
            "required_parameters": [asdict(p) for p in ep.required_parameters],
            "optional_parameters": [asdict(p) for p in ep.optional_parameters],
            "response_schema": ep.response_schema,
        }
        out["endpoints"].append(ep_dict)
    return out
