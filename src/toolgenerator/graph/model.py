"""
Graph model: node type constants and node ID conventions.

All graph nodes have attribute "node_type" set to one of the NODE_TYPE_* constants.
Node IDs are strings with a prefix so we can filter by type when traversing.
"""

# Node type attribute key (stored on every node)
NODE_ATTR_TYPE = "node_type"

# Node types (value for NODE_ATTR_TYPE)
NODE_TYPE_CONCEPT = "concept"       # Category or keyword tag
NODE_TYPE_TOOL = "tool"
NODE_TYPE_ENDPOINT = "endpoint"
NODE_TYPE_PARAMETER = "parameter"
NODE_TYPE_RESPONSE_FIELD = "response_field"

# ID prefixes (node_id starts with these for easy filtering)
PREFIX_CONCEPT = "concept:"
PREFIX_TOOL = "tool:"
PREFIX_ENDPOINT = "endpoint:"
PREFIX_PARAMETER = "param:"
PREFIX_RESPONSE_FIELD = "response:"


def concept_id(name: str) -> str:
    """Globally unique concept node id (category or keyword)."""
    return f"{PREFIX_CONCEPT}{name}"


def tool_id(tool_id_from_registry: str) -> str:
    """Graph node id for a tool."""
    return f"{PREFIX_TOOL}{tool_id_from_registry}"


def endpoint_id(endpoint_id_from_registry: str) -> str:
    """Graph node id for an endpoint (same as registry endpoint_id with prefix)."""
    return f"{PREFIX_ENDPOINT}{endpoint_id_from_registry}"


def parameter_id(endpoint_id_from_registry: str, param_name: str) -> str:
    """Graph node id for a parameter."""
    return f"{PREFIX_PARAMETER}{endpoint_id_from_registry}::{param_name}"


def response_field_id(endpoint_id_from_registry: str, field_key: str) -> str:
    """Graph node id for a response field (from schema)."""
    return f"{PREFIX_RESPONSE_FIELD}{endpoint_id_from_registry}::{field_key}"


def strip_prefix(node_id: str) -> str:
    """Return the suffix after the first ':' (e.g. endpoint_id from graph node id)."""
    idx = node_id.find(":")
    return node_id[idx + 1:] if idx >= 0 else node_id
