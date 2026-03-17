"""
Build a NetworkX DiGraph from a ToolRegistry.

Nodes: Concept (category + keyword tags), Tool, Endpoint, Parameter, ResponseField.
Edges: Concept <- Tool; Tool -> Endpoint; Endpoint -> Parameter; Endpoint -> ResponseField (when schema present).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import networkx as nx

from toolgenerator.registry import ToolRegistry
from toolgenerator.registry.normalizer import Endpoint, Tool

from toolgenerator.graph.model import (
    NODE_TYPE_CONCEPT,
    NODE_TYPE_ENDPOINT,
    NODE_TYPE_PARAMETER,
    NODE_TYPE_RESPONSE_FIELD,
    NODE_TYPE_TOOL,
    NODE_ATTR_TYPE,
    concept_id,
    endpoint_id,
    parameter_id,
    response_field_id,
    tool_id,
)

# Simple English stopwords for keyword extraction (no external NLP)
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "it", "its", "you", "your", "we", "they",
        "i", "me", "my", "he", "she", "his", "her", "api", "get", "use",
    }
)


def _extract_keywords(text: str, max_keywords: int = 3, min_len: int = 3) -> list[str]:
    """
    Extract up to max_keywords from text: tokenize on non-alpha, lowercase,
    drop stopwords and short tokens, return unique order-preserving list.
    """
    if not (text and text.strip()):
        return []
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if len(t) >= min_len and t not in _STOPWORDS and t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) >= max_keywords:
                break
    return out


def _add_response_field_nodes(
    G: nx.DiGraph,
    ep: Endpoint,
    ep_node: str,
) -> None:
    """Add ResponseField nodes and edges from endpoint when response_schema has properties."""
    schema = ep.response_schema
    if not schema or not isinstance(schema.get("properties"), dict):
        return
    for key in schema.get("properties", {}):
        if not key or not isinstance(key, str):
            continue
        rf_node = response_field_id(ep.endpoint_id, key)
        if not G.has_node(rf_node):
            G.add_node(rf_node, **{NODE_ATTR_TYPE: NODE_TYPE_RESPONSE_FIELD})
        G.add_edge(ep_node, rf_node)


def build_tool_graph(registry: ToolRegistry) -> nx.DiGraph:
    """
    Build a directed graph from the tool registry.

    - Concept nodes: one per category (from tool.category), plus 2-3 keyword
      concepts per tool from tool_description.
    - Tool -> Concept edges for category and keyword concepts.
    - Tool -> Endpoint -> Parameter edges.
    - Endpoint -> ResponseField when response_schema has properties.

    Returns a NetworkX DiGraph. Each node has attribute NODE_ATTR_TYPE.
    """
    G: nx.DiGraph = nx.DiGraph()

    for tool in registry.list_tools():
        t_node = tool_id(tool.tool_id)
        G.add_node(t_node, **{NODE_ATTR_TYPE: NODE_TYPE_TOOL})

        # Category concept (RapidAPI category = directory name)
        cat_node = concept_id(tool.category)
        if not G.has_node(cat_node):
            G.add_node(cat_node, **{NODE_ATTR_TYPE: NODE_TYPE_CONCEPT})
        G.add_edge(t_node, cat_node)

        # Keyword concepts from tool_description
        for kw in _extract_keywords(tool.tool_description, max_keywords=3):
            kw_node = concept_id(kw)
            if not G.has_node(kw_node):
                G.add_node(kw_node, **{NODE_ATTR_TYPE: NODE_TYPE_CONCEPT})
            G.add_edge(t_node, kw_node)

        for ep in tool.endpoints:
            ep_node = endpoint_id(ep.endpoint_id)
            G.add_node(ep_node, **{NODE_ATTR_TYPE: NODE_TYPE_ENDPOINT})
            G.add_edge(t_node, ep_node)

            for p in ep.required_parameters + ep.optional_parameters:
                p_node = parameter_id(ep.endpoint_id, p.name)
                G.add_node(p_node, **{NODE_ATTR_TYPE: NODE_TYPE_PARAMETER})
                G.add_edge(ep_node, p_node)

            _add_response_field_nodes(G, ep, ep_node)

    return G


def write_tool_graph(G: nx.DiGraph, path: Path | str) -> None:
    """Serialize the graph to a pickle file (e.g. artifacts/tool_graph.gpickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)


def read_tool_graph(path: Path | str) -> nx.DiGraph:
    """Load a graph from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
