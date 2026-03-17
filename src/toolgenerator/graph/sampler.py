"""
Tool graph samplers: multi-step chains and parallel endpoint sets.

All sampling is done by traversing the NetworkX graph (no hardcoded endpoint lists).
The generator must use these samplers during data generation.
"""

from __future__ import annotations

import random

import networkx as nx

from toolgenerator.graph.model import (
    NODE_ATTR_TYPE,
    NODE_TYPE_CONCEPT,
    NODE_TYPE_ENDPOINT,
    NODE_TYPE_TOOL,
    PREFIX_ENDPOINT,
    strip_prefix,
)
from toolgenerator.registry import ToolRegistry
from toolgenerator.registry.normalizer import Endpoint


def _endpoint_nodes(G: nx.DiGraph) -> list[str]:
    """Return all node ids that are endpoints (by attribute)."""
    return [
        n for n in G.nodes
        if G.nodes[n].get(NODE_ATTR_TYPE) == NODE_TYPE_ENDPOINT
    ]


def _tool_nodes(G: nx.DiGraph) -> list[str]:
    """Return all node ids that are tools."""
    return [
        n for n in G.nodes
        if G.nodes[n].get(NODE_ATTR_TYPE) == NODE_TYPE_TOOL
    ]


def _endpoint_to_registry_id(node_id: str) -> str:
    """Strip graph prefix to get registry endpoint_id."""
    return strip_prefix(node_id) if node_id.startswith(PREFIX_ENDPOINT) else node_id


def _get_tool_for_endpoint(G: nx.DiGraph, ep_node: str) -> str | None:
    """Return the tool node that has an edge to this endpoint (predecessor)."""
    preds = list(G.predecessors(ep_node))
    for p in preds:
        if G.nodes[p].get(NODE_ATTR_TYPE) == NODE_TYPE_TOOL:
            return p
    return None


def _endpoints_of_tool(G: nx.DiGraph, tool_node: str) -> list[str]:
    """Return endpoint node ids that are successors of this tool."""
    return [
        n for n in G.successors(tool_node)
        if G.nodes[n].get(NODE_ATTR_TYPE) == NODE_TYPE_ENDPOINT
    ]


def _tools_under_concept(G: nx.DiGraph, concept_node: str) -> list[str]:
    """Return tool node ids that point to this concept (tools under this concept)."""
    return [
        n for n in G.predecessors(concept_node)
        if G.nodes[n].get(NODE_ATTR_TYPE) == NODE_TYPE_TOOL
    ]


def sample_tool_chain_multi_step(
    G: nx.DiGraph,
    registry: ToolRegistry,
    length: int,
    rng: random.Random,
) -> list[Endpoint]:
    """
    Sample an ordered chain of Endpoints by traversing the graph (multi-step pattern).

    Traversal: pick a random endpoint -> get its tool -> then either pick another
    endpoint of the same tool or (via a concept) another tool and one of its endpoints.
    Repeats until length endpoints. No hardcoded lists; all choices come from graph
    structure.
    """
    endpoint_nodes = _endpoint_nodes(G)
    if not endpoint_nodes or length < 1:
        return []

    chain: list[str] = []  # graph node ids
    current_ep = rng.choice(endpoint_nodes)
    chain.append(current_ep)

    while len(chain) < length:
        tool_node = _get_tool_for_endpoint(G, current_ep)
        if not tool_node:
            break
        same_tool_eps = _endpoints_of_tool(G, tool_node)
        # Option A: another endpoint from same tool (exclude current)
        candidates = [n for n in same_tool_eps if n != current_ep]
        # Option B: go to a concept, then to another tool, then to an endpoint
        for succ in G.successors(tool_node):
            if G.nodes[succ].get(NODE_ATTR_TYPE) == NODE_TYPE_CONCEPT:
                other_tools = _tools_under_concept(G, succ)
                for ot in other_tools:
                    if ot != tool_node:
                        candidates.extend(_endpoints_of_tool(G, ot))
        chain_reg_ids = {strip_prefix(n) for n in chain}
        if not candidates:
            break
        current_ep = rng.choice(candidates)
        ep_id = _endpoint_to_registry_id(current_ep)
        if ep_id not in chain_reg_ids:  # avoid trivial duplicates in chain
            chain.append(current_ep)
        else:
            # pick different one if possible
            others = [c for c in candidates if strip_prefix(c) not in chain_reg_ids]
            if not others:
                break
            current_ep = rng.choice(others)
            chain.append(current_ep)

    endpoints = [registry.get_endpoint(strip_prefix(node_id)) for node_id in chain]
    return [e for e in endpoints if e is not None]


def sample_parallel_endpoints(
    G: nx.DiGraph,
    registry: ToolRegistry,
    count: int,
    rng: random.Random,
) -> list[Endpoint]:
    """
    Sample a set of Endpoints that can be called in parallel, by traversing the graph.

    Traversal: pick a random tool, then take up to `count` of its endpoints (siblings).
    All choices come from graph structure.
    """
    tool_nodes = _tool_nodes(G)
    if not tool_nodes or count < 1:
        return []

    tool_node = rng.choice(tool_nodes)
    ep_nodes = _endpoints_of_tool(G, tool_node)
    if not ep_nodes:
        return []

    how_many = min(count, len(ep_nodes))
    chosen = rng.sample(ep_nodes, how_many)
    endpoints = [registry.get_endpoint(strip_prefix(node_id)) for node_id in chosen]
    return [e for e in endpoints if e is not None]


class ToolGraphSampler:
    """
    Convenience wrapper: holds graph and registry, exposes multi-step and parallel
    sampling with a shared RNG (for reproducibility when seed is fixed).
    """

    def __init__(
        self,
        G: nx.DiGraph,
        registry: ToolRegistry,
        *,
        seed: int | None = None,
    ) -> None:
        self._G = G
        self._registry = registry
        self._rng = random.Random(seed)

    def sample_multi_step_chain(self, length: int = 3) -> list[Endpoint]:
        """Ordered list of Endpoints (multi-step chain)."""
        return sample_tool_chain_multi_step(
            self._G, self._registry, length, self._rng
        )

    def sample_parallel(self, count: int = 2) -> list[Endpoint]:
        """Unordered list of Endpoints (parallel call set)."""
        return sample_parallel_endpoints(
            self._G, self._registry, count, self._rng
        )
