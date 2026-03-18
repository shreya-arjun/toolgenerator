"""
Sampler agent: proposes tool chains by traversing the graph (no hardcoded lists).

Uses only graph/sampler. Pipeline passes a ToolGraphSampler (graph + registry + seed);
this agent exposes a single entry point that returns list[Endpoint] for the chosen pattern.
"""

from __future__ import annotations

from toolgenerator.graph.sampler import ToolGraphSampler
from toolgenerator.registry.normalizer import Endpoint


class SamplerAgent:
    """
    Proposes which endpoints to use for a conversation by calling the graph sampler.

    Satisfies spec: generator must use the graph sampler; no hardcoded endpoint lists.
    """

    def __init__(self, graph_sampler: ToolGraphSampler) -> None:
        self._sampler = graph_sampler

    def propose_tool_chain(
        self,
        pattern: str,
        *,
        length: int = 3,
        count: int = 2,
    ) -> list[Endpoint]:
        """
        Propose a list of endpoints for this conversation.

        pattern: "multi_step" -> ordered chain (length >= 3 for spec).
                 "parallel"   -> unordered set from one tool (count >= 2).
        length: used for multi_step (default 3 for ≥3 tool calls).
        count: used for parallel (default 2 for ≥2 endpoints).
        Returns list[Endpoint]; may be empty if graph/sampler returns nothing.
        """
        pattern = (pattern or "multi_step").strip().lower()
        if pattern == "parallel":
            return self._sampler.sample_parallel(count=max(1, count))
        return self._sampler.sample_multi_step_chain(length=max(1, length))
