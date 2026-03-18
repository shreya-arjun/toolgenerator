"""
Planner agent: produces a conversation plan from a tool chain and optional corpus context.

Uses MemoryStore (corpus scope) for read path and LLM. Pipeline calls plan() after
sampling a tool chain; when corpus_memory_enabled, agent queries corpus and prepends
to the planning prompt for diversity.
"""

from __future__ import annotations

import json
import os
from typing import Any

from toolgenerator.agents.types import Plan
from toolgenerator.memory.interface import MemoryStore
from toolgenerator.registry.normalizer import Endpoint

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]


def _tool_chain_description(endpoints: list[Endpoint]) -> str:
    """Short description of the tool chain for prompts and corpus query."""
    if not endpoints:
        return "no endpoints"
    parts = [f"{e.endpoint_id} ({e.name})" for e in endpoints]
    return "; ".join(parts)


def _format_corpus_summaries(retrieved: list[dict]) -> str:
    """Format memory search results as text for the prompt."""
    if not retrieved:
        return "(none)"
    return "\n\n".join(
        (item.get("content") or "").strip()
        for item in retrieved
        if isinstance(item, dict)
    )


def _parse_plan_response(text: str) -> Plan:
    """Parse LLM JSON into Plan; fallback to minimal plan on failure."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        obj = json.loads(text)
        user_goal = str(obj.get("user_goal") or "User wants to use the tools.").strip()
        steps = obj.get("steps")
        if not isinstance(steps, list):
            steps = []
        steps = [str(s).strip() for s in steps if s]
        clarification_points = obj.get("clarification_points")
        if not isinstance(clarification_points, list):
            clarification_points = []
        clarification_points = [str(c).strip() for c in clarification_points if c]
        return Plan(user_goal=user_goal, steps=steps, clarification_points=clarification_points)
    except (json.JSONDecodeError, TypeError):
        return Plan(
            user_goal="User wants to accomplish a task using the given API endpoints.",
            steps=[],
            clarification_points=[],
        )


class PlannerAgent:
    """
    Plans a conversation given a tool chain and optional corpus memory.

    Corpus read: when corpus_memory_enabled, calls memory.search(scope="corpus")
    and prepends retrieved summaries to the planning prompt.
    """

    def __init__(
        self,
        memory: MemoryStore,
        *,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str | None = None,
        seed: int = 42,
    ) -> None:
        self._memory = memory
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._seed = seed

    def plan(
        self,
        tool_chain: list[Endpoint],
        corpus_memory_enabled: bool = True,
    ) -> Plan:
        """
        Produce a plan (user_goal, steps, clarification_points) for this tool chain.

        When corpus_memory_enabled, queries corpus and prepends to prompt as:
        [Prior conversations in corpus] {summaries} Given the above, plan a new
        diverse conversation using: {tool_chain}
        """
        if not tool_chain:
            return Plan(
                user_goal="User wants to use the tools.",
                steps=[],
                clarification_points=[],
            )

        tool_chain_str = _tool_chain_description(tool_chain)
        corpus_block = ""
        if corpus_memory_enabled and self._llm_api_key:
            retrieved = self._memory.search(
                query=tool_chain_str,
                scope="corpus",
                top_k=5,
            )
            summaries = _format_corpus_summaries(retrieved)
            corpus_block = (
                "[Prior conversations in corpus]\n"
                f"{summaries}\n\n"
                "Given the above, plan a new diverse conversation using the following tool chain.\n\n"
            )

        prompt = (
            f"{corpus_block}"
            f"Tool chain: {tool_chain_str}\n\n"
            "Output a JSON object with exactly these keys:\n"
            '- "user_goal": one sentence describing what the user wants to achieve.\n'
            '- "steps": list of short step descriptions (what will happen in order).\n'
            '- "clarification_points": list of moments where the assistant should ask the user '
            "a clarifying question (e.g. missing required input, ambiguous intent).\n"
            "Return only valid JSON, no markdown or explanation."
        )

        if OpenAI is None or not self._llm_api_key:
            return Plan(
                user_goal=f"User wants to use: {tool_chain_str}",
                steps=[f"Call {e.name}" for e in tool_chain],
                clarification_points=[],
            )

        try:
            client = OpenAI(api_key=self._llm_api_key)
            resp = client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                seed=self._seed,
            )
            text = (resp.choices[0].message.content or "").strip()
            return _parse_plan_response(text)
        except Exception:
            return Plan(
                user_goal=f"User wants to use: {tool_chain_str}",
                steps=[f"Call {e.name}" for e in tool_chain],
                clarification_points=[],
            )
