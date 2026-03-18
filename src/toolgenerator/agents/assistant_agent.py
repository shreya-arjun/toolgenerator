"""
Assistant agent: produces clarification questions or tool calls with filled arguments.

Uses MemoryStore (session scope) for read path, Executor (for validation/run by pipeline),
and LLM. Clarification is schema-based: required parameters must be fillable from
context + session memory; otherwise the agent asks a clarifying question.
Grounding is tracked explicitly: had_retrieved_memory is True when session memory
was used for a non-first tool call (pipeline uses this for memory_grounding_rate).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from toolgenerator.agents.types import Plan
from toolgenerator.execution.session_state import SessionState
from toolgenerator.memory.interface import MemoryStore
from toolgenerator.registry.normalizer import Endpoint

from toolgenerator.execution.executor import Executor

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]


@dataclass
class AssistantTurnResult:
    """Result of one assistant turn: either clarification or tool call, plus grounding flag."""

    type: str  # "clarification" | "tool_call"
    content: str | None  # clarification text when type == "clarification"
    endpoint_id: str | None  # set when type == "tool_call"
    arguments: dict[str, Any] | None  # set when type == "tool_call"
    had_retrieved_memory: bool  # True if session memory was used to fill args (non-first call)


def _format_session_context(session_state: SessionState) -> str:
    """Format prior tool outputs for the prompt."""
    entries = session_state.get_all()
    if not entries:
        return "(no prior tool outputs)"
    return "\n\n".join(
        f"Endpoint {e['endpoint_id']}: {e['output']}"
        for e in entries
    )


def _format_memory_context(retrieved: list[dict]) -> str:
    """Format memory.search results for [Memory context] block."""
    if not retrieved:
        return ""
    return "\n\n".join(
        (item.get("content") or "").strip()
        for item in retrieved
        if isinstance(item, dict)
    )


def _params_schema_text(endpoint: Endpoint) -> str:
    """Describe required and optional parameters for the prompt."""
    lines = []
    for p in endpoint.required_parameters:
        lines.append(f"  - {p.name} ({p.type}, required): {p.description or ''}")
    for p in endpoint.optional_parameters:
        lines.append(f"  - {p.name} ({p.type}, optional): {p.description or ''}")
    return "\n".join(lines) if lines else "  (none)"


class AssistantAgent:
    """
    Produces assistant turns: clarification (when required params can't be filled)
    or tool call with arguments. Session memory is read before filling args for
    non-first tool calls; had_retrieved_memory is set when retrieved entries were used.
    """

    def __init__(
        self,
        memory: MemoryStore,
        executor: Executor,
        *,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str | None = None,
        seed: int = 42,
    ) -> None:
        self._memory = memory
        self._executor = executor
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._seed = seed

    def next_turn(
        self,
        plan: Plan,
        tool_chain: list[Endpoint],
        current_endpoint_index: int,
        messages_so_far: list[dict[str, Any]],
        session_state: SessionState,
        conversation_id: str,
    ) -> AssistantTurnResult:
        """
        Decide next assistant action: clarification or tool call for the current endpoint.

        Clarification is chosen when required parameters cannot be filled from
        context (session_state + messages). For non-first calls, session memory
        is searched and injected; had_retrieved_memory is True if any entry was used.
        """
        if current_endpoint_index >= len(tool_chain):
            return AssistantTurnResult(
                type="clarification",
                content="I've completed the requested actions.",
                endpoint_id=None,
                arguments=None,
                had_retrieved_memory=False,
            )

        endpoint = tool_chain[current_endpoint_index]
        is_first_tool_call = current_endpoint_index == 0

        # Session memory read for non-first tool calls
        retrieved: list[dict] = []
        if not is_first_tool_call:
            retrieved = self._memory.search(
                query=endpoint.name or endpoint.endpoint_id,
                scope="session",
                top_k=5,
                conversation_id=conversation_id,
            )
        had_retrieved_memory = len(retrieved) >= 1

        # Build context from session state and recent messages
        session_context = _format_session_context(session_state)
        recent = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}"
            for m in messages_so_far[-10:]
        )

        memory_block = ""
        if retrieved:
            memory_block = (
                "[Memory context]\n"
                f"{_format_memory_context(retrieved)}\n\n"
                "Given the above context and the current tool schema, fill in the arguments.\n\n"
            )

        params_text = _params_schema_text(endpoint)
        prompt = (
            f"{memory_block}"
            f"Endpoint: {endpoint.name} ({endpoint.endpoint_id}). Description: {endpoint.description or 'N/A'}.\n\n"
            f"Parameters:\n{params_text}\n\n"
            f"Context (prior tool outputs and conversation):\n{session_context}\n\n"
            f"Recent messages:\n{recent}\n\n"
            "If you can fill all required parameters from the context above, respond with a single line: JSON: followed by a JSON object with parameter names as keys and values. "
            "If information is missing for any required parameter, respond with a single line: CLARIFY: followed by one short clarifying question the assistant would ask the user. "
            "Use only the keys shown in Parameters. Output nothing else."
        )

        response = self._call_llm(prompt)
        if not response:
            return AssistantTurnResult(
                type="clarification",
                content="Could you provide more details about what you need?",
                endpoint_id=None,
                arguments=None,
                had_retrieved_memory=False,
            )

        response = response.strip()
        if response.upper().startswith("CLARIFY:"):
            question = response[8:].strip()
            return AssistantTurnResult(
                type="clarification",
                content=question or "Could you provide the missing information?",
                endpoint_id=None,
                arguments=None,
                had_retrieved_memory=False,
            )

        if response.upper().startswith("JSON:"):
            json_str = response[5:].strip()
            try:
                arguments = json.loads(json_str)
                if isinstance(arguments, dict):
                    return AssistantTurnResult(
                        type="tool_call",
                        content=None,
                        endpoint_id=endpoint.endpoint_id,
                        arguments=arguments,
                        had_retrieved_memory=had_retrieved_memory,
                    )
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse the whole response as JSON
        try:
            arguments = json.loads(response)
            if isinstance(arguments, dict):
                return AssistantTurnResult(
                    type="tool_call",
                    content=None,
                    endpoint_id=endpoint.endpoint_id,
                    arguments=arguments,
                    had_retrieved_memory=had_retrieved_memory,
                )
        except json.JSONDecodeError:
            pass

        return AssistantTurnResult(
            type="clarification",
            content="Could you provide more details so I can complete this request?",
            endpoint_id=None,
            arguments=None,
            had_retrieved_memory=False,
        )

    def _call_llm(self, prompt: str) -> str:
        """Single LLM call with temperature=0 and seed."""
        if OpenAI is None or not self._llm_api_key:
            return ""
        try:
            client = OpenAI(api_key=self._llm_api_key)
            resp = client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                seed=self._seed,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""
