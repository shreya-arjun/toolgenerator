"""
User-proxy agent: generates user-side messages (initial request and follow-ups).

Uses LLM only. Pipeline calls generate_initial_request(plan) for the first user
message and generate_follow_up(plan, last_assistant_message, ...) after clarification.
"""

from __future__ import annotations

import os
from typing import Any

from toolgenerator.agents.types import Plan

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]


def _format_turn(t: dict[str, Any]) -> str:
    """Format one message for context in the prompt."""
    role = t.get("role", "")
    content = (t.get("content") or "").strip()
    return f"{role}: {content}"


class UserProxyAgent:
    """
    Generates user messages from the plan and (for follow-ups) the last assistant message.

    All LLM calls use temperature=0 and seed for reproducibility.
    """

    def __init__(
        self,
        *,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str | None = None,
        seed: int = 42,
    ) -> None:
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._seed = seed

    def generate_initial_request(self, plan: Plan) -> str:
        """
        Generate the first user message from the plan (user_goal and steps).

        Returns a short natural-language user request that aligns with the plan.
        """
        prompt = (
            f"Generate a single short user message (1-2 sentences) that a user would say to request help.\n"
            f"User goal: {plan.user_goal}\n"
            f"Steps (for context): {chr(10).join(plan.steps) if plan.steps else 'N/A'}\n"
            "Output only the user message text, no quotes or explanation."
        )
        return self._call_llm(prompt) or f"I'd like to {plan.user_goal}."

    def generate_follow_up(
        self,
        plan: Plan,
        last_assistant_message: str,
        conversation_so_far: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Generate a user reply after the assistant asked a clarifying question or requested info.

        last_assistant_message: the assistant's question or request.
        conversation_so_far: optional list of {role, content} for context.
        """
        context = ""
        if conversation_so_far:
            context = "Recent conversation:\n" + "\n".join(
                _format_turn(t) for t in conversation_so_far[-6:]
            ) + "\n\n"
        prompt = (
            f"{context}"
            f"Assistant just said: {last_assistant_message}\n\n"
            f"User goal (for consistency): {plan.user_goal}\n\n"
            "Generate a single short user reply that answers the assistant or provides the requested information. "
            "Output only the user message text, no quotes or explanation."
        )
        return self._call_llm(prompt) or "Here's the information you asked for."

    def _call_llm(self, prompt: str) -> str:
        """Single LLM call with temperature=0 and seed. Returns stripped content or empty string on failure."""
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
