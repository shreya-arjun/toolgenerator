"""
Shared data types for agents. No other toolgenerator dependencies.

Allows planner and user_proxy to share Plan without coupling user_proxy to planner.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Plan:
    """High-level plan for one conversation (user_goal, steps, clarification points)."""

    user_goal: str
    steps: list[str]
    clarification_points: list[str]
