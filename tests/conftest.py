from __future__ import annotations

from pathlib import Path

import pytest

from toolgenerator.graph import build_tool_graph
from toolgenerator.memory import FakeMemoryStore
from toolgenerator.registry import ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_registry() -> ToolRegistry:
    return ToolRegistry.from_toolbench_path(PROJECT_ROOT / "data" / "sample")


@pytest.fixture
def sample_graph(sample_registry):
    return build_tool_graph(sample_registry)


@pytest.fixture
def fake_memory():
    return FakeMemoryStore()


@pytest.fixture
def sample_conversation_record() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "Find me product info"},
            {"role": "assistant", "content": "Which product would you like?", "tool_call": None, "tool_output": None},
            {"role": "user", "content": "Use product A"},
            {"role": "assistant", "content": "", "tool_call": {"endpoint_id": "Tools__alpha::search", "arguments": {"query": "A"}}, "tool_output": None},
            {"role": "tool", "content": "", "tool_call": None, "tool_output": {"id": "a1"}},
            {"role": "assistant", "content": "", "tool_call": {"endpoint_id": "Data__beta::details", "arguments": {"id": "a1"}}, "tool_output": None},
            {"role": "tool", "content": "", "tool_call": None, "tool_output": {"status": "ok"}},
            {"role": "assistant", "content": "", "tool_call": {"endpoint_id": "Tools__alpha::reviews", "arguments": {"id": "a1"}}, "tool_output": None},
            {"role": "tool", "content": "", "tool_call": None, "tool_output": {"rating": 5}},
        ],
        "tool_calls": [
            {"endpoint_id": "Tools__alpha::search", "arguments": {"query": "A"}},
            {"endpoint_id": "Data__beta::details", "arguments": {"id": "a1"}},
            {"endpoint_id": "Tools__alpha::reviews", "arguments": {"id": "a1"}},
        ],
        "tool_outputs": [
            {"endpoint_id": "Tools__alpha::search", "output": {"id": "a1"}, "success": True},
            {"endpoint_id": "Data__beta::details", "output": {"status": "ok"}, "success": True},
            {"endpoint_id": "Tools__alpha::reviews", "output": {"rating": 5}, "success": True},
        ],
        "metadata": {
            "seed": 42,
            "tool_ids_used": ["Tools__alpha", "Data__beta"],
            "num_turns": 9,
            "num_clarification_questions": 1,
            "memory_grounding_rate": 1.0,
            "corpus_memory_enabled": True,
            "pattern_type": "multi_step",
            "conversation_id": "conv-001",
        },
    }
