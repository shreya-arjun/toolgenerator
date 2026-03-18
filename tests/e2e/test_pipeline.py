from __future__ import annotations

import os
from pathlib import Path

import pytest

from toolgenerator.agents import AssistantTurnResult, Plan, ValidatorAgent
from toolgenerator.dataset.jsonl_io import read_jsonl
from toolgenerator.generator.metrics import unique_tool_chain_ratio
from toolgenerator.generator.pipeline import run_pipeline
from toolgenerator.graph import build_tool_graph
from toolgenerator.graph.sampler import ToolGraphSampler
from toolgenerator.memory import FakeMemoryStore
from toolgenerator.registry import ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_PATH = PROJECT_ROOT / "data" / "toolenv" / "tools"


def _default_arg_for_type(t: str):
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return {}
    return ""


@pytest.mark.slow
def test_pipeline_generates_50_valid_records(monkeypatch, tmp_path: Path):
    os.environ.pop("OPENAI_API_KEY", None)

    class MockPlanner:
        def __init__(self, *args, **kwargs):
            pass

        def plan(self, tool_chain, corpus_memory_enabled=True):
            return Plan(
                user_goal="User wants to look up product or API information.",
                steps=[f"Call {e.name}" for e in tool_chain],
                clarification_points=["Which item to look up?"],
            )

    class MockUserProxy:
        def __init__(self, *args, **kwargs):
            pass

        def generate_initial_request(self, plan):
            return "I want to look up some product info."

        def generate_follow_up(self, plan, last_assistant_message, conversation_so_far=None):
            return "The product ID is B08BX7N9SK."

    class MockAssistant:
        def __init__(self, *args, **kwargs):
            pass

        def next_turn(
            self,
            plan,
            tool_chain,
            current_endpoint_index,
            messages_so_far,
            session_state,
            conversation_id,
        ):
            has_clarification = any(
                m.get("role") == "assistant"
                and (m.get("content") or "").strip()
                and not m.get("tool_call")
                for m in messages_so_far
            )
            if current_endpoint_index == 0 and not has_clarification:
                return AssistantTurnResult(
                    type="clarification",
                    content="Which product or item would you like to look up?",
                    endpoint_id=None,
                    arguments=None,
                    had_retrieved_memory=False,
                )

            endpoint = tool_chain[current_endpoint_index]
            arguments = {}
            for p in endpoint.required_parameters:
                arguments[p.name] = p.default if p.default is not None else _default_arg_for_type(p.type)
            for p in endpoint.optional_parameters:
                arguments[p.name] = p.default if p.default is not None else _default_arg_for_type(p.type)
            return AssistantTurnResult(
                type="tool_call",
                content=None,
                endpoint_id=endpoint.endpoint_id,
                arguments=arguments,
                had_retrieved_memory=current_endpoint_index > 0,
            )

    monkeypatch.setattr("toolgenerator.generator.pipeline.PlannerAgent", MockPlanner)
    monkeypatch.setattr("toolgenerator.generator.pipeline.UserProxyAgent", MockUserProxy)
    monkeypatch.setattr("toolgenerator.generator.pipeline.AssistantAgent", MockAssistant)

    registry = ToolRegistry.from_toolbench_path(TOOLS_PATH)
    graph = build_tool_graph(registry)
    graph_sampler = ToolGraphSampler(graph, registry, seed=42)
    memory = FakeMemoryStore()
    output_path = tmp_path / "pipeline_output.jsonl"

    written = run_pipeline(
        seed=42,
        output_path=output_path,
        corpus_memory_enabled=True,
        num_conversations=50,
        registry=registry,
        graph_sampler=graph_sampler,
        memory=memory,
        pattern_type="multi_step",
        mock_mode="template",
    )

    assert written >= 50

    records = read_jsonl(output_path)
    assert len(records) >= 50

    validator = ValidatorAgent()
    required_metadata_keys = {
        "seed",
        "tool_ids_used",
        "num_turns",
        "num_clarification_questions",
        "memory_grounding_rate",
        "corpus_memory_enabled",
        "pattern_type",
        "conversation_id",
    }
    for record in records:
        metadata = record.metadata.model_dump()
        assert required_metadata_keys.issubset(metadata.keys())
        assert validator.validate(record.model_dump()).valid is True

    assert unique_tool_chain_ratio([r.model_dump() for r in records]) > 0
