#!/usr/bin/env python3
"""
End-to-end smoke test for run_pipeline.

Uses real registry (data/toolenv/tools) and built graph, FakeMemoryStore,
mock_mode="template". Mocks planner, user_proxy, and assistant so the test
passes without an OpenAI API key.

To speed up: pre-build the graph with
  toolgenerator build --tools-path data/toolenv/tools
so artifacts/tool_graph.gpickle exists.
"""

import os
from pathlib import Path
from unittest.mock import patch

# Ensure no OpenAI key is used
os.environ.pop("OPENAI_API_KEY", None)

from toolgenerator.agents import AssistantTurnResult, Plan, ValidatorAgent
from toolgenerator.dataset.jsonl_io import read_jsonl
from toolgenerator.generator.pipeline import run_pipeline
from toolgenerator.graph import build_tool_graph, read_tool_graph, write_tool_graph
from toolgenerator.graph.sampler import ToolGraphSampler
from toolgenerator.memory import FakeMemoryStore
from toolgenerator.registry import ToolRegistry

OUTPUT_PATH = Path("artifacts/test_pipeline_output.jsonl")
TOOLS_PATH = Path("data/toolenv/tools")
GRAPH_PATH = Path("artifacts/tool_graph.gpickle")


def _default_arg_for_type(t: str) -> object:
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


def mock_next_turn(
    plan: Plan,
    tool_chain: list,
    current_endpoint_index: int,
    messages_so_far: list,
    session_state,
    conversation_id: str,
) -> AssistantTurnResult:
    """Return one clarification then tool_calls with schema-based default args."""
    has_clarification = any(
        m.get("role") == "assistant" and (m.get("content") or "").strip() and not m.get("tool_call")
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
    if current_endpoint_index >= len(tool_chain):
        return AssistantTurnResult(
            type="clarification",
            content="I've completed the requested actions.",
            endpoint_id=None,
            arguments=None,
            had_retrieved_memory=False,
        )
    endpoint = tool_chain[current_endpoint_index]
    arguments = {}
    for p in endpoint.required_parameters:
        v = p.default
        if v is None:
            v = _default_arg_for_type(p.type)
        arguments[p.name] = v
    for p in endpoint.optional_parameters:
        v = p.default
        if v is None:
            v = _default_arg_for_type(p.type)
        arguments[p.name] = v
    had_retrieved = current_endpoint_index > 0
    return AssistantTurnResult(
        type="tool_call",
        content=None,
        endpoint_id=endpoint.endpoint_id,
        arguments=arguments,
        had_retrieved_memory=had_retrieved,
    )


def mock_plan(self, tool_chain, corpus_memory_enabled=True):
    return Plan(
        user_goal="User wants to look up product or API information.",
        steps=[f"Call {e.name}" for e in tool_chain],
        clarification_points=["Which item to look up?"],
    )


def main() -> None:
    # 1. Load registry from data/toolenv/tools
    registry = ToolRegistry.from_toolbench_path(TOOLS_PATH)
    print(f"Registry: {len(registry)} tools, {len(registry.list_endpoints())} endpoints")

    # 2. Build graph and ToolGraphSampler with seed=42
    if GRAPH_PATH.exists():
        G = read_tool_graph(GRAPH_PATH)
    else:
        G = build_tool_graph(registry)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_tool_graph(G, GRAPH_PATH)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    graph_sampler = ToolGraphSampler(G, registry, seed=42)

    # 3. FakeMemoryStore
    memory = FakeMemoryStore()

    # 4. Run pipeline with mocks (no OpenAI key)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    with (
        patch("toolgenerator.generator.pipeline.PlannerAgent") as MockPlanner,
        patch("toolgenerator.generator.pipeline.UserProxyAgent") as MockUserProxy,
        patch("toolgenerator.generator.pipeline.AssistantAgent") as MockAssistant,
    ):
        MockPlanner.return_value.plan.side_effect = lambda tc, corpus_memory_enabled=True: Plan(
            user_goal="User wants to look up product or API information.",
            steps=[f"Call {e.name}" for e in tc],
            clarification_points=["Which item to look up?"],
        )
        MockUserProxy.return_value.generate_initial_request.return_value = "I want to look up some product info."
        MockUserProxy.return_value.generate_follow_up.return_value = "The product ID is B08BX7N9SK."
        MockAssistant.return_value.next_turn.side_effect = mock_next_turn

        written = run_pipeline(
            seed=42,
            output_path=OUTPUT_PATH,
            corpus_memory_enabled=True,
            num_conversations=3,
            registry=registry,
            graph_sampler=graph_sampler,
            memory=memory,
            pattern_type="multi_step",
            mock_mode="template",
        )

    # 5. Print number of records written
    print(f"\nRecords written: {written}")

    # 6. Read back and print per-record info
    records = read_jsonl(OUTPUT_PATH)
    validator = ValidatorAgent()
    required_metadata_keys = {
        "seed", "tool_ids_used", "num_turns", "num_clarification_questions",
        "memory_grounding_rate", "corpus_memory_enabled", "pattern_type", "conversation_id",
    }
    for i, rec in enumerate(records):
        meta = rec.metadata
        val = validator.validate(rec.model_dump())
        print(f"\n--- Record {i + 1} ---")
        print(f"  conversation_id: {meta.conversation_id}")
        print(f"  num_turns: {meta.num_turns}")
        print(f"  num_clarification_questions: {meta.num_clarification_questions}")
        print(f"  memory_grounding_rate: {meta.memory_grounding_rate}")
        print(f"  tool_ids_used: {meta.tool_ids_used}")
        print(f"  pattern_type: {meta.pattern_type}")
        print(f"  len(tool_calls): {len(rec.tool_calls)}")
        print(f"  validator: {'valid' if val.valid else 'errors: ' + str(val.errors)}")

        # 8. Assert required metadata fields
        meta_dict = rec.metadata.model_dump()
        for key in required_metadata_keys:
            assert key in meta_dict, f"Missing metadata field: {key}"

    # 7. Assert at least 1 record written
    assert written >= 1, f"Expected at least 1 record written, got {written}"
    assert len(records) >= 1, f"Expected at least 1 record in JSONL, got {len(records)}"

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
