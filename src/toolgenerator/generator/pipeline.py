"""
Pipeline: orchestrate agents to generate conversations and write JSONL.

Sets random.seed(seed) at start; passes seed to sampler, executor, and agents.
Session memory write after every tool call; corpus read before planning, corpus write
after each conversation. append_jsonl after each conversation (crash-safe).
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

from toolgenerator.agents import (
    AssistantAgent,
    AssistantTurnResult,
    Plan,
    PlannerAgent,
    SamplerAgent,
    UserProxyAgent,
    ValidatorAgent,
)
from toolgenerator.dataset.jsonl_io import append_jsonl
from toolgenerator.execution import Executor, SessionState
from toolgenerator.generator.conversation import ConversationBuilder
from toolgenerator.generator.metrics import compute_memory_grounding_rate
from toolgenerator.graph.sampler import ToolGraphSampler
from toolgenerator.memory.interface import MemoryStore
from toolgenerator.registry import ToolRegistry
from toolgenerator.registry.normalizer import Endpoint


def _tool_id_from_endpoint_id(endpoint_id: str) -> str:
    if "::" in endpoint_id:
        return endpoint_id.split("::", 1)[0]
    return endpoint_id


def _tool_chain_description(endpoints: list[Endpoint]) -> str:
    return "; ".join(f"{e.endpoint_id}" for e in endpoints)


def _conversation_summary(plan: Plan, tool_ids_used: list[str], pattern_type: str) -> str:
    """Short summary for corpus memory write."""
    return f"Goal: {plan.user_goal}. Tools: {tool_ids_used}. Pattern: {pattern_type}."


def run_pipeline(
    seed: int,
    output_path: Path | str,
    corpus_memory_enabled: bool,
    num_conversations: int,
    registry: ToolRegistry,
    graph_sampler: ToolGraphSampler,
    memory: MemoryStore,
    *,
    pattern_type: str = "multi_step",
    llm_model: str = "gpt-4o-mini",
    llm_api_key: str | None = None,
    mock_mode: str = "template",
) -> int:
    """
    Generate num_conversations and append each to output_path. Returns count written.

    Sets random.seed(seed); passes seed to sampler, executor, and all agents.
    Session memory write after every tool call; corpus read before planning when
    corpus_memory_enabled; corpus write after each conversation when enabled.
    """
    random.seed(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    executor = Executor(mock_mode=mock_mode, llm_model=llm_model, llm_api_key=llm_api_key, seed=seed)
    planner = PlannerAgent(memory=memory, llm_model=llm_model, llm_api_key=llm_api_key, seed=seed)
    user_proxy = UserProxyAgent(llm_model=llm_model, llm_api_key=llm_api_key, seed=seed)
    assistant = AssistantAgent(
        memory=memory,
        executor=executor,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        seed=seed,
    )
    validator = ValidatorAgent()
    sampler_agent = SamplerAgent(graph_sampler)

    written = 0
    for _ in range(num_conversations):
        conversation_id = str(uuid.uuid4())
        tool_chain = sampler_agent.propose_tool_chain(pattern_type, length=3, count=2)
        if not tool_chain:
            continue

        plan = planner.plan(tool_chain, corpus_memory_enabled=corpus_memory_enabled)
        session_state = SessionState()
        builder = ConversationBuilder()

        # Initial user message
        user_text = user_proxy.generate_initial_request(plan)
        builder.add_user_message(user_text)

        messages_so_far: list[dict[str, Any]] = [
            {"role": "user", "content": user_text},
        ]
        grounded_count = 0
        non_first_total = 0
        current_endpoint_index = 0
        step_index = 0

        while current_endpoint_index < len(tool_chain):
            result: AssistantTurnResult = assistant.next_turn(
                plan=plan,
                tool_chain=tool_chain,
                current_endpoint_index=current_endpoint_index,
                messages_so_far=messages_so_far,
                session_state=session_state,
                conversation_id=conversation_id,
            )

            if result.type == "clarification":
                builder.add_assistant_message(result.content or "", tool_call=None)
                messages_so_far.append({"role": "assistant", "content": result.content or ""})
                follow_up = user_proxy.generate_follow_up(plan, result.content or "", messages_so_far)
                builder.add_user_message(follow_up)
                messages_so_far.append({"role": "user", "content": follow_up})
                continue

            if result.type != "tool_call" or result.endpoint_id is None or result.arguments is None:
                continue

            endpoint = registry.get_endpoint(result.endpoint_id)
            if not endpoint:
                continue

            run_result = executor.run(endpoint, result.arguments, session_state)
            step_index += 1

            builder.add_tool_call(result.endpoint_id, result.arguments)
            builder.add_assistant_message(
                content="",
                tool_call={"endpoint_id": result.endpoint_id, "arguments": result.arguments},
            )
            builder.add_tool_message(
                content=json.dumps(run_result.get("output") or {}),
                tool_output=run_result.get("output"),
            )
            builder.add_tool_output(
                result.endpoint_id,
                run_result.get("output") or {},
                run_result.get("success", False),
            )

            messages_so_far.append({
                "role": "assistant",
                "content": "",
                "tool_call": {"endpoint_id": result.endpoint_id, "arguments": result.arguments},
            })
            messages_so_far.append({
                "role": "tool",
                "content": json.dumps(run_result.get("output") or {}),
            })

            # Session memory write (after every tool call)
            memory.add(
                content=json.dumps(run_result.get("output") or {}),
                scope="session",
                metadata={
                    "conversation_id": conversation_id,
                    "step": step_index,
                    "endpoint": result.endpoint_id,
                },
            )

            is_first = current_endpoint_index == 0
            if not is_first:
                non_first_total += 1
                if result.had_retrieved_memory:
                    grounded_count += 1
            current_endpoint_index += 1

        memory_grounding_rate = compute_memory_grounding_rate(grounded_count, non_first_total)
        tool_ids_used = list({_tool_id_from_endpoint_id(ep.endpoint_id) for ep in tool_chain})

        record = builder.build(
            seed=seed,
            tool_ids_used=tool_ids_used,
            corpus_memory_enabled=corpus_memory_enabled,
            pattern_type=pattern_type,
            conversation_id=conversation_id,
            memory_grounding_rate=memory_grounding_rate,
        )
        record_dict = record.model_dump()
        validation = validator.validate(record_dict)
        if not validation.valid:
            continue
        append_jsonl(record, output_path)
        written += 1

        # Corpus memory write (after conversation complete)
        if corpus_memory_enabled:
            summary = _conversation_summary(plan, tool_ids_used, pattern_type)
            memory.add(
                content=summary,
                scope="corpus",
                metadata={
                    "conversation_id": conversation_id,
                    "tools": tool_ids_used,
                    "pattern_type": pattern_type,
                },
            )

    return written
