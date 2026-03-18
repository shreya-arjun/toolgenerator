from __future__ import annotations

from copy import deepcopy

from toolgenerator.agents import ValidatorAgent


def test_valid_conversation_passes_all_checks(sample_conversation_record):
    result = ValidatorAgent().validate(sample_conversation_record)
    assert result.valid is True
    assert result.errors == []


def test_conversation_with_two_tool_calls_fails(sample_conversation_record):
    record = deepcopy(sample_conversation_record)
    record["tool_calls"] = record["tool_calls"][:2]
    record["tool_outputs"] = record["tool_outputs"][:2]
    result = ValidatorAgent().validate(record)
    assert result.valid is False
    assert any("len(tool_calls) >= 3" in e for e in result.errors)


def test_conversation_with_one_distinct_tool_fails(sample_conversation_record):
    record = deepcopy(sample_conversation_record)
    for tc in record["tool_calls"]:
        tc["endpoint_id"] = "Tools__alpha::same"
    for to in record["tool_outputs"]:
        to["endpoint_id"] = "Tools__alpha::same"
    result = ValidatorAgent().validate(record)
    assert result.valid is False
    assert any("at least 2 distinct tools" in e for e in result.errors)


def test_conversation_with_no_clarification_turn_fails(sample_conversation_record):
    record = deepcopy(sample_conversation_record)
    record["messages"] = [m for m in record["messages"] if not (m["role"] == "assistant" and m["tool_call"] is None and m["content"])]
    result = ValidatorAgent().validate(record)
    assert result.valid is False
    assert any("clarification turn" in e for e in result.errors)


def test_tool_call_count_not_equal_tool_output_count_fails(sample_conversation_record):
    record = deepcopy(sample_conversation_record)
    record["tool_outputs"] = record["tool_outputs"][:2]
    result = ValidatorAgent().validate(record)
    assert result.valid is False
    assert any("tool_calls length" in e for e in result.errors)


def test_missing_memory_grounding_rate_fails(sample_conversation_record):
    record = deepcopy(sample_conversation_record)
    del record["metadata"]["memory_grounding_rate"]
    result = ValidatorAgent().validate(record)
    assert result.valid is False
    assert any("memory_grounding_rate" in e for e in result.errors)
