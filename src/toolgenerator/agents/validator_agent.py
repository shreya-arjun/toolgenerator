"""
Validator agent: checks a conversation record against spec rules.

Uses dataset/schema for structure. Returns errors (does not raise).
All five required checks: tool_calls count, distinct tools, clarification turn,
tool_call/tool_output pairing, memory_grounding_rate in metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    """Result of validating one conversation record."""

    valid: bool
    errors: list[str]


def _tool_id_from_endpoint_id(endpoint_id: str) -> str:
    """Extract tool_id from endpoint_id (format: tool_id::endpoint_name)."""
    if "::" in endpoint_id:
        return endpoint_id.split("::", 1)[0]
    return endpoint_id


def validate_conversation(record: dict[str, Any]) -> ValidationResult:
    """
    Validate a conversation record against all spec rules. Does not raise.

    Checks:
    1. len(tool_calls) >= 3
    2. len(distinct tools from endpoint_ids) >= 2
    3. at least one clarification turn exists (assistant message with content, no tool_call)
    4. every tool_call has a matching tool_output (same order, same endpoint_id)
    5. memory_grounding_rate is present in metadata (key exists; value may be None)
    """
    errors: list[str] = []

    tool_calls = record.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []
    tool_outputs = record.get("tool_outputs")
    if not isinstance(tool_outputs, list):
        tool_outputs = []
    messages = record.get("messages")
    if not isinstance(messages, list):
        messages = []
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    # 1. len(tool_calls) >= 3
    if len(tool_calls) < 3:
        errors.append(f"len(tool_calls) >= 3 required, got {len(tool_calls)}")

    # 2. len(distinct tools from endpoint_ids) >= 2
    endpoint_ids = []
    for tc in tool_calls:
        if isinstance(tc, dict) and "endpoint_id" in tc:
            endpoint_ids.append(tc["endpoint_id"])
    tool_ids = {_tool_id_from_endpoint_id(eid) for eid in endpoint_ids}
    if len(tool_ids) < 2:
        errors.append(f"at least 2 distinct tools required, got {len(tool_ids)} (from {len(endpoint_ids)} endpoint_ids)")

    # 3. at least one clarification turn exists
    has_clarification = False
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        content = (m.get("content") or "").strip()
        tool_call = m.get("tool_call")
        if content and not tool_call:
            has_clarification = True
            break
    if not has_clarification:
        errors.append("at least one clarification turn required (assistant message with content and no tool_call)")

    # 4. every tool_call has a matching tool_output
    if len(tool_calls) != len(tool_outputs):
        errors.append(f"tool_calls length ({len(tool_calls)}) must equal tool_outputs length ({len(tool_outputs)})")
    else:
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            eid_tc = tc.get("endpoint_id")
            if i >= len(tool_outputs):
                break
            to = tool_outputs[i]
            if not isinstance(to, dict):
                errors.append(f"tool_outputs[{i}] is not a dict")
                continue
            eid_to = to.get("endpoint_id")
            if eid_tc != eid_to:
                errors.append(f"tool_call[{i}] endpoint_id {eid_tc!r} does not match tool_outputs[{i}] endpoint_id {eid_to!r}")

    # 5. memory_grounding_rate is present in metadata
    if "memory_grounding_rate" not in metadata:
        errors.append("metadata must contain 'memory_grounding_rate' (value may be None)")

    return ValidationResult(valid=len(errors) == 0, errors=errors)


class ValidatorAgent:
    """
    Validates conversation records against spec. Returns ValidationResult; never raises.
    """

    def validate(self, record: dict[str, Any]) -> ValidationResult:
        """Run all five checks; return result with valid flag and list of error messages."""
        return validate_conversation(record)
