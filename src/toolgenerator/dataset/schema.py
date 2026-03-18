"""
Pydantic v2 schema for one output record (conversation + tool calls + metadata).

Exact shape required by the assessment: generate, validate, and metrics all depend on this.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Message: one role-tagged turn in the conversation
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in the conversation (user, assistant, or tool)."""

    role: Literal["user", "assistant", "tool"] = Field(
        ...,
        description="Speaker of this turn.",
    )
    content: str = Field(
        default="",
        description="Text content of the message.",
    )
    tool_call: dict[str, Any] | None = Field(
        default=None,
        description="Present on assistant turns that invoke a tool; endpoint_id and arguments.",
    )
    tool_output: dict[str, Any] | None = Field(
        default=None,
        description="Present on tool turns; the mock output for the preceding tool call.",
    )


# ---------------------------------------------------------------------------
# Tool call / tool output entries (parallel lists or flattened)
# ---------------------------------------------------------------------------


class ToolCallEntry(BaseModel):
    """One tool call: endpoint and arguments."""

    endpoint_id: str = Field(..., description="Registry endpoint id (e.g. tool_id::endpoint_name).")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the endpoint.")


class ToolOutputEntry(BaseModel):
    """One tool output: endpoint, result, and success flag."""

    endpoint_id: str = Field(..., description="Endpoint that produced this output.")
    output: dict[str, Any] = Field(default_factory=dict, description="Mock response body.")
    success: bool = Field(..., description="Whether the call succeeded (validation passed).")


# ---------------------------------------------------------------------------
# Metadata attached to each record
# ---------------------------------------------------------------------------


class RecordMetadata(BaseModel):
    """Metadata for reproducibility and metrics."""

    seed: int = Field(..., description="Random seed used for this conversation.")
    tool_ids_used: list[str] = Field(
        default_factory=list,
        description="Tool ids (e.g. category__tool_name) used in this conversation.",
    )
    num_turns: int = Field(..., description="Number of conversation turns (messages).")
    num_clarification_questions: int = Field(
        default=0,
        description="Number of assistant clarification questions before tool use.",
    )
    memory_grounding_rate: float | None = Field(
        default=None,
        description="(non-first-step tool calls with memory context) / (total non-first-step tool calls).",
    )
    corpus_memory_enabled: bool = Field(
        default=False,
        description="Whether corpus memory was used when generating this conversation.",
    )
    pattern_type: str = Field(
        default="multi_step",
        description="Sampling pattern used, e.g. 'multi_step' or 'parallel'.",
    )
    conversation_id: str = Field(
        ...,
        description="Unique id for this conversation (e.g. UUID).",
    )


# ---------------------------------------------------------------------------
# Top-level record: one line in the JSONL dataset
# ---------------------------------------------------------------------------


class ConversationRecord(BaseModel):
    """
    One generated conversation record: messages, tool calls, tool outputs, metadata.

    Written as one JSON object per line in the output JSONL file.
    """

    messages: list[Message] = Field(
        default_factory=list,
        description="Conversation turns in order (role-tagged; tool_call on assistant, tool_output on tool).",
    )
    tool_calls: list[ToolCallEntry] = Field(
        default_factory=list,
        description="All tool calls in order (endpoint_id + arguments).",
    )
    tool_outputs: list[ToolOutputEntry] = Field(
        default_factory=list,
        description="All tool outputs in order (endpoint_id, output, success).",
    )
    metadata: RecordMetadata = Field(
        ...,
        description="Seed, tool_ids_used, num_turns, clarification count, memory stats, pattern_type, conversation_id.",
    )
