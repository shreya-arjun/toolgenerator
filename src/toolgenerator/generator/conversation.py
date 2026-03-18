"""
ConversationBuilder: in-memory conversation state and the only place that constructs ConversationRecord.

Only imports from dataset/schema. Tracks messages, tool_calls, tool_outputs, and
num_clarification_questions. Pipeline passes grounded_count/non_first_total to
metrics.compute_memory_grounding_rate() and passes the result into build().
"""

from __future__ import annotations

from typing import Any

from toolgenerator.dataset.schema import (
    ConversationRecord,
    Message,
    RecordMetadata,
    ToolCallEntry,
    ToolOutputEntry,
)


class ConversationBuilder:
    """
    Builds a conversation in memory; build() produces the final ConversationRecord
    with all metadata fields populated. The only place that constructs ConversationRecord.
    """

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._tool_outputs: list[dict[str, Any]] = []
        self._num_clarification_questions: int = 0

    def add_user_message(self, content: str) -> None:
        """Append a user turn."""
        self._messages.append({"role": "user", "content": content or ""})

    def add_assistant_message(
        self,
        content: str,
        tool_call: dict[str, Any] | None = None,
    ) -> None:
        """Append an assistant turn. If content is non-empty and tool_call is None, counts as clarification."""
        self._messages.append({
            "role": "assistant",
            "content": content or "",
            "tool_call": tool_call,
            "tool_output": None,
        })
        if (content or "").strip() and tool_call is None:
            self._num_clarification_questions += 1

    def add_tool_message(self, content: str, tool_output: dict[str, Any] | None = None) -> None:
        """Append a tool turn (output of a tool call)."""
        self._messages.append({
            "role": "tool",
            "content": content or "",
            "tool_call": None,
            "tool_output": tool_output,
        })

    def add_tool_call(self, endpoint_id: str, arguments: dict[str, Any]) -> None:
        """Record a tool call (endpoint_id + arguments)."""
        self._tool_calls.append({"endpoint_id": endpoint_id, "arguments": dict(arguments)})

    def add_tool_output(self, endpoint_id: str, output: dict[str, Any], success: bool) -> None:
        """Record a tool output (endpoint_id, output, success)."""
        self._tool_outputs.append({
            "endpoint_id": endpoint_id,
            "output": dict(output),
            "success": success,
        })

    @property
    def num_clarification_questions(self) -> int:
        """Number of assistant clarification turns added so far."""
        return self._num_clarification_questions

    def build(
        self,
        seed: int,
        tool_ids_used: list[str],
        corpus_memory_enabled: bool,
        pattern_type: str,
        conversation_id: str,
        memory_grounding_rate: float | None,
    ) -> ConversationRecord:
        """
        Produce the final ConversationRecord with all metadata fields populated.

        memory_grounding_rate is computed by the pipeline via metrics.compute_memory_grounding_rate
        (grounded_count, non_first_total) and passed in.
        """
        num_turns = len(self._messages)
        metadata = RecordMetadata(
            seed=seed,
            tool_ids_used=list(tool_ids_used),
            num_turns=num_turns,
            num_clarification_questions=self._num_clarification_questions,
            memory_grounding_rate=memory_grounding_rate,
            corpus_memory_enabled=corpus_memory_enabled,
            pattern_type=pattern_type,
            conversation_id=conversation_id,
        )
        messages = [
            Message(
                role=m["role"],
                content=m.get("content") or "",
                tool_call=m.get("tool_call"),
                tool_output=m.get("tool_output"),
            )
            for m in self._messages
        ]
        tool_calls = [
            ToolCallEntry(endpoint_id=tc["endpoint_id"], arguments=tc.get("arguments") or {})
            for tc in self._tool_calls
        ]
        tool_outputs = [
            ToolOutputEntry(
                endpoint_id=to["endpoint_id"],
                output=to.get("output") or {},
                success=to.get("success", False),
            )
            for to in self._tool_outputs
        ]
        return ConversationRecord(
            messages=messages,
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
            metadata=metadata,
        )
