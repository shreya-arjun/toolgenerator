"""Dataset schema and JSONL I/O for generated conversations."""

from toolgenerator.dataset.jsonl_io import append_jsonl, read_jsonl, write_jsonl
from toolgenerator.dataset.schema import (
    ConversationRecord,
    Message,
    RecordMetadata,
    ToolCallEntry,
    ToolOutputEntry,
)

__all__ = [
    "ConversationRecord",
    "Message",
    "RecordMetadata",
    "ToolCallEntry",
    "ToolOutputEntry",
    "append_jsonl",
    "read_jsonl",
    "write_jsonl",
]
