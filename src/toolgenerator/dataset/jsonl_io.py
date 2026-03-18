"""
Read and write conversation records as JSONL.

- write_jsonl: write a full list of records (e.g. tests).
- append_jsonl: append one record (primary path during generation; preserves progress on crash).
- read_jsonl: read and validate records; skip and log malformed lines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from toolgenerator.dataset.schema import ConversationRecord

logger = logging.getLogger(__name__)


def write_jsonl(records: list[ConversationRecord], path: Path | str) -> None:
    """
    Write a complete list of records to a JSONL file (one JSON object per line).

    Overwrites the file if it exists. Use for writing a full dataset at once (e.g. tests).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            line = record.model_dump_json(exclude_none=False)
            f.write(line + "\n")


def append_jsonl(record: ConversationRecord, path: Path | str) -> None:
    """
    Append a single record to a JSONL file (one line).

    Creates the file and parent directories if needed. This is the primary write path
    during generation so that progress is preserved if generation crashes mid-run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = record.model_dump_json(exclude_none=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_jsonl(path: Path | str) -> list[ConversationRecord]:
    """
    Read a JSONL file and return validated ConversationRecords.

    Malformed lines (invalid JSON or schema validation failure) are skipped and
    logged; the function does not raise. Returns only successfully parsed records.
    """
    path = Path(path)
    if not path.is_file():
        return []

    records: list[ConversationRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skip line %d: invalid JSON: %s", line_no, e)
                continue
            try:
                record = ConversationRecord.model_validate(obj)
                records.append(record)
            except ValidationError as e:
                logger.warning("Skip line %d: schema validation failed: %s", line_no, e)
                continue
    return records
