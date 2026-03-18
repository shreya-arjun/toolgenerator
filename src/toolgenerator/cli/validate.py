"""Validate command: schema-safe JSONL read and rule validation."""

from __future__ import annotations

from pathlib import Path

import typer

from toolgenerator.agents import ValidatorAgent
from toolgenerator.dataset.jsonl_io import read_jsonl


def run_validate(input_path: Path) -> None:
    """Validate each record in a JSONL dataset and print a summary."""
    records = read_jsonl(input_path)
    validator = ValidatorAgent()

    valid_count = 0
    for idx, record in enumerate(records, start=1):
        result = validator.validate(record.model_dump())
        if result.valid:
            valid_count += 1
            typer.echo(
                f"[{idx}] valid conversation_id={record.metadata.conversation_id}"
            )
        else:
            typer.echo(
                f"[{idx}] invalid conversation_id={record.metadata.conversation_id} errors={result.errors}"
            )

    typer.echo(f"Summary: {valid_count}/{len(records)} valid")
