"""Metrics command: diversity metrics for one dataset or a comparison pair."""

from __future__ import annotations

from pathlib import Path

import typer

from toolgenerator.dataset.jsonl_io import read_jsonl
from toolgenerator.generator.metrics import distinct_n_grams, unique_tool_chain_ratio


def _assistant_texts(records: list) -> list[str]:
    texts: list[str] = []
    for record in records:
        for message in record.messages:
            if message.role == "assistant" and (message.content or "").strip():
                texts.append(message.content)
    return texts


def _print_metrics(label: str, records: list) -> None:
    record_dicts = [record.model_dump() for record in records]
    u_ratio = unique_tool_chain_ratio(record_dicts)
    d2 = distinct_n_grams(_assistant_texts(records), n=2)
    typer.echo(f"{label}:")
    typer.echo(f"  unique_tool_chain_ratio={u_ratio:.4f}")
    typer.echo(f"  distinct_2_gram={d2:.4f}")


def run_metrics(input_path: Path, compare_path: Path | None = None) -> None:
    """Compute and print diversity metrics for one or two datasets."""
    records = read_jsonl(input_path)
    _print_metrics("input", records)

    if compare_path is not None:
        compare_records = read_jsonl(compare_path)
        _print_metrics("compare", compare_records)
