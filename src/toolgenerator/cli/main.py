"""Typer entry point for the toolgenerator CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from toolgenerator.cli.build import run_build
from toolgenerator.cli.generate import run_generate
from toolgenerator.cli.metrics import run_metrics
from toolgenerator.cli.validate import run_validate

app = typer.Typer(help="Offline multi-agent tool-use conversation generator.")


@app.command()
def build(
    tools_path: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output_dir: Path = typer.Option(Path("artifacts")),
) -> None:
    """Build registry and graph artifacts."""
    run_build(tools_path=tools_path, output_dir=output_dir)


@app.command()
def generate(
    output: Path = typer.Option(...),
    num: int = typer.Option(..., min=1),
    seed: int = typer.Option(...),
    no_corpus_memory: bool = typer.Option(False, "--no-corpus-memory"),
    mock_mode: str = typer.Option("template"),
    tools_path: Path | None = typer.Option(None),
    graph_path: Path = typer.Option(Path("artifacts/tool_graph.gpickle")),
    llm_model: str = typer.Option("gpt-4o-mini"),
    pattern: str = typer.Option("multi_step"),
) -> None:
    """Generate conversations into a JSONL dataset."""
    try:
        run_generate(
            output=output,
            num=num,
            seed=seed,
            corpus_memory_enabled=not no_corpus_memory,
            mock_mode=mock_mode,
            tools_path=tools_path,
            graph_path=graph_path,
            llm_model=llm_model,
            pattern=pattern,
        )
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc


@app.command()
def validate(
    input: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Validate a generated JSONL dataset."""
    run_validate(input)


@app.command()
def metrics(
    input: Path = typer.Option(..., exists=True, dir_okay=False),
    compare: Path | None = typer.Option(None),
) -> None:
    """Compute diversity metrics for one dataset or compare two datasets."""
    run_metrics(input, compare)


if __name__ == "__main__":
    app()
