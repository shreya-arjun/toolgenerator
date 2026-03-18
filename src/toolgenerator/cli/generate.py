"""Generate command: load artifacts, initialize memory, run the pipeline."""

from __future__ import annotations

from pathlib import Path

import typer

from toolgenerator.generator.pipeline import run_pipeline
from toolgenerator.graph import ToolGraphSampler, read_tool_graph
from toolgenerator.memory import FakeMemoryStore
from toolgenerator.registry import ToolRegistry

try:
    from toolgenerator.memory import Mem0MemoryStore
except ImportError:  # pragma: no cover - fallback path
    Mem0MemoryStore = None


def _resolve_registry(tools_path: Path | None, registry_path: Path) -> ToolRegistry:
    if tools_path is not None:
        return ToolRegistry.from_toolbench_path(tools_path)
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Registry artifact not found: {registry_path}. Run `toolgenerator build` first or pass --tools-path."
        )
    return ToolRegistry.load_json(registry_path)


def _resolve_memory():
    if Mem0MemoryStore is not None:
        try:
            return Mem0MemoryStore(), "mem0"
        except Exception:
            pass
    return FakeMemoryStore(), "fake"


def run_generate(
    output: Path,
    num: int,
    seed: int,
    corpus_memory_enabled: bool,
    mock_mode: str,
    tools_path: Path | None,
    graph_path: Path,
    llm_model: str,
    pattern: str,
) -> None:
    """Run the generation pipeline with the requested settings."""
    registry_path = Path("artifacts/registry.json")
    registry = _resolve_registry(tools_path, registry_path)

    if not graph_path.is_file():
        raise FileNotFoundError(
            f"Graph artifact not found: {graph_path}. Run `toolgenerator build` first or pass --graph-path."
        )
    graph = read_tool_graph(graph_path)
    graph_sampler = ToolGraphSampler(graph, registry, seed=seed)

    memory, memory_name = _resolve_memory()
    typer.echo(
        f"Starting generation: output={output} num={num} seed={seed} pattern={pattern} mock_mode={mock_mode} corpus_memory={corpus_memory_enabled} memory={memory_name}"
    )
    written = run_pipeline(
        seed=seed,
        output_path=output,
        corpus_memory_enabled=corpus_memory_enabled,
        num_conversations=num,
        registry=registry,
        graph_sampler=graph_sampler,
        memory=memory,
        pattern_type=pattern,
        llm_model=llm_model,
        mock_mode=mock_mode,
    )
    typer.echo(f"Generation complete: wrote {written} record(s) to {output}")
