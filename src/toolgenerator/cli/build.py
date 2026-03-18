"""Build command: registry + tool graph artifacts."""

from __future__ import annotations

from pathlib import Path

import typer

from toolgenerator.graph import build_tool_graph, write_tool_graph
from toolgenerator.registry import ToolRegistry


def run_build(tools_path: Path, output_dir: Path) -> None:
    """Build and save registry.json and tool_graph.gpickle artifacts."""
    registry = ToolRegistry.from_toolbench_path(tools_path)
    graph = build_tool_graph(registry)

    registry_path = output_dir / "registry.json"
    graph_path = output_dir / "tool_graph.gpickle"

    registry.save_json(registry_path)
    write_tool_graph(graph, graph_path)

    typer.echo(f"Saved registry: {registry_path}")
    typer.echo(
        f"  tools={len(registry)} endpoints={len(registry.list_endpoints())}"
    )
    typer.echo(f"Saved graph: {graph_path}")
    typer.echo(
        f"  nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}"
    )
