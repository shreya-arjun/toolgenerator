from __future__ import annotations

import json
from pathlib import Path

from toolgenerator.graph import ToolGraphSampler, build_tool_graph
from toolgenerator.registry import ToolRegistry
from toolgenerator.registry.normalizer import Endpoint


def _write_tool(path: Path, tool_name: str, endpoints: list[str]) -> None:
    payload = {
        "tool_name": tool_name,
        "tool_description": f"{tool_name} data lookup and actions",
        "title": tool_name,
        "api_list": [
            {
                "name": ep,
                "url": f"https://example.com/{tool_name}/{ep}",
                "description": f"{ep} for {tool_name}",
                "method": "GET",
                "required_parameters": [],
                "optional_parameters": [],
            }
            for ep in endpoints
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_rich_registry(tmp_path: Path) -> ToolRegistry:
    cat_a = tmp_path / "CategoryA"
    cat_b = tmp_path / "CategoryB"
    cat_a.mkdir()
    cat_b.mkdir()
    _write_tool(cat_a / "tool_a.json", "ToolA", ["a1", "a2"])
    _write_tool(cat_a / "tool_b.json", "ToolB", ["b1", "b2"])
    _write_tool(cat_b / "tool_c.json", "ToolC", ["c1", "c2"])
    _write_tool(cat_b / "tool_d.json", "ToolD", ["d1", "d2"])
    return ToolRegistry.from_toolbench_path(tmp_path)


def test_multi_step_sampler_returns_endpoint_objects(tmp_path: Path):
    registry = _build_rich_registry(tmp_path)
    graph = build_tool_graph(registry)
    sampler = ToolGraphSampler(graph, registry, seed=42)
    chain = sampler.sample_multi_step_chain(length=3)
    assert isinstance(chain, list)
    assert all(isinstance(e, Endpoint) for e in chain)


def test_multi_step_sampler_returns_at_least_three_endpoints_when_length_three(tmp_path: Path):
    registry = _build_rich_registry(tmp_path)
    graph = build_tool_graph(registry)
    sampler = ToolGraphSampler(graph, registry, seed=42)
    chain = sampler.sample_multi_step_chain(length=3)
    assert len(chain) >= 3


def test_parallel_sampler_returns_endpoint_objects(tmp_path: Path):
    registry = _build_rich_registry(tmp_path)
    graph = build_tool_graph(registry)
    sampler = ToolGraphSampler(graph, registry, seed=42)
    parallel = sampler.sample_parallel(count=2)
    assert isinstance(parallel, list)
    assert all(isinstance(e, Endpoint) for e in parallel)


def test_sampler_same_seed_returns_same_chain(tmp_path: Path):
    registry = _build_rich_registry(tmp_path)
    graph = build_tool_graph(registry)
    chain1 = ToolGraphSampler(graph, registry, seed=42).sample_multi_step_chain(length=3)
    chain2 = ToolGraphSampler(graph, registry, seed=42).sample_multi_step_chain(length=3)
    assert [e.endpoint_id for e in chain1] == [e.endpoint_id for e in chain2]


def test_sampler_different_seeds_return_different_chains(tmp_path: Path):
    registry = _build_rich_registry(tmp_path)
    graph = build_tool_graph(registry)
    chain1 = ToolGraphSampler(graph, registry, seed=1).sample_multi_step_chain(length=4)
    chain2 = ToolGraphSampler(graph, registry, seed=999).sample_multi_step_chain(length=4)
    assert [e.endpoint_id for e in chain1] != [e.endpoint_id for e in chain2]
