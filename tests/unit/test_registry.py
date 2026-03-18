from __future__ import annotations

import json
from pathlib import Path

from toolgenerator.registry.loader import load_toolbench_tools
from toolgenerator.registry.normalizer import normalize_tool
from toolgenerator.registry.registry import ToolRegistry


def test_parse_valid_tool_json_produces_tool_endpoint_parameter():
    raw = {
        "tool_name": "Weather",
        "tool_description": "Weather lookup",
        "title": "Weather",
        "api_list": [
            {
                "name": "get_weather",
                "url": "https://example.com/weather",
                "description": "Get weather",
                "method": "POST",
                "required_parameters": [
                    {"name": "city", "type": "STRING", "description": "City name", "default": "SF"}
                ],
                "optional_parameters": [
                    {"name": "units", "type": "string", "description": "Units", "default": "metric"}
                ],
            }
        ],
    }
    tool = normalize_tool(raw, "Weather")
    assert tool is not None
    assert tool.tool_name == "Weather"
    assert len(tool.endpoints) == 1
    endpoint = tool.endpoints[0]
    assert endpoint.name == "get_weather"
    assert endpoint.method == "POST"
    assert len(endpoint.required_parameters) == 1
    assert endpoint.required_parameters[0].name == "city"
    assert endpoint.required_parameters[0].type == "string"


def test_missing_url_defaults_to_empty_string():
    raw = {
        "tool_name": "NoUrl",
        "title": "NoUrl",
        "api_list": [{"name": "lookup", "required_parameters": [], "optional_parameters": []}],
    }
    tool = normalize_tool(raw, "Misc")
    assert tool is not None
    assert tool.endpoints[0].url == ""


def test_missing_method_defaults_to_get():
    raw = {
        "tool_name": "NoMethod",
        "title": "NoMethod",
        "api_list": [{"name": "lookup", "required_parameters": [], "optional_parameters": []}],
    }
    tool = normalize_tool(raw, "Misc")
    assert tool is not None
    assert tool.endpoints[0].method == "GET"


def test_missing_type_defaults_to_string():
    raw = {
        "tool_name": "NoType",
        "title": "NoType",
        "api_list": [
            {
                "name": "lookup",
                "required_parameters": [{"name": "query", "description": "Query"}],
                "optional_parameters": [],
            }
        ],
    }
    tool = normalize_tool(raw, "Misc")
    assert tool is not None
    assert tool.endpoints[0].required_parameters[0].type == "string"


def test_empty_api_list_returns_none():
    raw = {"tool_name": "Empty", "title": "Empty", "api_list": []}
    assert normalize_tool(raw, "Misc") is None


def test_invalid_json_file_is_skipped_without_crashing(tmp_path: Path):
    category = tmp_path / "Weather"
    category.mkdir()
    (category / "bad.json").write_text("{invalid json", encoding="utf-8")
    loaded = load_toolbench_tools(tmp_path)
    assert loaded == []


def test_registry_lookup_by_tool_id_and_endpoint_id(sample_registry: ToolRegistry):
    tools = sample_registry.list_tools()
    assert len(tools) >= 1
    tool = tools[0]
    endpoint = tool.endpoints[0]
    assert sample_registry.get_tool(tool.tool_id) is not None
    assert sample_registry.get_endpoint(endpoint.endpoint_id) is not None
