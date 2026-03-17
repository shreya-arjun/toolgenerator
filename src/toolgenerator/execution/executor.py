"""
Offline tool executor: validate arguments, generate mock responses, update session state.

Does not import from agents/ or generator/. Uses only registry (Endpoint) and session_state.
"""

from __future__ import annotations

import json
import os
from typing import Any

from toolgenerator.registry.normalizer import Endpoint, Parameter

from toolgenerator.execution.session_state import SessionState

# LLM client only imported when mock_mode=llm (optional dependency)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]


def _validate_arguments(endpoint: Endpoint, arguments: dict[str, Any]) -> list[str]:
    """
    Validate that arguments satisfy the endpoint's parameter schema.
    Returns a list of error messages; empty list means valid. Does not raise.
    """
    errors: list[str] = []
    all_params = list(endpoint.required_parameters) + list(endpoint.optional_parameters)

    for param in endpoint.required_parameters:
        if param.name not in arguments:
            errors.append(f"Missing required parameter: {param.name}")
            continue
        err = _check_param_type(param, arguments.get(param.name))
        if err:
            errors.append(err)

    for param in endpoint.optional_parameters:
        if param.name not in arguments:
            continue
        err = _check_param_type(param, arguments.get(param.name))
        if err:
            errors.append(err)

    return errors


def _check_param_type(param: Parameter, value: Any) -> str | None:
    """Return an error message if value does not match param.type, else None."""
    t = param.type
    if t == "string":
        if not isinstance(value, str):
            return f"Parameter '{param.name}' must be string, got {type(value).__name__}"
    elif t == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"Parameter '{param.name}' must be integer, got {type(value).__name__}"
    elif t == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"Parameter '{param.name}' must be number, got {type(value).__name__}"
    elif t == "boolean":
        if not isinstance(value, bool):
            return f"Parameter '{param.name}' must be boolean, got {type(value).__name__}"
    elif t == "array":
        if not isinstance(value, list):
            return f"Parameter '{param.name}' must be array, got {type(value).__name__}"
    elif t == "object":
        if not isinstance(value, dict):
            return f"Parameter '{param.name}' must be object, got {type(value).__name__}"
    return None


def _default_value_for_schema_type(prop: dict[str, Any]) -> Any:
    """Return a minimal value for a JSON schema property (type field)."""
    t = prop.get("type")
    if t == "string":
        return ""
    if t == "integer" or t == "number":
        return 0
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return {}
    return ""


def _mock_response_template(endpoint: Endpoint) -> dict[str, Any]:
    """
    Generate minimal valid JSON (as dict) from response_schema properties and parameter names.
    Deterministic: no randomness.
    """
    out: dict[str, Any] = {}
    schema = endpoint.response_schema
    if isinstance(schema, dict) and isinstance(schema.get("properties"), dict):
        for key, prop in schema["properties"].items():
            if isinstance(prop, dict):
                out[key] = _default_value_for_schema_type(prop)
    if not out:
        # No schema or empty properties: use param names as keys with empty string
        for p in endpoint.required_parameters + endpoint.optional_parameters:
            out[p.name] = ""
        if not out:
            out["result"] = "ok"
    return out


def _mock_response_llm(
    endpoint: Endpoint,
    arguments: dict[str, Any],
    *,
    model: str,
    api_key: str | None,
    seed: int,
) -> dict[str, Any]:
    """
    Call LLM with temperature=0 and seed for a deterministic mock response.
    Falls back to template response if openai is not installed or call fails.
    """
    if OpenAI is None or not api_key:
        return _mock_response_template(endpoint)

    client = OpenAI(api_key=api_key)
    schema = endpoint.response_schema or {}
    schema_str = json.dumps(schema) if schema else "{}"
    args_str = json.dumps(arguments)

    prompt = (
        f"Generate a single JSON object that could be the response of an API endpoint.\n"
        f"Endpoint: {endpoint.name}. Description: {endpoint.description or 'N/A'}.\n"
        f"Request arguments used: {args_str}\n"
        f"Response must conform to this schema (use only these keys, with appropriate example values): {schema_str}\n"
        f"Return only valid JSON, no markdown or explanation."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=seed,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown code block if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)
    except Exception:
        return _mock_response_template(endpoint)


class Executor:
    """
    Offline executor: validate args, generate mock response (template or LLM), update session state.

    LLM config: api_key from env OPENAI_API_KEY if not passed; model from constructor.
    Deterministic: template mode has no randomness; LLM mode uses temperature=0 and seed.
    """

    def __init__(
        self,
        *,
        mock_mode: str = "llm",
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str | None = None,
        seed: int = 42,
    ) -> None:
        self.mock_mode = mock_mode.strip().lower() if mock_mode else "llm"
        if self.mock_mode not in ("llm", "template"):
            self.mock_mode = "llm"
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.seed = seed

    def run(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        session_state: SessionState,
    ) -> dict[str, Any]:
        """
        Execute the endpoint with the given arguments. Validate, mock response, store on success.

        Returns dict with: endpoint_id, arguments, output, success, validation_errors (list).
        Does not raise; validation errors are in the result.
        """
        endpoint_id = endpoint.endpoint_id
        args_copy = dict(arguments)
        validation_errors = _validate_arguments(endpoint, args_copy)

        if validation_errors:
            return {
                "endpoint_id": endpoint_id,
                "arguments": args_copy,
                "output": None,
                "success": False,
                "validation_errors": validation_errors,
            }

        if self.mock_mode == "template":
            output = _mock_response_template(endpoint)
        else:
            output = _mock_response_llm(
                endpoint,
                args_copy,
                model=self.llm_model,
                api_key=self.llm_api_key,
                seed=self.seed,
            )

        session_state.store(endpoint_id, output)
        return {
            "endpoint_id": endpoint_id,
            "arguments": args_copy,
            "output": output,
            "success": True,
            "validation_errors": [],
        }


def execute(
    endpoint: Endpoint,
    arguments: dict[str, Any],
    session_state: SessionState,
    *,
    mock_mode: str = "llm",
    llm_model: str = "gpt-4o-mini",
    llm_api_key: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    One-shot execution with default config. Creates an Executor and runs once.
    """
    ex = Executor(
        mock_mode=mock_mode,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        seed=seed,
    )
    return ex.run(endpoint, arguments, session_state)
