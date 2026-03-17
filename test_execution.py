from toolgenerator.execution import Executor, SessionState
from toolgenerator.registry import ToolRegistry
from pathlib import Path

print("Loading registry...")
reg = ToolRegistry.from_toolbench_path(Path('data/toolenv/tools'))
endpoint = reg.list_endpoints()[0]
print(f"Testing with endpoint: {endpoint.endpoint_id}")

# ── Test 1: SessionState ───────────────────────────────────────────
print("\n-- SessionState tests --")
state = SessionState()

state.store("ep1", {"id": 123, "name": "test"})
state.store("ep2", {"result": "ok"})

assert state.get("ep1") == {"id": 123, "name": "test"}
assert state.get("ep2") == {"result": "ok"}
assert state.get("nonexistent") is None
print("PASS: store and get work correctly")

all_outputs = state.get_all()
assert len(all_outputs) == 2
assert all_outputs[0]["endpoint_id"] == "ep1"
assert all_outputs[1]["endpoint_id"] == "ep2"
print("PASS: get_all returns outputs in call order")

# calling same endpoint twice — get() returns latest
state.store("ep1", {"id": 999, "name": "updated"})
assert state.get("ep1")["id"] == 999
print("PASS: get() returns latest output for repeated endpoint")

# ── Test 2: Template mode execution ───────────────────────────────
print("\n-- Executor template mode tests --")
executor = Executor(mock_mode="template", seed=42)
state2 = SessionState()

# Build minimal args from required params
args = {}
for p in endpoint.required_parameters:
    if p.param_type == "integer":
        args[p.name] = 1
    elif p.param_type == "boolean":
        args[p.name] = True
    else:
        args[p.name] = "test"

result = executor.run(endpoint, args, state2)
print(f"Result keys: {list(result.keys())}")
assert "endpoint_id" in result
assert "arguments" in result
assert "output" in result
assert "success" in result
assert "validation_errors" in result
print("PASS: result has correct shape")

if result["success"]:
    assert result["output"] is not None
    stored = state2.get(endpoint.endpoint_id)
    assert stored is not None
    print("PASS: successful execution stored in session state")
else:
    print(f"WARN: execution failed (validation errors: {result['validation_errors']})")

# ── Test 3: Validation catches missing required params ─────────────
print("\n-- Validation tests --")
# Find an endpoint with required parameters
endpoint_with_required = None
for e in reg.list_endpoints():
    if len(e.required_parameters) > 0:
        endpoint_with_required = e
        break

if endpoint_with_required:
    state3 = SessionState()
    bad_result = executor.run(endpoint_with_required, {}, state3)
    assert bad_result["success"] is False
    assert len(bad_result["validation_errors"]) > 0
    assert bad_result["output"] is None
    assert state3.get(endpoint_with_required.endpoint_id) is None
    print(f"PASS: missing required params caught: {bad_result['validation_errors']}")
else:
    print("WARN: no endpoint with required params found to test validation")

# ── Test 4: Determinism ────────────────────────────────────────────
print("\n-- Determinism test --")
ex1 = Executor(mock_mode="template", seed=42)
ex2 = Executor(mock_mode="template", seed=42)
state4a = SessionState()
state4b = SessionState()
result1 = ex1.run(endpoint, args, state4a)
result2 = ex2.run(endpoint, args, state4b)
assert result1["output"] == result2["output"]
print("PASS: same seed produces same output")

print("\nAll Phase 4 checks passed!")