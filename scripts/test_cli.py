"""
Manual CLI integration test.
Run with: python test_cli.py
Tests all four CLI commands end-to-end without needing an OpenAI key.
"""

import subprocess
import sys
import json
from pathlib import Path

def run(cmd):
    """Run a CLI command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def check(condition, label):
    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")

print("=" * 50)
print("CLI Integration Test")
print("=" * 50)

# ── Test 1: help works ─────────────────────────────────────────────
print("\n-- Help --")
code, out, err = run([sys.executable, "-m", "toolgenerator.cli.main", "--help"])
check(code == 0, "CLI --help exits cleanly")
check("build" in out, "build command listed in help")
check("generate" in out, "generate command listed in help")
check("validate" in out, "validate command listed in help")
check("metrics" in out, "metrics command listed in help")

# ── Test 2: build ──────────────────────────────────────────────────
print("\n-- Build --")
code, out, err = run([
    sys.executable, "-m", "toolgenerator.cli.main",
    "build",
    "--tools-path", "data/toolenv/tools",
    "--output-dir", "artifacts"
])
check(code == 0, "build exits cleanly")
check(Path("artifacts/registry.json").exists(), "registry.json created")
check(Path("artifacts/tool_graph.gpickle").exists(), "tool_graph.gpickle created")
check("tools" in out.lower() or "endpoints" in out.lower(), "build prints counts")
if code != 0:
    print(f"  stderr: {err[:300]}")

# ── Test 3: validate on existing output ───────────────────────────
print("\n-- Validate --")
if Path("artifacts/test_pipeline_output.jsonl").exists():
    code, out, err = run([
        sys.executable, "-m", "toolgenerator.cli.main",
        "validate",
        "--input", "artifacts/test_pipeline_output.jsonl"
    ])
    check(code == 0, "validate exits cleanly")
    check("valid" in out.lower(), "validate prints valid/invalid summary")
    if code != 0:
        print(f"  stderr: {err[:300]}")
else:
    print("SKIP: artifacts/test_pipeline_output.jsonl not found - run test_pipeline.py first")

# ── Test 4: metrics on existing output ────────────────────────────
print("\n-- Metrics --")
if Path("artifacts/test_pipeline_output.jsonl").exists():
    code, out, err = run([
        sys.executable, "-m", "toolgenerator.cli.main",
        "metrics",
        "--input", "artifacts/test_pipeline_output.jsonl"
    ])
    check(code == 0, "metrics exits cleanly")
    check("unique_tool_chain_ratio" in out.lower() or "ratio" in out.lower(),
          "metrics prints tool chain ratio")
    check("distinct" in out.lower() or "gram" in out.lower(),
          "metrics prints n-gram diversity")
    if code != 0:
        print(f"  stderr: {err[:300]}")
else:
    print("SKIP: artifacts/test_pipeline_output.jsonl not found - run test_pipeline.py first")

# ── Test 5: validate fails gracefully on bad input ─────────────────
print("\n-- Edge cases --")
bad_path = "artifacts/nonexistent_file.jsonl"
code, out, err = run([
    sys.executable, "-m", "toolgenerator.cli.main",
    "validate",
    "--input", bad_path
])
check(code != 0 or "error" in out.lower() or "error" in err.lower(),
      "validate handles missing file gracefully")

# ── Test 6: generate fails fast without artifacts ─────────────────
import shutil, os
temp_artifacts = Path("artifacts/_backup")
temp_artifacts.mkdir(parents=True, exist_ok=True)

# Temporarily hide artifacts to test fail-fast behavior
registry_path = Path("artifacts/registry.json")
graph_path = Path("artifacts/tool_graph.gpickle")

if registry_path.exists() and graph_path.exists():
    shutil.copy(registry_path, temp_artifacts / "registry.json")
    shutil.copy(graph_path, temp_artifacts / "tool_graph.gpickle")
    registry_path.unlink()
    graph_path.unlink()

    code, out, err = run([
        sys.executable, "-m", "toolgenerator.cli.main",
        "generate",
        "--output", "artifacts/test_generate.jsonl",
        "--num", "1",
        "--seed", "42",
        "--mock-mode", "template"
    ])
    check(code != 0 or "error" in (out + err).lower(),
          "generate fails fast when artifacts missing")

    # Restore artifacts
    shutil.copy(temp_artifacts / "registry.json", registry_path)
    shutil.copy(temp_artifacts / "tool_graph.gpickle", graph_path)
    shutil.rmtree(temp_artifacts)
else:
    print("SKIP: artifacts not built yet - run build first")

print("\n" + "=" * 50)
print("Done. Fix any FAIL lines before moving to Phase 8 tests.")
print("=" * 50)