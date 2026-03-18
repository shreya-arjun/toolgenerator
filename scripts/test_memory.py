from toolgenerator.memory import FakeMemoryStore, Mem0MemoryStore

# ── FakeMemoryStore tests ──────────────────────────────────────────
fake = FakeMemoryStore()

# Test 1: add then search returns the entry
fake.add("weather API returned sunny", scope="session", metadata={"conversation_id": "conv1", "step": 1})
results = fake.search("weather", scope="session", conversation_id="conv1")
assert len(results) > 0
assert results[0]["content"] == "weather API returned sunny"
print("PASS: add -> search returns entry")

# Test 2: scope isolation — session conv1 vs conv2
fake.add("maps API returned New York", scope="session", metadata={"conversation_id": "conv2", "step": 1})
conv1_results = fake.search("API", scope="session", conversation_id="conv1")
conv2_results = fake.search("API", scope="session", conversation_id="conv2")
assert all("New York" not in r["content"] for r in conv1_results)
assert all("sunny" not in r["content"] for r in conv2_results)
print("PASS: session scope isolation (conv1 vs conv2)")

# Test 3: session vs corpus isolation
fake.add("corpus summary: travel tools used", scope="corpus", metadata={"conversation_id": "conv1"})
session_results = fake.search("travel", scope="session", conversation_id="conv1")
corpus_results = fake.search("travel", scope="corpus")
assert all("corpus summary" not in r["content"] for r in session_results)
assert len(corpus_results) > 0
print("PASS: session vs corpus scope isolation")

# ── Mem0MemoryStore smoke test ─────────────────────────────────────
try:
    mem = Mem0MemoryStore()
    mem.add("test tool output", scope="session", metadata={"conversation_id": "test1", "step": 1})
    results = mem.search("tool output", scope="session", conversation_id="test1")
    assert isinstance(results, list)
    print("PASS: Mem0MemoryStore add -> search returns list")
except Exception as e:
    print(f"WARN: Mem0MemoryStore failed (may be expected): {e}")