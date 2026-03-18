from __future__ import annotations

import os
import uuid

import pytest

from toolgenerator.memory import FakeMemoryStore

try:
    from toolgenerator.memory import Mem0MemoryStore
except ImportError:  # pragma: no cover
    Mem0MemoryStore = None


MEM0_READY = Mem0MemoryStore is not None and bool(os.environ.get("OPENAI_API_KEY"))


def _run_common_memory_tests(store):
    token_session_1 = f"session-token-{uuid.uuid4()}"
    token_session_2 = f"session-token-{uuid.uuid4()}"
    token_corpus = f"corpus-token-{uuid.uuid4()}"

    store.add(token_session_1, scope="session:conv1", metadata={"conversation_id": "conv1"})
    results = store.search(token_session_1, scope="session:conv1")
    assert len(results) >= 1
    assert any(token_session_1 in (r.get("content") or "") for r in results)

    store.add(token_session_2, scope="session:conv2", metadata={"conversation_id": "conv2"})
    conv2_results = store.search(token_session_1, scope="session:conv2")
    assert all(token_session_1 not in (r.get("content") or "") for r in conv2_results)

    store.add(token_corpus, scope="corpus", metadata={"conversation_id": "conv1"})
    session_results = store.search(token_corpus, scope="session:conv1")
    assert all(token_corpus not in (r.get("content") or "") for r in session_results)

    corpus_results = store.search(token_session_1, scope="corpus")
    assert all(token_session_1 not in (r.get("content") or "") for r in corpus_results)


def test_fake_memory_add_then_search_returns_entry():
    store = FakeMemoryStore()
    store.add("hello-memory", scope="session:conv1", metadata={"conversation_id": "conv1"})
    results = store.search("hello-memory", scope="session:conv1")
    assert len(results) >= 1
    assert results[0]["content"] == "hello-memory"


def test_fake_memory_session_conv_isolation():
    store = FakeMemoryStore()
    store.add("conv1-only", scope="session:conv1", metadata={"conversation_id": "conv1"})
    results = store.search("conv1-only", scope="session:conv2")
    assert all("conv1-only" not in r["content"] for r in results)


def test_fake_memory_session_not_returned_in_corpus():
    store = FakeMemoryStore()
    store.add("session-only", scope="session:conv1", metadata={"conversation_id": "conv1"})
    results = store.search("session-only", scope="corpus")
    assert all("session-only" not in r["content"] for r in results)


def test_fake_memory_corpus_not_returned_in_session():
    store = FakeMemoryStore()
    store.add("corpus-only", scope="corpus", metadata={"conversation_id": "conv1"})
    results = store.search("corpus-only", scope="session:conv1")
    assert all("corpus-only" not in r["content"] for r in results)


@pytest.mark.slow
@pytest.mark.skipif(not MEM0_READY, reason="mem0 or OPENAI_API_KEY not available")
def test_mem0_memory_add_then_search_returns_entry():
    store = Mem0MemoryStore()
    token = f"mem0-token-{uuid.uuid4()}"
    store.add(token, scope="session:conv1", metadata={"conversation_id": "conv1"})
    results = store.search(token, scope="session:conv1")
    assert len(results) >= 1
    assert any(token in (r.get("content") or "") for r in results)


@pytest.mark.slow
@pytest.mark.skipif(not MEM0_READY, reason="mem0 or OPENAI_API_KEY not available")
def test_mem0_memory_session_conv_isolation():
    store = Mem0MemoryStore()
    token = f"mem0-conv1-{uuid.uuid4()}"
    store.add(token, scope="session:conv1", metadata={"conversation_id": "conv1"})
    results = store.search(token, scope="session:conv2")
    assert all(token not in (r.get("content") or "") for r in results)


@pytest.mark.slow
@pytest.mark.skipif(not MEM0_READY, reason="mem0 or OPENAI_API_KEY not available")
def test_mem0_memory_session_not_returned_in_corpus():
    store = Mem0MemoryStore()
    token = f"mem0-session-{uuid.uuid4()}"
    store.add(token, scope="session:conv1", metadata={"conversation_id": "conv1"})
    results = store.search(token, scope="corpus")
    assert all(token not in (r.get("content") or "") for r in results)


@pytest.mark.slow
@pytest.mark.skipif(not MEM0_READY, reason="mem0 or OPENAI_API_KEY not available")
def test_mem0_memory_corpus_not_returned_in_session():
    store = Mem0MemoryStore()
    token = f"mem0-corpus-{uuid.uuid4()}"
    store.add(token, scope="corpus", metadata={"conversation_id": "conv1"})
    results = store.search(token, scope="session:conv1")
    assert all(token not in (r.get("content") or "") for r in results)
