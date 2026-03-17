"""Abstract MemoryStore interface and FakeMemoryStore for tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryStore(ABC):
    """
    Abstract memory store: add content under a scope, search by query within a scope.

    Scopes used in this project:
    - "session": in-conversation grounding (tool outputs); isolated per conversation.
    - "corpus": cross-conversation (summaries); shared.

    Other components must depend only on this interface, not on any concrete backend.
    """

    @abstractmethod
    def add(self, content: str, scope: str, metadata: dict) -> None:
        """Store content under the given scope. metadata is stored with the entry."""
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        scope: str,
        top_k: int = 5,
        conversation_id: str | None = None,
    ) -> list[dict]:
        """
        Search within a scope; return list of dicts with at least "content" and optionally "metadata".

        For scope="session", conversation_id must be provided so the store can
        query only that conversation's entries. For scope="corpus", conversation_id
        is ignored.
        """
        ...


def _user_id(scope: str, conversation_id: str | None) -> str:
    """Build mem0 user_id from scope and optional conversation_id."""
    if scope == "session":
        return f"session:{conversation_id or ''}"
    if scope == "corpus":
        return "corpus"
    return scope


class FakeMemoryStore(MemoryStore):
    """
    In-memory, dict-backed MemoryStore for unit tests.

    No mem0 dependency. Scope isolation is implemented: entries under
    user_id A are not returned when searching under user_id B.
    """

    def __init__(self) -> None:
        # user_id -> list of {"content": str, "metadata": dict}
        self._store: dict[str, list[dict[str, Any]]] = {}

    def add(self, content: str, scope: str, metadata: dict) -> None:
        user_id = _user_id(scope, metadata.get("conversation_id"))
        if user_id not in self._store:
            self._store[user_id] = []
        self._store[user_id].append({"content": content, "metadata": dict(metadata)})

    def search(
        self,
        query: str,
        scope: str,
        top_k: int = 5,
        conversation_id: str | None = None,
    ) -> list[dict]:
        user_id = _user_id(scope, conversation_id)
        entries = self._store.get(user_id, [])
        # Simple "search": return all entries for this scope, capped by top_k
        # (no real retrieval; tests only need add -> search returns stored entry)
        return list(entries)[:top_k]
