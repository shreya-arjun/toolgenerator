"""mem0-backed MemoryStore implementation. Only module that imports mem0."""

from __future__ import annotations

from toolgenerator.memory.interface import MemoryStore, _user_id

# Only place in the codebase that imports mem0
from mem0 import Memory  # noqa: I001


class Mem0MemoryStore(MemoryStore):
    """
    MemoryStore backed by mem0 (in-process Qdrant).

    Scope isolation:
    - scope="session:{conversation_id}" → user_id="session:{conversation_id}"
    - scope="corpus" → user_id="corpus"
    """

    def __init__(self) -> None:
        self._memory = Memory()

    def add(self, content: str, scope: str, metadata: dict) -> None:
        conversation_id = metadata.get("conversation_id")
        user_id = _user_id(scope, conversation_id)
        self._memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=user_id,
            metadata=metadata,
        )

    def search(
        self,
        query: str,
        scope: str,
        top_k: int = 5,
        conversation_id: str | None = None,
    ) -> list[dict]:
        user_id = _user_id(scope, conversation_id)
        raw = self._memory.search(
            query=query,
            user_id=user_id,
            limit=top_k,
        )
        # mem0 may return {"results": [...]} or a list; each item has "memory" (content) and optionally "metadata"
        if isinstance(raw, list):
            results = raw
        elif isinstance(raw, dict):
            results = raw.get("results", [])
        else:
            results = []
        if not isinstance(results, list):
            results = []
        return [
            {
                "content": item.get("memory", ""),
                "metadata": item.get("metadata", {}),
            }
            for item in results
            if isinstance(item, dict)
        ]
