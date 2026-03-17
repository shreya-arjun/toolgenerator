"""MemoryStore abstraction for session and corpus scopes (mem0-backed or fake for tests)."""

from toolgenerator.memory.interface import FakeMemoryStore, MemoryStore

__all__ = [
    "FakeMemoryStore",
    "MemoryStore",
]

try:
    from toolgenerator.memory.mem0_store import Mem0MemoryStore
    __all__.append("Mem0MemoryStore")
except ImportError:
    pass  # mem0ai not installed; only FakeMemoryStore available
