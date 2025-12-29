#!/usr/bin/env python3
"""
Memory_Node.py
Sentience 5.5 – Stable, bounded, low-CPU memory layer

Responsibilities:
- Store interactions (input, classification, response)
- Bounded memory (no runaway growth)
- Zero background loops (idle-safe)
- Simple, transparent state for introspection
"""

from collections import deque
import time


class MemoryNode:
    def __init__(self, max_entries: int = 500):
        # Deque gives us O(1) append/pop and hard bounds
        self._memory = deque(maxlen=max_entries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store(self, text: str, classification: str, response: str):
        """
        Store a single interaction.
        This is intentionally synchronous and lightweight.
        """
        entry = {
            "timestamp": time.time(),
            "text": text,
            "classification": classification,
            "response": response,
        }
        self._memory.append(entry)

    def recent(self, n: int = 5):
        """
        Return the last n memory entries (most recent last).
        """
        if n <= 0:
            return []
        return list(self._memory)[-n:]

    def size(self) -> int:
        """
        Current number of stored memory entries.
        """
        return len(self._memory)

    def snapshot(self):
        """
        Introspection-safe snapshot (no mutation).
        """
        return {
            "entries": len(self._memory),
            "oldest_timestamp": self._memory[0]["timestamp"] if self._memory else None,
            "newest_timestamp": self._memory[-1]["timestamp"] if self._memory else None,
        }

    # ------------------------------------------------------------------
    # Grounding helpers (optional, but useful)
    # ------------------------------------------------------------------
    def last_response(self):
        if not self._memory:
            return None
        return self._memory[-1]["response"]

    def last_classification(self):
        if not self._memory:
            return None
        return self._memory[-1]["classification"]


if __name__ == "__main__":
    # Minimal self-test (does not run in production)
    mem = MemoryNode(max_entries=3)
    mem.store("hello", "greeting", "Hello.")
    mem.store("what is 2+2", "question", "4")
    mem.store("bye", "greeting", "Goodbye.")
    mem.store("overflow", "test", "Oldest dropped")

    print(mem.snapshot())
    print(mem.recent(5))
