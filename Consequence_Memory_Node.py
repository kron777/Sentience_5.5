#!/usr/bin/env python3
from collections import defaultdict
from datetime import datetime, timezone

class ConsequenceMemoryNode:
    def __init__(self):
        self.node_name = "Consequence_Memory_Node"
        self.failures = defaultdict(int)
        self.successes = defaultdict(int)

    def record(self, domain: str, success: bool):
        if success:
            self.successes[domain] += 1
        else:
            self.failures[domain] += 1

    def adjustments(self):
        adj = {}
        for k, v in self.failures.items():
            if v >= 3:
                adj[k] = "reduce_confidence"
        for k, v in self.successes.items():
            if v >= 3:
                adj[k] = "increase_confidence"
        return adj

    def snapshot(self):
        return {
            "failures": dict(self.failures),
            "successes": dict(self.successes),
            "time": datetime.now(timezone.utc).isoformat()
        }
