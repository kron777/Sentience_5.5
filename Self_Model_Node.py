#!/usr/bin/env python3
"""
Self_Model_Node.py

Purpose:
- Maintain an internal, inspectable model of the system
- Track registered nodes, heartbeats, and recent events
- Provide grounded self-introspection without theatre or hallucination
- Act as the system’s “truth ledger”, not its personality
"""

import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any


class SelfModelNode:
    def __init__(self):
        self.node_name = "Self_Model_Node"

        # Registered nodes and status
        self.nodes: Dict[str, Dict[str, Any]] = {}

        # Event history (bounded)
        self.event_log = []
        self.max_events = 200

        # Lock for thread-safety
        self._lock = threading.Lock()

        self._log("INFO", "Self_Model_Node initialized")

    # ------------------------------------------------------------------
    # Logging (internal only)
    # ------------------------------------------------------------------
    def _log(self, level: str, msg: str):
        ts = datetime.now(timezone.utc).isoformat()
        print(f"[{ts}] [SELF_MODEL] [{level}] {msg}")

    # ------------------------------------------------------------------
    # Node registration & heartbeat
    # ------------------------------------------------------------------
    def register_node(self, node_name: str, metadata: Dict[str, Any] | None = None):
        with self._lock:
            self.nodes[node_name] = {
                "status": "ALIVE",
                "last_heartbeat": time.time(),
                "metadata": metadata or {}
            }
        self._log("INFO", f"Registered node: {node_name}")

    def heartbeat(self, node_name: str):
        with self._lock:
            if node_name not in self.nodes:
                self.register_node(node_name)
            else:
                self.nodes[node_name]["last_heartbeat"] = time.time()
                self.nodes[node_name]["status"] = "ALIVE"

    # ------------------------------------------------------------------
    # Event observation
    # ------------------------------------------------------------------
    def observe_event(self, event: Dict[str, Any]):
        """
        Observes an event routed by the orchestrator.
        Does not interpret, only records.
        """
        with self._lock:
            self.event_log.append({
                "timestamp": time.time(),
                "event": event
            })
            if len(self.event_log) > self.max_events:
                self.event_log.pop(0)

        source = event.get("source", "unknown")
        self._log("EVENT", f"Observed event from {source}: {event}")

    # ------------------------------------------------------------------
    # Introspection API
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """
        Returns a grounded snapshot of internal state.
        """
        with self._lock:
            return {
                "timestamp": time.time(),
                "node": self.node_name,
                "nodes": self._node_snapshot(),
                "recent_events": list(self.event_log)[-10:],
                "health": self._health_assessment()
            }

    def _node_snapshot(self) -> Dict[str, Any]:
        snapshot = {}
        now = time.time()
        for name, data in self.nodes.items():
            age = now - data["last_heartbeat"]
            snapshot[name] = {
                "status": "STALE" if age > 5.0 else data["status"],
                "last_heartbeat": data["last_heartbeat"],
                "seconds_since_heartbeat": round(age, 2),
                "metadata": data["metadata"]
            }
        return snapshot

    def _health_assessment(self) -> str:
        """
        Simple grounded system health indicator.
        """
        stale = [
            name for name, data in self.nodes.items()
            if (time.time() - data["last_heartbeat"]) > 5.0
        ]
        if stale:
            return f"DEGRADED ({len(stale)} stale nodes)"
        return "HEALTHY"

    # ------------------------------------------------------------------
    # Grounded self-knowledge helpers
    # ------------------------------------------------------------------
    def known_facts(self) -> Dict[str, Any]:
        """
        Facts directly supported by internal state.
        """
        return {
            "running": True,
            "registered_nodes": list(self.nodes.keys()),
            "event_count": len(self.event_log),
            "time_utc": datetime.now(timezone.utc).isoformat()
        }

    def unknowns(self) -> list[str]:
        """
        Explicitly acknowledged unknowns.
        """
        return [
            "external world state",
            "user intentions beyond text",
            "future events",
            "subjective experience"
        ]


if __name__ == "__main__":
    # Standalone sanity check
    sm = SelfModelNode()
    sm.register_node("Example_Node")
    sm.observe_event({"type": "test", "source": "unit_test"})
    print(sm.snapshot())
