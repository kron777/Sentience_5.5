#!/usr/bin/env python3
"""
evolver.py
----------
External learning-pressure engine for Sentience nodes.

This module:
- observes node metrics
- applies decay
- applies learning pressure
- updates node parameters safely
- never calls LLMs
- never narrates
"""

import json
import time
import threading
import sqlite3
from pathlib import Path
from typing import Dict, Any

DB_PATH = Path("./evolver_state.db")
STATE_SAVE_INTERVAL = 2.0

# -----------------------------
# Learning State per Node
# -----------------------------

DEFAULT_NODE_STATE = {
    "threshold": 0.7,
    "sensitivity": 1.0,
    "confidence": 1.0,
    "learning_rate": 0.05,
    "decay_rate": 0.01,
    "last_update": 0.0
}


class Evolver:
    def __init__(self):
        self.node_states: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._setup_db()
        self._load_state()

    # -----------------------------
    # Persistence
    # -----------------------------

    def _setup_db(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS node_state (
                node TEXT PRIMARY KEY,
                state_json TEXT,
                updated REAL
            )
        """)
        self.conn.commit()

    def _load_state(self):
        for row in self.cur.execute("SELECT node, state_json FROM node_state"):
            self.node_states[row[0]] = json.loads(row[1])

    def _save_state(self):
        with self._lock:
            for node, state in self.node_states.items():
                self.cur.execute("""
                    INSERT OR REPLACE INTO node_state (node, state_json, updated)
                    VALUES (?, ?, ?)
                """, (node, json.dumps(state), time.time()))
            self.conn.commit()

    # -----------------------------
    # Core Learning Logic
    # -----------------------------

    def process_metrics(self, metrics: Dict[str, Any]):
        node = metrics.get("node")
        if not node:
            return

        with self._lock:
            state = self.node_states.setdefault(node, DEFAULT_NODE_STATE.copy())

            # --- Extract signals ---
            success = float(metrics.get("outcome_success", 0.5))
            confidence = float(metrics.get("prediction_confidence", 0.5))
            false_pos = bool(metrics.get("false_positive", False))
            false_neg = bool(metrics.get("false_negative", False))

            # --- Compute error ---
            error = success - confidence

            # --- Learning update ---
            lr = state["learning_rate"]
            state["threshold"] -= lr * error
            state["sensitivity"] += lr * error
            state["confidence"] *= (1.0 - state["decay_rate"])

            # --- Penalties ---
            if false_pos:
                state["threshold"] += lr * 0.2
            if false_neg:
                state["threshold"] -= lr * 0.2

            # --- Clamp values ---
            state["threshold"] = max(0.1, min(0.95, state["threshold"]))
            state["sensitivity"] = max(0.1, min(3.0, state["sensitivity"]))
            state["confidence"] = max(0.1, min(1.0, state["confidence"]))

            state["last_update"] = time.time()

    # -----------------------------
    # Node Query Interface
    # -----------------------------

    def get_node_params(self, node: str) -> Dict[str, float]:
        """
        Nodes can call this to fetch their current adaptive parameters.
        """
        return self.node_states.get(node, DEFAULT_NODE_STATE)

    # -----------------------------
    # Background Save Loop
    # -----------------------------

    def run(self):
        try:
            while True:
                time.sleep(STATE_SAVE_INTERVAL)
                self._save_state()
        except KeyboardInterrupt:
            self._save_state()
            print("Evolver shutdown cleanly.")


# -----------------------------
# Example usage (manual feed)
# -----------------------------
if __name__ == "__main__":
    evolver = Evolver()

    # Example simulated metric input
    evolver.process_metrics({
        "node": "memory_node",
        "prediction_confidence": 0.8,
        "outcome_success": 0.3,
        "false_positive": True
    })

    evolver.run()
