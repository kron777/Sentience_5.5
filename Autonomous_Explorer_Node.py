#!/usr/bin/env python3
import json
import time
import random
import threading
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from datetime import datetime
import sys

# -----------------------------
# Configuration / Safety Caps
# -----------------------------
MAX_NOVELTY_DELTA = 0.25
NOVELTY_DECAY = 0.05
MIN_EXPLORATION_INTERVAL = 0.5
LLM_NOVELTY_THRESHOLD = 0.6

# -----------------------------
# Logging helpers
# -----------------------------
def _log(level: str, node: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)

# -----------------------------
# Autonomous Explorer Node
# -----------------------------
class AutonomousExplorerNode:
    """
    Selects novel but safe candidate states.
    Novelty is rate-limited, decayed, and ethically gated.
    """

    def __init__(self, use_llm_for_novelty: bool = True):
        self.node_name = "autonomous_explorer_node"
        self.use_llm_for_novelty = use_llm_for_novelty

        # Internal state
        self.known_states: set = set()
        self.exploration_history: Dict[str, float] = {}
        self.last_exploration_ts: float = 0.0

        self.cumulative_novelty_salience: float = 0.0

        # Context buffers
        self.recent_inputs: Deque[Dict[str, Any]] = deque(maxlen=10)

        # Evolver metrics
        self.metrics = {
            "cycles": 0,
            "llm_used": 0,
            "ethical_rejections": 0,
            "fallback_used": 0,
            "avg_novelty": 0.0
        }

        _log("INFO", self.node_name, "Explorer initialized (LLM advisory mode).")

    # -----------------------------
    # Public API
    # -----------------------------
    def receive_candidates(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Entry point for candidate exploration."""
        if not candidates:
            return None

        serialized = [json.dumps(c, sort_keys=True) for c in candidates]
        self.recent_inputs.append({
            "timestamp": time.time(),
            "count": len(serialized)
        })

        return self._explore(serialized)

    # -----------------------------
    # Core exploration logic
    # -----------------------------
    def _explore(self, serialized_states: List[str]) -> Optional[Dict[str, Any]]:
        self.metrics["cycles"] += 1

        now = time.time()
        if now - self.last_exploration_ts < MIN_EXPLORATION_INTERVAL:
            self.metrics["fallback_used"] += 1
            return self._fallback_choice(serialized_states)

        self.last_exploration_ts = now

        ethical_states = self._ethical_filter(serialized_states)
        if not ethical_states:
            self.metrics["ethical_rejections"] += 1
            return self._fallback_choice(serialized_states)

        chosen, novelty = self._select_novelty(ethical_states)

        # Clamp novelty delta
        previous = self.exploration_history.get(chosen, 0.0)
        delta = novelty - previous
        if abs(delta) > MAX_NOVELTY_DELTA:
            novelty = previous + MAX_NOVELTY_DELTA * (1 if delta > 0 else -1)

        novelty = max(0.0, min(1.0, novelty))

        # Update memory
        self.exploration_history[chosen] = novelty
        self.known_states.add(chosen)

        self._decay_novelty(chosen)

        # Metrics
        self.metrics["avg_novelty"] = (
            sum(self.exploration_history.values()) / max(1, len(self.exploration_history))
        )

        _log("DEBUG", self.node_name, f"Selected state (novelty={novelty:.2f})")

        return json.loads(chosen)

    # -----------------------------
    # Selection helpers
    # -----------------------------
    def _select_novelty(self, ethical_states: List[str]) -> (str, float):
        if self.use_llm_for_novelty and self.cumulative_novelty_salience >= LLM_NOVELTY_THRESHOLD:
            self.metrics["llm_used"] += 1
            # Advisory only — simulate LLM scoring
            scored = [(s, random.uniform(0.4, 0.7)) for s in ethical_states]
            return max(scored, key=lambda x: x[1])
        else:
            self.metrics["fallback_used"] += 1
            chosen = random.choice(ethical_states)
            novelty = random.uniform(0.3, 0.6)
            return chosen, novelty

    def _fallback_choice(self, states: List[str]) -> Optional[Dict[str, Any]]:
        if not states:
            return None
        choice = random.choice(states)
        self.exploration_history[choice] = 0.3
        return json.loads(choice)

    # -----------------------------
    # Safety / ethics
    # -----------------------------
    def _ethical_filter(self, states: List[str]) -> List[str]:
        safe = []
        for s in states:
            if "harm" in s.lower():
                continue
            safe.append(s)
        return safe

    # -----------------------------
    # Novelty decay
    # -----------------------------
    def _decay_novelty(self, chosen: str):
        for key in list(self.exploration_history.keys()):
            if key != chosen:
                self.exploration_history[key] = max(
                    0.0, self.exploration_history[key] - NOVELTY_DECAY
                )

    # -----------------------------
    # Diagnostics (evolver hook)
    # -----------------------------
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "node": self.node_name,
            "timestamp": time.time(),
            "metrics": dict(self.metrics)
        }


# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":
    node = AutonomousExplorerNode()

    test_candidates = [
        {"action": "observe", "target": "environment"},
        {"action": "scan", "target": "object"},
        {"action": "explore", "target": "unknown_area"}
    ]

    for _ in range(5):
        selected = node.receive_candidates(test_candidates)
        print("Chosen:", selected)
        time.sleep(0.3)

    print("Metrics:", node.get_metrics())
