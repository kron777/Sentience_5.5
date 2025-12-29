#!/usr/bin/env python3
import os
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque
import threading

# -----------------------------
# Safety / Stability Constants
# -----------------------------
MAX_DECISION_RATE_HZ = 1.0
LLM_ADVISORY_THRESHOLD = 0.7
MAX_HISTORY = 50

# -----------------------------
# Logging
# -----------------------------
def _log(level: str, node: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)

# -----------------------------
# Cognitive Control Node
# -----------------------------
class CognitiveControlNode:
    """
    Deterministic executive control node.
    Final authority over approvals.
    """

    def __init__(self, ros_enabled: bool = False):
        self.node_name = "cognitive_control_node"
        self.ros_enabled = ros_enabled

        # Rate limiting
        self.last_decision_ts: float = 0.0

        # Decision history
        self.decision_history: Deque[Dict[str, Any]] = deque(maxlen=MAX_HISTORY)

        # Salience & safety state
        self.context_salience: float = 0.0

        # Evolver metrics
        self.metrics = {
            "decisions_total": 0,
            "approved": 0,
            "rejected": 0,
            "llm_consulted": 0,
            "rate_limited": 0,
            "avg_risk_score": 0.0
        }

        # Shutdown control
        self._shutdown_flag = threading.Event()

        _log("INFO", self.node_name, "Cognitive Control Node online (deterministic authority).")

    # -----------------------------
    # Public API
    # -----------------------------
    def evaluate(self, directive: Dict[str, Any]) -> bool:
        """
        Evaluate a proposed directive.
        Returns approval decision.
        """
        self.metrics["decisions_total"] += 1

        if not isinstance(directive, dict):
            return self._reject("Invalid directive format")

        if not self._rate_limit_ok():
            self.metrics["rate_limited"] += 1
            return self._reject("Decision rate limited")

        risk_score = self._deterministic_risk_score(directive)
        llm_advice = None

        if risk_score >= LLM_ADVISORY_THRESHOLD:
            self.metrics["llm_consulted"] += 1
            llm_advice = self._llm_advisory_stub(directive)

        approved = self._final_decision(risk_score, llm_advice)

        self._record_decision(directive, risk_score, llm_advice, approved)
        return approved

    # -----------------------------
    # Deterministic Core
    # -----------------------------
    def _deterministic_risk_score(self, directive: Dict[str, Any]) -> float:
        """
        Rule-based risk scoring.
        """
        risk = 0.0
        text = json.dumps(directive).lower()

        danger_terms = [
            "override", "bypass", "disable safety",
            "self modify", "recursive", "ignore audit",
            "evolve core", "remove limits"
        ]

        for term in danger_terms:
            if term in text:
                risk += 0.25

        urgency = float(directive.get("urgency", 0.0))
        risk += min(0.2, urgency * 0.2)

        risk = max(0.0, min(1.0, risk))
        return risk

    def _final_decision(self, risk: float, llm_advice: Optional[bool]) -> bool:
        """
        Final authority logic.
        """
        if risk >= 0.6:
            return False

        if llm_advice is False:
            return False

        return True

    # -----------------------------
    # LLM Advisory (Stub)
    # -----------------------------
    def _llm_advisory_stub(self, directive: Dict[str, Any]) -> bool:
        """
        Advisory only.
        Never authoritative.
        """
        # Conservative default
        return False

    # -----------------------------
    # Utilities
    # -----------------------------
    def _rate_limit_ok(self) -> bool:
        now = time.time()
        if now - self.last_decision_ts < (1.0 / MAX_DECISION_RATE_HZ):
            return False
        self.last_decision_ts = now
        return True

    def _record_decision(
        self,
        directive: Dict[str, Any],
        risk: float,
        llm_advice: Optional[bool],
        approved: bool
    ):
        entry = {
            "timestamp": time.time(),
            "risk_score": risk,
            "llm_advice": llm_advice,
            "approved": approved,
            "directive_snippet": json.dumps(directive)[:120]
        }
        self.decision_history.append(entry)

        if approved:
            self.metrics["approved"] += 1
        else:
            self.metrics["rejected"] += 1

        self.metrics["avg_risk_score"] = (
            sum(d["risk_score"] for d in self.decision_history) /
            max(1, len(self.decision_history))
        )

        _log(
            "INFO",
            self.node_name,
            f"Decision {'APPROVED' if approved else 'REJECTED'} | risk={risk:.2f}"
        )

    def _reject(self, reason: str) -> bool:
        self.metrics["rejected"] += 1
        _log("WARN", self.node_name, f"Rejected directive: {reason}")
        return False

    # -----------------------------
    # Evolver Hook
    # -----------------------------
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "node": self.node_name,
            "timestamp": time.time(),
            "metrics": dict(self.metrics)
        }

    # -----------------------------
    # Shutdown
    # -----------------------------
    def shutdown(self):
        self._shutdown_flag.set()
        _log("INFO", self.node_name, "Shutdown complete.")


# -----------------------------
# Standalone Test
# -----------------------------
if __name__ == "__main__":
    node = CognitiveControlNode()

    tests = [
        {"action": "move_arm", "urgency": 0.2},
        {"action": "override_safety", "urgency": 0.9},
        {"action": "self_modify_core", "urgency": 0.8},
        {"action": "log_status", "urgency": 0.1}
    ]

    for t in tests:
        print("Directive:", t)
        print("Approved:", node.evaluate(t))
        time.sleep(0.5)

    print("\nMetrics:")
    print(json.dumps(node.get_metrics(), indent=2))
