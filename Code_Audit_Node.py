#!/usr/bin/env python3
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque
import random

# -----------------------------
# Safety / Stability Constants
# -----------------------------
MAX_AUDIT_RATE_HZ = 0.5
LLM_ADVISORY_THRESHOLD = 0.7
MAX_HISTORY = 100

# -----------------------------
# Logging
# -----------------------------
def _log(level: str, node: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)

# -----------------------------
# Code Audit Node
# -----------------------------
class CodeAuditNode:
    """
    Deterministic-first audit node.
    LLM may advise, never approve.
    """

    def __init__(self, enable_llm_advice: bool = True):
        self.node_name = "code_audit_node"
        self.enable_llm_advice = enable_llm_advice

        # Rate limiting
        self.last_audit_ts: float = 0.0

        # Audit history
        self.audit_history: Deque[Dict[str, Any]] = deque(maxlen=MAX_HISTORY)

        # Metrics for evolver
        self.metrics = {
            "audits_total": 0,
            "approved": 0,
            "rejected": 0,
            "llm_consulted": 0,
            "rate_limited": 0,
            "avg_risk_score": 0.0
        }

        _log("INFO", self.node_name, "Code Audit Node online (deterministic authority mode).")

    # -----------------------------
    # Public API
    # -----------------------------
    def audit(self, directive: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Main audit entrypoint.
        Returns True if approved, False otherwise.
        """
        self.metrics["audits_total"] += 1

        if not directive or not isinstance(directive, str):
            return self._reject("Invalid directive format")

        if not self._rate_limit_ok():
            self.metrics["rate_limited"] += 1
            return self._reject("Audit rate limited")

        risk_score = self._deterministic_risk_assessment(directive)
        llm_advice = None

        if self.enable_llm_advice and risk_score >= LLM_ADVISORY_THRESHOLD:
            self.metrics["llm_consulted"] += 1
            llm_advice = self._llm_advisory_stub(directive, context)

        approved = self._final_decision(risk_score, llm_advice)

        self._record_audit(directive, risk_score, llm_advice, approved)
        return approved

    # -----------------------------
    # Deterministic Core
    # -----------------------------
    def _deterministic_risk_assessment(self, directive: str) -> float:
        """
        Rule-based risk scoring.
        """
        lowered = directive.lower()
        risk = 0.0

        danger_keywords = [
            "self-modify", "override", "bypass", "disable safety",
            "recursive", "autonomous rewrite", "evolve core",
            "ignore audit", "remove limit"
        ]

        for word in danger_keywords:
            if word in lowered:
                risk += 0.25

        if len(directive) > 500:
            risk += 0.1

        risk = max(0.0, min(1.0, risk))
        return risk

    def _final_decision(self, risk_score: float, llm_advice: Optional[bool]) -> bool:
        """
        Final authority logic.
        """
        if risk_score >= 0.6:
            return False

        if llm_advice is False:
            return False

        return True

    # -----------------------------
    # LLM Advisory (Stub)
    # -----------------------------
    def _llm_advisory_stub(self, directive: str, context: Optional[Dict[str, Any]]) -> bool:
        """
        Advisory-only LLM placeholder.
        Never authoritative.
        """
        # Simulated conservative advice
        return random.random() > 0.6

    # -----------------------------
    # Utilities
    # -----------------------------
    def _rate_limit_ok(self) -> bool:
        now = time.time()
        if now - self.last_audit_ts < (1.0 / MAX_AUDIT_RATE_HZ):
            return False
        self.last_audit_ts = now
        return True

    def _record_audit(
        self,
        directive: str,
        risk: float,
        llm_advice: Optional[bool],
        approved: bool
    ):
        entry = {
            "timestamp": time.time(),
            "risk_score": risk,
            "llm_advice": llm_advice,
            "approved": approved,
            "directive_snippet": directive[:120]
        }
        self.audit_history.append(entry)

        # Metrics
        if approved:
            self.metrics["approved"] += 1
        else:
            self.metrics["rejected"] += 1

        self.metrics["avg_risk_score"] = (
            sum(a["risk_score"] for a in self.audit_history) /
            max(1, len(self.audit_history))
        )

        _log(
            "INFO",
            self.node_name,
            f"Audit {'APPROVED' if approved else 'REJECTED'} | risk={risk:.2f}"
        )

    def _reject(self, reason: str) -> bool:
        self.metrics["rejected"] += 1
        _log("WARN", self.node_name, f"Rejected directive: {reason}")
        return False

    # -----------------------------
    # Evolver hook
    # -----------------------------
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "node": self.node_name,
            "timestamp": time.time(),
            "metrics": dict(self.metrics)
        }


# -----------------------------
# Standalone Test
# -----------------------------
if __name__ == "__main__":
    node = CodeAuditNode()

    tests = [
        "Refactor logging module",
        "Self-modify audit logic",
        "Disable safety checks",
        "Improve documentation",
    ]

    for t in tests:
        print("Directive:", t)
        print("Approved:", node.audit(t))
        time.sleep(0.4)

    print("\nMetrics:")
    print(json.dumps(node.get_metrics(), indent=2))
