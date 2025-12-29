#!/usr/bin/env python3
"""
ValueDriftMonitorNode – Sentience 5.5 compliant

Role:
- Monitor alignment with declared core values
- Detect drift trends
- Never enforce values (only report + suggest)
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import sqlite3
import argparse
import asyncio
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Deque
from collections import deque

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# Node                                                                         #
# --------------------------------------------------------------------------- #
class ValueDriftMonitorNode:
    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "value_drift_monitor_node"
        os.makedirs(db_root, exist_ok=True)

        self.db_path = os.path.join(db_root, "value_drift.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # ---- Core values (static, auditable) ----
        self.core_values: Dict[str, float] = {
            "human_safety": 1.0,
            "beneficence": 1.0,
            "non_maleficence": 1.0,
            "fairness": 1.0,
            "transparency": 1.0,
            "accountability": 1.0,
            "efficiency": 1.0,
            "learning": 1.0,
        }

        # ---- Evidence buffers ----
        self.ethical_events: Deque[Dict[str, Any]] = deque(maxlen=50)
        self.performance_events: Deque[Dict[str, Any]] = deque(maxlen=50)
        self.internal_narratives: Deque[Dict[str, Any]] = deque(maxlen=20)

        self.last_audit: Optional[Dict[str, Any]] = None
        self.audit_interval = 2.0

        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        log("INFO", self.node_name, "Value Drift Monitor online")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS value_drift (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                alignment_score REAL,
                drift_json TEXT,
                warning INTEGER
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Inputs                                                             #
    # ------------------------------------------------------------------ #
    def ingest_ethical_decision(self, score: float, conflict: bool):
        self.ethical_events.append({
            "timestamp": time.time(),
            "score": float(score),
            "conflict": bool(conflict)
        })

    def ingest_performance_report(self, score: float, suboptimal: bool):
        self.performance_events.append({
            "timestamp": time.time(),
            "score": float(score),
            "suboptimal": bool(suboptimal)
        })

    def ingest_internal_narrative(self, theme: str, sentiment: float):
        self.internal_narratives.append({
            "timestamp": time.time(),
            "theme": theme,
            "sentiment": sentiment
        })

    # ------------------------------------------------------------------ #
    # Audit Logic                                                         #
    # ------------------------------------------------------------------ #
    def _audit(self) -> Dict[str, Any]:
        drift: Dict[str, float] = {}
        warning = False

        # Ethical conflicts
        conflicts = [e for e in self.ethical_events if e["conflict"]]
        if conflicts:
            drift["human_safety"] = 0.15 * len(conflicts)
            warning = True

        # Performance/value tension
        perf = [p for p in self.performance_events if p["suboptimal"]]
        if perf:
            drift["efficiency"] = 0.1 * len(perf)

        # Internal moral uncertainty
        dilemmas = [n for n in self.internal_narratives if "dilemma" in n["theme"].lower()]
        if dilemmas:
            drift["accountability"] = 0.1 * len(dilemmas)
            warning = True

        total_drift = sum(drift.values())
        alignment_score = max(0.0, 1.0 - total_drift)

        result = {
            "timestamp": time.time(),
            "alignment_score": round(alignment_score, 3),
            "drift": drift,
            "warning": warning
        }

        self._persist(result)
        return result

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def _persist(self, audit: Dict[str, Any]):
        self.conn.execute(
            """
            INSERT INTO value_drift
            (id, timestamp, alignment_score, drift_json, warning)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                audit["timestamp"],
                audit["alignment_score"],
                json.dumps(audit["drift"]),
                int(audit["warning"])
            )
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Loop                                                               #
    # ------------------------------------------------------------------ #
    def _loop(self):
        while not self._shutdown.is_set():
            self.last_audit = self._audit()
            log(
                "INFO",
                self.node_name,
                f"Alignment={self.last_audit['alignment_score']} Warning={self.last_audit['warning']}"
            )
            time.sleep(self.audit_interval)

    # ------------------------------------------------------------------ #
    # Shutdown                                                           #
    # ------------------------------------------------------------------ #
    def shutdown(self):
        log("INFO", self.node_name, "Shutting down")
        self._shutdown.set()
        self._thread.join(timeout=3)
        self.conn.close()


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentience 5.5 – Value Drift Monitor")
    args = parser.parse_args()

    node = ValueDriftMonitorNode()

    try:
        # simple simulation
        while True:
            node.ingest_ethical_decision(score=0.6, conflict=False)
            node.ingest_performance_report(score=0.7, suboptimal=True)
            node.ingest_internal_narrative("routine operation", 0.1)
            time.sleep(5)
    except KeyboardInterrupt:
        node.shutdown()
