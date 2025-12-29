#!/usr/bin/env python3
"""
Feedback_Loop_Node (UPDATED – Evolver-safe)

Purpose:
- Aggregate outcomes from action, emotion, motivation, value drift, and world state
- Generate deterministic feedback signals
- Emit cognitive directives (NO decision authority)
- Provide learning metrics to evolver.py

LLM REMOVED.
All feedback is rule-based, auditable, and bounded.
"""

import os
import json
import time
import sqlite3
import threading
from collections import deque
from uuid import uuid4
from typing import Dict, Any, Optional

# -------------------------
# Logging
# -------------------------
def log(level: str, msg: str):
    print(f"[{time.time():.2f}] [FeedbackLoop] [{level}] {msg}")

# -------------------------
# Feedback Loop Node
# -------------------------
class FeedbackLoopNode:

    def __init__(self):
        self.node_name = "feedback_loop_node"

        # ---- Evolver-mutable parameters ----
        self.params = {
            "positive_gain": 0.05,
            "negative_gain": 0.08,
            "ethical_gain": 0.1,
            "value_drift_threshold": 0.5,
            "emotion_distress_threshold": -0.5,
            "confidence_floor": 0.3
        }

        # ---- State buffers ----
        self.latest = {
            "actuator": {},
            "emotion": {},
            "motivation": {},
            "value_drift": {},
            "world": {}
        }

        self.feedback_history = deque(maxlen=100)

        # ---- Metrics ----
        self.metrics = {
            "positive_feedback_rate": 0.0,
            "corrective_feedback_rate": 0.0,
            "ethical_interventions": 0.0,
            "avg_confidence": 0.0
        }

        # ---- Persistence ----
        self.db_path = "/tmp/sentience_db/feedback_log.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

        # ---- Loop ----
        self._shutdown = False
        threading.Thread(target=self._loop, daemon=True).start()

        log("INFO", "Feedback Loop Node online (deterministic).")

    # -------------------------
    def _init_db(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT,
                timestamp REAL,
                feedback_type TEXT,
                target_node TEXT,
                confidence REAL,
                payload TEXT
            )
        """)
        self.conn.commit()

    # -------------------------
    # Ingestors
    # -------------------------
    def ingest_actuator(self, data: Dict[str, Any]):
        self.latest["actuator"] = data

    def ingest_emotion(self, data: Dict[str, Any]):
        self.latest["emotion"] = data

    def ingest_motivation(self, data: Dict[str, Any]):
        self.latest["motivation"] = data

    def ingest_value_drift(self, data: Dict[str, Any]):
        self.latest["value_drift"] = data

    def ingest_world(self, data: Dict[str, Any]):
        self.latest["world"] = data

    # -------------------------
    # Core Feedback Logic
    # -------------------------
    def _generate_feedback(self) -> Optional[Dict[str, Any]]:
        action = self.latest.get("actuator", {})
        emotion = self.latest.get("emotion", {})
        motivation = self.latest.get("motivation", {})
        drift = self.latest.get("value_drift", {})

        feedback_type = None
        target_node = None
        confidence = max(self.params["confidence_floor"], action.get("confidence", 0.5))
        payload = {}

        # Positive reinforcement
        if emotion.get("sentiment", 0.0) > 0.4:
            feedback_type = "positive"
            target_node = "experience_motivation_node"
            payload["delta"] = self.params["positive_gain"]

        # Corrective feedback
        if emotion.get("sentiment", 0.0) < self.params["emotion_distress_threshold"]:
            feedback_type = "corrective"
            target_node = "cognitive_control_node"
            payload["delta"] = self.params["negative_gain"]

        # Ethical intervention
        if drift.get("drift_score", 0.0) > self.params["value_drift_threshold"]:
            feedback_type = "ethical_adjustment"
            target_node = "value_drift_monitor_node"
            payload["delta"] = self.params["ethical_gain"]

        if not feedback_type:
            return None

        return {
            "id": str(uuid4()),
            "timestamp": time.time(),
            "feedback_type": feedback_type,
            "target_node": target_node,
            "confidence": round(confidence, 3),
            "payload": payload
        }

    # -------------------------
    # Metrics
    # -------------------------
    def _update_metrics(self):
        if not self.feedback_history:
            return

        total = len(self.feedback_history)
        positives = sum(1 for f in self.feedback_history if f["feedback_type"] == "positive")
        correctives = sum(1 for f in self.feedback_history if f["feedback_type"] == "corrective")
        ethical = sum(1 for f in self.feedback_history if f["feedback_type"] == "ethical_adjustment")

        self.metrics["positive_feedback_rate"] = positives / total
        self.metrics["corrective_feedback_rate"] = correctives / total
        self.metrics["ethical_interventions"] = ethical / total
        self.metrics["avg_confidence"] = sum(f["confidence"] for f in self.feedback_history) / total

    # -------------------------
    # Evolver Interface
    # -------------------------
    def export_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "params": dict(self.params),
            "metrics": dict(self.metrics),
            "recent_feedback": list(self.feedback_history)[-5:]
        }

    def apply_mutation(self, mutation: Dict[str, float]):
        for k, v in mutation.items():
            if k in self.params:
                self.params[k] = float(v)

    # -------------------------
    # Loop
    # -------------------------
    def _loop(self):
        while not self._shutdown:
            feedback = self._generate_feedback()
            if feedback:
                self.feedback_history.append(feedback)
                self._persist(feedback)
                self._update_metrics()
                log("INFO", f"Feedback {feedback['feedback_type']} → {feedback['target_node']}")
            time.sleep(0.5)

    def _persist(self, feedback: Dict[str, Any]):
        self.cursor.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
            (
                feedback["id"],
                feedback["timestamp"],
                feedback["feedback_type"],
                feedback["target_node"],
                feedback["confidence"],
                json.dumps(feedback["payload"])
            )
        )
        self.conn.commit()

    # -------------------------
    def shutdown(self):
        self._shutdown = True
        self.conn.close()
        log("INFO", "Feedback Loop Node shutdown.")

# -------------------------
# Standalone
# -------------------------
if __name__ == "__main__":
    node = FeedbackLoopNode()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        node.shutdown()
