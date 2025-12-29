#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque
import threading
import uuid

# ---------------- ROS Compatibility ----------------
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    pass


# ---------------- Logging ----------------
def _log(level: str, node: str, msg: str):
    stream = sys.stderr if level in ("WARN", "ERROR") else sys.stdout
    print(f"[{datetime.now().isoformat()}] {node}: [{level}] {msg}", file=stream)

def _info(n, m): _log("INFO", n, m)
def _warn(n, m): _log("WARN", n, m)
def _error(n, m): _log("ERROR", n, m)
def _debug(n, m): _log("DEBUG", n, m)


# ---------------- Context Awareness Node ----------------
class ContextAwarenessNode:
    """
    Deterministic, evolvable context synthesis node.
    No LLM authority. Observable learning only.
    """

    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = "context_awareness_node"
        self.ros_enabled = ros_enabled or os.getenv("ROS_ENABLED", "false").lower() == "true"

        # ---------------- Config ----------------
        db_root = os.getenv("SENTIENCE_DB_ROOT", "/tmp/sentience_db")
        self.db_path = os.path.join(db_root, "context_awareness.db")
        os.makedirs(db_root, exist_ok=True)

        # ---------------- State ----------------
        self.context: Dict[str, Any] = {
            "environment": "neutral",
            "priority": "normal",
            "time_of_day": "unknown",
            "compassion_bias": 0.0,
        }

        self.context_history: Deque[Dict[str, Any]] = deque(maxlen=100)
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=25)

        # ---------------- Evolver Metrics ----------------
        self.evolver_metrics = {
            "cycles": 0,
            "updates_applied": 0,
            "rejected_updates": 0,
            "confidence_mean": 0.0,
            "last_update_ts": time.time(),
        }

        # ---------------- Learning State ----------------
        self.learning_state = {
            "priority_shift_rate": 0.0,
            "environment_entropy": 0.0,
        }

        # ---------------- DB ----------------
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_log (
                id TEXT PRIMARY KEY,
                ts REAL,
                environment TEXT,
                priority TEXT,
                compassion_bias REAL,
                confidence REAL
            )
        """)
        self.conn.commit()

        # ---------------- ROS ----------------
        self.pub_context = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_context = rospy.Publisher("/context_status", String, queue_size=10)

        # ---------------- Runtime ----------------
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        _info(self.node_name, "Context Awareness Node online (evolver-compliant).")

    # ---------------- Core Loop ----------------
    def _loop(self):
        while not self._shutdown.is_set():
            self._process_updates()
            self._cycle_metrics()
            time.sleep(0.5)

    # ---------------- Update Processing ----------------
    def _process_updates(self):
        if not self.pending_updates:
            return

        update = self.pending_updates.popleft()
        accepted = self._apply_update(update)

        if accepted:
            self.evolver_metrics["updates_applied"] += 1
            self.context_history.append(self.context.copy())
            self._log_context(confidence=accepted)
        else:
            self.evolver_metrics["rejected_updates"] += 1

    def _apply_update(self, update: Dict[str, Any]) -> float:
        """
        Deterministic rule-based update.
        Returns confidence score if accepted, else 0.0
        """
        confidence = 0.0

        env = update.get("environment")
        priority = update.get("priority")
        compassion_delta = update.get("compassion_delta", 0.0)

        if env:
            self.context["environment"] = env
            confidence += 0.4

        if priority:
            self.context["priority"] = priority
            confidence += 0.4

        if compassion_delta:
            self.context["compassion_bias"] = min(
                1.0, max(0.0, self.context["compassion_bias"] + compassion_delta)
            )
            confidence += 0.2

        self.context["time_of_day"] = datetime.now().strftime("%H:%M:%S")

        return confidence if confidence >= 0.4 else 0.0

    # ---------------- Metrics ----------------
    def _cycle_metrics(self):
        self.evolver_metrics["cycles"] += 1
        self.evolver_metrics["last_update_ts"] = time.time()

        if self.context_history:
            priorities = [c["priority"] for c in self.context_history]
            self.learning_state["priority_shift_rate"] = len(set(priorities)) / max(1, len(priorities))

        self.evolver_metrics["confidence_mean"] = (
            self.evolver_metrics["updates_applied"]
            / max(1, self.evolver_metrics["cycles"])
        )

    # ---------------- Logging ----------------
    def _log_context(self, confidence: float):
        try:
            self.cursor.execute(
                "INSERT INTO context_log VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    time.time(),
                    self.context["environment"],
                    self.context["priority"],
                    self.context["compassion_bias"],
                    confidence,
                ),
            )
            self.conn.commit()
        except Exception as e:
            _error(self.node_name, f"DB log failed: {e}")

        if self.pub_context:
            self.pub_context.publish(String(data=json.dumps(self.context)))

    # ---------------- External Interface ----------------
    def submit_update(self, update: Dict[str, Any]):
        """Safe external entrypoint"""
        if isinstance(update, dict):
            self.pending_updates.append(update)

    def get_context(self) -> Dict[str, Any]:
        return self.context.copy()

    def get_learning_metrics(self) -> Dict[str, Any]:
        return {
            "metrics": self.evolver_metrics.copy(),
            "learning": self.learning_state.copy(),
        }

    def apply_adjustment(self, delta: Dict[str, float]):
        """Used by evolver.py"""
        if "compassion_bias" in delta:
            self.context["compassion_bias"] = min(
                1.0, max(0.0, self.context["compassion_bias"] + delta["compassion_bias"])
            )

    def export_evolution_snapshot(self) -> Dict[str, Any]:
        return {
            "context": self.context.copy(),
            "metrics": self.evolver_metrics.copy(),
            "learning": self.learning_state.copy(),
        }

    # ---------------- Shutdown ----------------
    def shutdown(self):
        _info(self.node_name, "Shutting down.")
        self._shutdown.set()
        self.conn.close()

    def run(self):
        try:
            if ROS_AVAILABLE and self.ros_enabled:
                rospy.spin()
            else:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()


# ---------------- Entry ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = ContextAwarenessNode(ros_enabled=args.ros_enabled)
    node.run()
