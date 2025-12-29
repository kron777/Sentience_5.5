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


# ---------------- Control Node ----------------
class ControlNode:
    """
    Deterministic action arbitration node.
    Evolver-compatible. Learning via metrics only.
    """

    NODE_TYPE = "control"
    LEARNING_VERSION = "1.0"

    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = "control_node"
        self.ros_enabled = ros_enabled or os.getenv("ROS_ENABLED", "false").lower() == "true"

        # ---------------- Config ----------------
        db_root = os.getenv("SENTIENCE_DB_ROOT", "/tmp/sentience_db")
        self.db_path = os.path.join(db_root, "control_log.db")
        os.makedirs(db_root, exist_ok=True)

        self.ethical_compassion_bias = 0.2

        # ---------------- State ----------------
        self.current_action: Dict[str, Any] = {
            "action": "idle",
            "priority": "low",
            "confidence": 0.5,
        }

        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=25)
        self.control_history: Deque[Dict[str, Any]] = deque(maxlen=100)

        # ---------------- Learning Metrics ----------------
        self.metrics = {
            "cycles": 0,
            "actions_executed": 0,
            "priority_changes": 0,
            "adaptations_applied": 0,
            "last_update_ts": time.time(),
        }

        # ---------------- DB ----------------
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS control_log (
                id TEXT PRIMARY KEY,
                ts REAL,
                action TEXT,
                priority TEXT,
                confidence REAL
            )
        """)
        self.conn.commit()

        # ---------------- ROS ----------------
        self.pub_control = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_control = rospy.Publisher("/control_output", String, queue_size=10)

        # ---------------- Runtime ----------------
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        _info(self.node_name, "Control Node online (evolver-compliant).")

    # ---------------- Core Loop ----------------
    def _loop(self):
        while not self._shutdown.is_set():
            self._process_updates()
            self.metrics["cycles"] += 1
            self.metrics["last_update_ts"] = time.time()
            time.sleep(0.5)

    # ---------------- Update Processing ----------------
    def _process_updates(self):
        if not self.pending_updates:
            return

        update = self.pending_updates.popleft()
        source = update.get("source")
        data = update.get("data", {})

        if source == "integration":
            self._apply_integration(data)
        elif source == "motivation":
            self._apply_motivation(data)
        elif source == "adaptation":
            self._apply_adaptation(data)

        self.control_history.append(self.current_action.copy())

    # ---------------- Deterministic Rules ----------------
    def _apply_integration(self, data: Dict[str, Any]):
        action = data.get("final_action", {}).get("action", "idle")
        self.current_action["action"] = action
        self.current_action["confidence"] = 0.6
        self.metrics["actions_executed"] += 1
        self._emit()

    def _apply_motivation(self, data: Dict[str, Any]):
        level = float(data.get("motivation_level", 0.5))

        prev = self.current_action["priority"]

        if level < 0.3:
            self.current_action["priority"] = "low"
        elif level > 0.7:
            self.current_action["priority"] = "high"
        else:
            self.current_action["priority"] = "medium"

        if prev != self.current_action["priority"]:
            self.metrics["priority_changes"] += 1

        self.current_action["confidence"] = min(1.0, 0.4 + level * 0.6)
        self._emit()

    def _apply_adaptation(self, data: Dict[str, Any]):
        strategy = data.get("strategy", "balanced")

        prev = self.current_action["priority"]

        if strategy == "conservative":
            self.current_action["priority"] = "low"
        elif strategy == "optimized":
            self.current_action["priority"] = "medium"
        else:
            self.current_action["priority"] = "medium"

        if prev != self.current_action["priority"]:
            self.metrics["priority_changes"] += 1

        self.metrics["adaptations_applied"] += 1
        self.current_action["confidence"] = 0.55
        self._emit()

    # ---------------- Output ----------------
    def _emit(self):
        payload = json.dumps(self.current_action)

        if self.pub_control:
            self.pub_control.publish(String(data=payload))
        else:
            _info(self.node_name, f"Control output: {payload}")

        self._log_to_db()

    def _log_to_db(self):
        try:
            self.cursor.execute(
                "INSERT INTO control_log VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    time.time(),
                    self.current_action["action"],
                    self.current_action["priority"],
                    self.current_action["confidence"],
                ),
            )
            self.conn.commit()
        except Exception as e:
            _error(self.node_name, f"DB log failed: {e}")

    # ---------------- Evolver Hooks ----------------
    def get_evolution_state(self) -> Dict[str, Any]:
        return {
            "node": self.node_name,
            "type": self.NODE_TYPE,
            "version": self.LEARNING_VERSION,
            "current_action": self.current_action.copy(),
            "metrics": self.metrics.copy(),
            "history_len": len(self.control_history),
            "timestamp": time.time(),
        }

    def apply_evolution_update(self, update: Dict[str, Any]):
        if "ethical_compassion_bias" in update:
            self.ethical_compassion_bias = max(
                0.0, min(1.0, float(update["ethical_compassion_bias"]))
            )
        _info(self.node_name, f"Evolution update applied: {update}")

    # ---------------- External API ----------------
    def update_from_source(self, source: str, data: Dict[str, Any]):
        self.pending_updates.append({"source": source, "data": data})

    def get_current_action(self) -> Dict[str, Any]:
        return self.current_action.copy()

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

    node = ControlNode(ros_enabled=args.ros_enabled)
    node.run()
