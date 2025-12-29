#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque

import threading
from collections import deque

# ---------------- ROS OPTIONAL ----------------
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True

    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    MotivationState = ROSMsgFallback
    MemoryFeedback = ROSMsgFallback
    BatteryState = ROSMsgFallback
    SurpriseDetector = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    MotivationState = ROSMsgFallback
    MemoryFeedback = ROSMsgFallback
    BatteryState = ROSMsgFallback
    SurpriseDetector = ROSMsgFallback


# ---------------- LOGGING ----------------
def _log(level: str, node: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# ---------------- NODE ----------------
class DriveSystemNode:
    """
    PURPOSE:
    - Maintain internal motivational pressures (drives)
    - Decay, amplify, and bound drives deterministically
    - Export gradients for Evolver
    """

    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = "drive_system_node"
        self.ros_enabled = ros_enabled

        # --- Parameters ---
        self.decay_rate = 0.02
        self.drive_threshold = 0.5
        self.ethical_compassion_bias = 0.2

        # --- Drives ---
        self.drives: Dict[str, float] = {
            "curiosity": 0.4,
            "energy": 0.5,
            "social_contact": 0.3,
            "safety": 0.6,
        }

        # --- State ---
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=20)
        self.drive_history: Deque[Dict[str, float]] = deque(maxlen=100)
        self.sensory_data: Dict[str, Any] = {}

        # --- Metrics (observed, not trained here) ---
        self.metrics = {
            "updates": 0,
            "threshold_crossings": 0,
            "dominant_switches": 0,
        }
        self._last_dominant: Optional[str] = None

        # --- Database ---
        db_root = "/tmp/sentience_db"
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "drive_log.db")

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS drive_log (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                dominant_drive TEXT,
                drive_snapshot_json TEXT
            )
        """)
        self.conn.commit()

        _log("INFO", self.node_name, "Drive System Node online")

        # --- ROS / Dynamic ---
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_motivation = rospy.Publisher("/motivation_state", String, queue_size=10)

            rospy.Subscriber("/memory_feedback", String, self._memory_callback)
            rospy.Subscriber("/battery_state", String, self._battery_callback)
            rospy.Subscriber("/surprise_detector", String, self._surprise_callback)

            rospy.Timer(rospy.Duration(4.0), self._update_loop)
        else:
            self._shutdown_flag = threading.Event()
            self._thread = threading.Thread(target=self._dynamic_loop, daemon=True)
            self._thread.start()

    # ---------------- INPUT HANDLERS ----------------
    def _memory_callback(self, msg: Any):
        try:
            data = json.loads(msg.data) if hasattr(msg, "data") else msg
            self.pending_updates.append({"type": "memory", "data": data})
        except Exception:
            pass

    def _battery_callback(self, msg: Any):
        try:
            data = json.loads(msg.data) if hasattr(msg, "data") else msg
            self.pending_updates.append({"type": "battery", "data": data})
        except Exception:
            pass

    def _surprise_callback(self, msg: Any):
        try:
            data = json.loads(msg.data) if hasattr(msg, "data") else msg
            self.pending_updates.append({"type": "surprise", "data": data})
        except Exception:
            pass

    # ---------------- UPDATE LOOP ----------------
    def _dynamic_loop(self):
        while not self._shutdown_flag.is_set():
            self._update_loop()
            time.sleep(4)

    def _update_loop(self, event=None):
        self._apply_pending_updates()
        self._decay_drives()
        self._evaluate_state()
        self._log_state()
        self._publish_state()

    # ---------------- CORE LOGIC ----------------
    def _apply_pending_updates(self):
        while self.pending_updates:
            upd = self.pending_updates.popleft()
            t = upd["type"]
            d = upd["data"]

            if t == "memory":
                novelty = float(d.get("novelty_score", 0.5))
                self._adjust("curiosity", (0.5 - novelty) * 0.2)

            elif t == "battery":
                level = float(d.get("battery_level", 1.0))
                self.drives["energy"] = max(0.0, min(1.0, 1.0 - level))
                if level < 0.2:
                    self._adjust("safety", self.ethical_compassion_bias)

            elif t == "surprise":
                intensity = float(d.get("surprise_intensity", 0.0))
                self._adjust("safety", intensity * 0.2)
                if intensity > 0.5:
                    self._adjust("social_contact", self.ethical_compassion_bias * 0.5)

            self.metrics["updates"] += 1

    def _adjust(self, drive: str, delta: float):
        self.drives[drive] = max(0.0, min(1.0, self.drives[drive] + delta))

    def _decay_drives(self):
        for k in self.drives:
            self.drives[k] = max(0.0, self.drives[k] - self.decay_rate)

    def _evaluate_state(self):
        dominant = max(self.drives, key=self.drives.get)
        if dominant != self._last_dominant:
            self.metrics["dominant_switches"] += 1
            self._last_dominant = dominant

        if self.drives[dominant] > self.drive_threshold:
            self.metrics["threshold_crossings"] += 1

        self.drive_history.append(self.drives.copy())

    # ---------------- OUTPUT ----------------
    def _publish_state(self):
        state = {
            "timestamp": time.time(),
            "dominant_drive": max(self.drives, key=self.drives.get),
            "drives": self.drives,
        }

        if ROS_AVAILABLE and self.ros_enabled:
            self.pub_motivation.publish(String(data=json.dumps(state)))
        else:
            _log("INFO", self.node_name, f"Motivation state: {state}")

    def _log_state(self):
        try:
            self.cursor.execute(
                "INSERT INTO drive_log VALUES (?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    time.time(),
                    max(self.drives, key=self.drives.get),
                    json.dumps(self.drives),
                ),
            )
            self.conn.commit()
        except Exception as e:
            _log("ERROR", self.node_name, f"DB log failed: {e}")

    # ---------------- EVOLVER INTERFACE ----------------
    def export_learning_state(self) -> Dict[str, Any]:
        return {
            "node": self.node_name,
            "drives": self.drives,
            "metrics": self.metrics,
            "history_tail": list(self.drive_history)[-10:],
        }

    # ---------------- SHUTDOWN ----------------
    def shutdown(self):
        _log("INFO", self.node_name, "Shutting down")
        if hasattr(self, "_shutdown_flag"):
            self._shutdown_flag.set()
        self.conn.close()

    def run(self):
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.spin()
        else:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.shutdown()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = DriveSystemNode(ros_enabled=args.ros_enabled)
    node.run()
