#!/usr/bin/env python3
"""
SurpriseDetectorNode – Sentience 5.5 compliant (UPDATED)

Role:
- Compare predictions vs actual outcomes
- Compute surprise as prediction error
- Persist surprise events
- Publish surprise signals (no interpretation, no coaching)
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import sqlite3
import argparse
import random
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List
from collections import deque
import threading

# --------------------------------------------------------------------------- #
# Optional ROS Integration                                                     #
# --------------------------------------------------------------------------- #
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

    SurpriseEvent = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    SurpriseEvent = ROSMsgFallback


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# Surprise Detector Node                                                       #
# --------------------------------------------------------------------------- #
class SurpriseDetectorNode:
    def __init__(self, db_root: str = "/tmp/sentience_db", threshold: float = 0.7, ros_enabled: bool = False):
        self.node_name = "surprise_detector_node"
        self.ros_enabled = ros_enabled and ROS_AVAILABLE
        self.threshold = float(threshold)

        # Storage
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "surprise_log.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # State
        self.pending_predictions: Deque[float] = deque(maxlen=50)
        self.pending_actuals: Deque[float] = deque(maxlen=50)
        self.surprise_history: Deque[float] = deque(maxlen=200)

        # ROS
        self.pub_surprise_event = None
        if self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_surprise_event = rospy.Publisher("/surprise_event", String, queue_size=10)
            rospy.Subscriber("/prediction_output", String, self.prediction_callback)
            rospy.Subscriber("/actual_outcome", String, self.actual_callback)
            rospy.Timer(rospy.Duration(1.0), self._tick)
        else:
            self._shutdown_flag = threading.Event()
            threading.Thread(target=self._loop, daemon=True).start()

        log("INFO", self.node_name, "SurpriseDetectorNode online")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS surprise_log (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                predicted REAL,
                actual REAL,
                surprise REAL,
                is_surprise INTEGER
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Input Callbacks                                                    #
    # ------------------------------------------------------------------ #
    def prediction_callback(self, msg: Any):
        try:
            payload = json.loads(msg.data) if hasattr(msg, "data") else msg
            predicted = float(payload.get("predicted"))
            self.pending_predictions.append(predicted)
            log("DEBUG", self.node_name, f"Prediction received: {predicted}")
        except Exception as e:
            log("ERROR", self.node_name, f"Bad prediction input: {e}")

    def actual_callback(self, msg: Any):
        try:
            payload = json.loads(msg.data) if hasattr(msg, "data") else msg
            actual = float(payload.get("actual"))
            self.pending_actuals.append(actual)
            log("DEBUG", self.node_name, f"Actual received: {actual}")
        except Exception as e:
            log("ERROR", self.node_name, f"Bad actual input: {e}")

    # ------------------------------------------------------------------ #
    # Core Logic                                                         #
    # ------------------------------------------------------------------ #
    def compute_surprise(self, predicted: float, actual: float) -> float:
        return abs(predicted - actual)

    def is_surprise(self, score: float) -> bool:
        return score >= self.threshold

    def process_pair(self, predicted: float, actual: float):
        surprise = self.compute_surprise(predicted, actual)
        flag = self.is_surprise(surprise)

        self.surprise_history.append(surprise)
        self._persist(predicted, actual, surprise, flag)
        self._publish(surprise, flag)

    # ------------------------------------------------------------------ #
    # Persistence & Publishing                                           #
    # ------------------------------------------------------------------ #
    def _persist(self, predicted: float, actual: float, surprise: float, flag: bool):
        self.conn.execute(
            "INSERT INTO surprise_log VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                time.time(),
                predicted,
                actual,
                surprise,
                int(flag),
            ),
        )
        self.conn.commit()

    def _publish(self, surprise: float, flag: bool):
        payload = {
            "timestamp": time.time(),
            "surprise": surprise,
            "is_surprise": flag,
        }

        if self.ros_enabled and self.pub_surprise_event:
            self.pub_surprise_event.publish(String(data=json.dumps(payload)))
        else:
            log("INFO", self.node_name, f"Surprise={surprise:.3f} flag={flag}")

    # ------------------------------------------------------------------ #
    # Execution                                                          #
    # ------------------------------------------------------------------ #
    def _tick(self, _=None):
        if self.pending_predictions and self.pending_actuals:
            p = self.pending_predictions.popleft()
            a = self.pending_actuals.popleft()
            self.process_pair(p, a)

    def _loop(self):
        while not self._shutdown_flag.is_set():
            if not self.pending_predictions:
                self.pending_predictions.append(random.uniform(0.0, 1.0))
            if not self.pending_actuals:
                self.pending_actuals.append(random.uniform(0.0, 1.0))
            self._tick()
            time.sleep(1.0)

    # ------------------------------------------------------------------ #
    # Summary                                                            #
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        total = len(self.surprise_history)
        surprises = sum(1 for s in self.surprise_history if self.is_surprise(s))
        return {
            "events": total,
            "surprises": surprises,
            "rate": surprises / total if total else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Shutdown                                                           #
    # ------------------------------------------------------------------ #
    def shutdown(self):
        log("INFO", self.node_name, "Shutting down")
        if hasattr(self, "_shutdown_flag"):
            self._shutdown_flag.set()
        self.conn.close()
        if self.ros_enabled:
            rospy.signal_shutdown("shutdown")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentience Surprise Detector Node")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = SurpriseDetectorNode(threshold=args.threshold, ros_enabled=args.ros_enabled)

    try:
        if args.ros_enabled and ROS_AVAILABLE:
            rospy.spin()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
