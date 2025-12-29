#!/usr/bin/env python3
"""
ResourceAllocationNode – Sentience 5.5 compliant

Role:
- Consume health, prediction, optimization inputs
- Produce deterministic resource allocation
- No interpretation, no narrative
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import sqlite3
import argparse
import threading
import random
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque

# --------------------------------------------------------------------------- #
# ROS Optional                                                                #
# --------------------------------------------------------------------------- #
ROS_AVAILABLE = False
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    rospy = None
    String = None

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)

# --------------------------------------------------------------------------- #
# Resource Allocation Node                                                    #
# --------------------------------------------------------------------------- #
class ResourceAllocationNode:
    def __init__(self, db_root: str = "/tmp/sentience_db", ros_enabled: bool = False):
        self.node_name = "resource_allocation_node"
        self.ros_enabled = ros_enabled and ROS_AVAILABLE

        # --- State ---
        self.allocation = {"cpu": 0.5, "memory": 0.5}
        self.last_inputs: Dict[str, Any] = {}
        self.ethical_compassion_bias = 0.2  # bounded modifier only

        # --- DB ---
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "resource_allocation.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # --- ROS ---
        if self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub = rospy.Publisher("/resource_allocation", String, queue_size=10)
            rospy.Subscriber("/health_status", String, self.health_cb)
            rospy.Subscriber("/prediction_output", String, self.prediction_cb)
            rospy.Subscriber("/optimization_suggestions", String, self.optimization_cb)
            rospy.Timer(rospy.Duration(2.0), self.tick)
        else:
            self._shutdown = threading.Event()
            threading.Thread(target=self._loop, daemon=True).start()

        log("INFO", self.node_name, "Node online")

    # ------------------------------------------------------------------ #
    # DB                                                                  #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS allocation_log (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                cpu REAL,
                memory REAL,
                input_json TEXT
            )
        """)
        self.conn.commit()

    def _persist(self):
        self.conn.execute(
            "INSERT INTO allocation_log VALUES (?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                time.time(),
                self.allocation["cpu"],
                self.allocation["memory"],
                json.dumps(self.last_inputs),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Callbacks                                                           #
    # ------------------------------------------------------------------ #
    def health_cb(self, msg: Any):
        data = json.loads(msg.data) if hasattr(msg, "data") else msg
        self.last_inputs["health"] = data

    def prediction_cb(self, msg: Any):
        data = json.loads(msg.data) if hasattr(msg, "data") else msg
        self.last_inputs["prediction"] = data

    def optimization_cb(self, msg: Any):
        data = json.loads(msg.data) if hasattr(msg, "data") else msg
        self.last_inputs["optimization"] = data

    # ------------------------------------------------------------------ #
    # Core Logic                                                          #
    # ------------------------------------------------------------------ #
    def compute_allocation(self):
        cpu = 0.5
        mem = 0.5

        health = self.last_inputs.get("health", {})
        pred = self.last_inputs.get("prediction", {})
        opt = self.last_inputs.get("optimization", {})

        if health.get("cpu_usage", 0) > 80:
            cpu += 0.15

        if pred.get("predicted_cpu_usage", 0) > 85:
            cpu += 0.15

        if opt.get("action") == "reallocate_resources":
            cpu += 0.1

        # Compassion bias = damping, not storytelling
        cpu *= (1.0 - self.ethical_compassion_bias * 0.1)

        # Normalize
        cpu = min(max(cpu, 0.1), 0.9)
        mem = 1.0 - cpu

        self.allocation = {"cpu": round(cpu, 3), "memory": round(mem, 3)}

    # ------------------------------------------------------------------ #
    # Tick / Publish                                                      #
    # ------------------------------------------------------------------ #
    def tick(self, *_):
        self.compute_allocation()
        self._persist()
        self.publish()

    def publish(self):
        payload = json.dumps(self.allocation)
        if self.ros_enabled:
            self.pub.publish(String(data=payload))
        else:
            log("INFO", self.node_name, f"allocation={payload}")

    # ------------------------------------------------------------------ #
    # Non-ROS Loop                                                        #
    # ------------------------------------------------------------------ #
    def _loop(self):
        while not self._shutdown.is_set():
            # simulated inputs
            self.last_inputs["health"] = {"cpu_usage": random.randint(30, 95)}
            self.last_inputs["prediction"] = {"predicted_cpu_usage": random.randint(40, 95)}
            self.last_inputs["optimization"] = {"action": random.choice(["none", "reallocate_resources"])}
            self.tick()
            time.sleep(2)

    # ------------------------------------------------------------------ #
    # Shutdown                                                            #
    # ------------------------------------------------------------------ #
    def shutdown(self):
        log("INFO", self.node_name, "Shutdown")
        if hasattr(self, "_shutdown"):
            self._shutdown.set()
        self.conn.close()
        if self.ros_enabled:
            rospy.signal_shutdown("shutdown")


# --------------------------------------------------------------------------- #
# Entry                                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = ResourceAllocationNode(ros_enabled=args.ros_enabled)
    try:
        if args.ros_enabled:
            rospy.spin()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
