#!/usr/bin/env python3
"""
SimulationNode – Sentience 5.5 compliant

Role:
- Generate hypothetical / simulated world signals
- NO interpretation
- NO prediction
- NO decision-making
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import sqlite3
import random
import argparse
import threading
from datetime import datetime
from typing import Dict, Any, Deque
from collections import deque

# ---------------- ROS (optional) ---------------- #
ROS_AVAILABLE = False
try:
    import rospy
    from std_msgs.msg import String
    from sensor_msgs.msg import Range
    ROS_AVAILABLE = True
except Exception:
    rospy = None
    String = None
    Range = None

# ---------------- Logging ---------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)

# ---------------- Simulation Node ---------------- #
class SimulationNode:
    def __init__(self, db_root: str = "/tmp/sentience_db", rate_hz: float = 5.0):
        self.node_name = "simulation_node"
        self.rate_hz = max(rate_hz, 0.1)
        self.interval = 1.0 / self.rate_hz

        # DB
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "simulation.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # Internal queues
        self.sim_queue: Deque[Dict[str, Any]] = deque(maxlen=32)

        # ROS
        self.ros_enabled = ROS_AVAILABLE and os.getenv("ROS_ENABLED", "false").lower() == "true"
        if self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_range = rospy.Publisher("/simulation/proximity", Range, queue_size=10)
            self.pub_state = rospy.Publisher("/simulation/state", String, queue_size=10)

        self._shutdown = threading.Event()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

        log("INFO", self.node_name, "SimulationNode initialized")

    # ---------------- DB ---------------- #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                signal_type TEXT,
                payload_json TEXT
            )
        """)
        self.conn.commit()

    def _persist(self, signal_type: str, payload: Dict[str, Any]):
        self.conn.execute(
            "INSERT INTO simulations VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), time.time(), signal_type, json.dumps(payload))
        )
        self.conn.commit()

    # ---------------- Simulation ---------------- #
    def simulate_proximity(self) -> Dict[str, Any]:
        return {
            "range": round(random.uniform(0.2, 4.0), 3),
            "min": 0.2,
            "max": 4.0,
            "fov": 0.5,
            "radiation": "INFRARED"
        }

    def simulate_state(self) -> str:
        return random.choice(["idle", "processing", "exploring", "charging"])

    def generate(self):
        self.sim_queue.append({
            "proximity": self.simulate_proximity(),
            "state": self.simulate_state()
        })

    # ---------------- Publish ---------------- #
    def publish(self, sim: Dict[str, Any]):
        # Persist
        self._persist("proximity", sim["proximity"])
        self._persist("state", {"state": sim["state"]})

        if self.ros_enabled:
            r = Range()
            r.header.stamp = rospy.Time.now()
            r.min_range = sim["proximity"]["min"]
            r.max_range = sim["proximity"]["max"]
            r.field_of_view = sim["proximity"]["fov"]
            r.range = sim["proximity"]["range"]
            self.pub_range.publish(r)
            self.pub_state.publish(String(data=sim["state"]))
        else:
            log("INFO", self.node_name,
                f"Simulated → range={sim['proximity']['range']}m | state={sim['state']}")

    # ---------------- Main Loop ---------------- #
    def _loop(self):
        while not self._shutdown.is_set():
            self.generate()
            while self.sim_queue:
                sim = self.sim_queue.popleft()
                self.publish(sim)
            time.sleep(self.interval)

    # ---------------- Shutdown ---------------- #
    def shutdown(self):
        log("INFO", self.node_name, "Shutting down")
        self._shutdown.set()
        if self.conn:
            self.conn.close()
        if self.ros_enabled:
            rospy.signal_shutdown("SimulationNode shutdown")

# ---------------- CLI ---------------- #
def main():
    parser = argparse.ArgumentParser("Sentience Simulation Node")
    parser.add_argument("--rate", type=float, default=5.0)
    args = parser.parse_args()

    node = SimulationNode(rate_hz=args.rate)
    try:
        if node.ros_enabled:
            rospy.spin()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()

if __name__ == "__main__":
    main()
