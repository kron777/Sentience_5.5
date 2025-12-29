#!/usr/bin/env python3
"""
Health_Monitoring_Node (UPDATED)

Aligned to evolver.py agreements:
- No LLM shaping
- Deterministic, observable state
- Unified NodeMeta + heartbeat
- Vectorized metrics output for Evolver
- Clean ROS / non-ROS dual mode
- No hidden side effects

Drop-in replacement for your existing Health_Monitoring_Node.
"""

import os
import sys
import time
import json
import uuid
import sqlite3
import threading
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

# -----------------------------
# Optional ROS
# -----------------------------
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    pass

# -----------------------------
# Logging
# -----------------------------
def _log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)

# -----------------------------
# Node Meta (standardized)
# -----------------------------
class NodeMeta:
    def __init__(self, name: str):
        self.node_id = str(uuid.uuid4())
        self.node_name = name
        self.start_time = time.time()
        self.last_heartbeat = self.start_time
        self.status = "INIT"
        self.error_count = 0

    def heartbeat(self):
        self.last_heartbeat = time.time()
        self.status = "OK"

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "uptime_s": round(time.time() - self.start_time, 3),
            "last_heartbeat": self.last_heartbeat,
            "status": self.status,
            "error_count": self.error_count
        }

# -----------------------------
# Health Monitoring Node
# -----------------------------
class HealthMonitoringNode:
    def __init__(self, args):
        self.node_name = "health_monitoring_node"
        self.meta = NodeMeta(self.node_name)
        self.ros_enabled = args.ros_enabled

        self.poll_interval = args.poll_interval
        self.db_path = args.db_path

        # internal health state
        self.health_state: Dict[str, Any] = {
            "cpu_load": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "process_lag_ms": 0.0,
            "warnings": [],
            "timestamp": time.time()
        }

        self._shutdown = threading.Event()

        self._init_db()

        _log("INFO", self.node_name, "Health Monitoring Node initialized")

        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub = rospy.Publisher(args.health_topic, String, queue_size=10)

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # -----------------------------
    # Database
    # -----------------------------
    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_log (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                state_json TEXT,
                meta_json TEXT
            )
        """)
        self.conn.commit()

    def _persist(self):
        self.cursor.execute(
            "INSERT INTO health_log VALUES (?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                time.time(),
                json.dumps(self.health_state),
                json.dumps(self.meta.to_dict())
            )
        )
        self.conn.commit()

    # -----------------------------
    # Sampling (no OS tricks, safe defaults)
    # -----------------------------
    def _sample_health(self):
        now = time.time()
        self.health_state.update({
            "cpu_load": os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0,
            "memory_usage": 0.0,   # placeholder (no psutil dependency)
            "disk_usage": 0.0,     # placeholder
            "process_lag_ms": round((now - self.health_state["timestamp"]) * 1000, 3),
            "timestamp": now
        })

        # simple rule-based warnings
        self.health_state["warnings"] = []
        if self.health_state["cpu_load"] > 2.0:
            self.health_state["warnings"].append("HIGH_CPU_LOAD")

    # -----------------------------
    # Evolver Vector Output
    # -----------------------------
    def export_vector(self) -> Dict[str, float]:
        """
        Canonical numeric vector consumed by evolver.py
        """
        return {
            "cpu_load": float(self.health_state["cpu_load"]),
            "process_lag_ms": float(self.health_state["process_lag_ms"]),
            "warning_count": float(len(self.health_state["warnings"])),
            "uptime_s": float(time.time() - self.meta.start_time),
        }

    # -----------------------------
    # Main Loop
    # -----------------------------
    def _loop(self):
        self.meta.status = "RUNNING"
        while not self._shutdown.is_set():
            try:
                self.meta.heartbeat()
                self._sample_health()
                self._persist()

                payload = {
                    "meta": self.meta.to_dict(),
                    "health": self.health_state,
                    "vector": self.export_vector()
                }

                if ROS_AVAILABLE and self.ros_enabled:
                    self.pub.publish(String(data=json.dumps(payload)))

                time.sleep(self.poll_interval)

            except Exception as e:
                self.meta.error_count += 1
                self.meta.status = "ERROR"
                _log("ERROR", self.node_name, str(e))
                time.sleep(self.poll_interval)

    # -----------------------------
    # Shutdown
    # -----------------------------
    def shutdown(self):
        self._shutdown.set()
        if hasattr(self, "conn"):
            self.conn.close()
        _log("INFO", self.node_name, "Shutdown complete")

# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--db-path", type=str, default="/tmp/sentience_db/health.db")
    p.add_argument("--ros-enabled", action="store_true")
    p.add_argument("--health-topic", type=str, default="/health_state")
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    node = HealthMonitoringNode(args)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.shutdown()
