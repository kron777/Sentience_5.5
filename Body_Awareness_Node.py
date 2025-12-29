#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import uuid
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List

import asyncio
import aiohttp
import threading
from collections import deque

# -------------------- Safety / Stability Constants --------------------
MAX_SALIENCE_DELTA = 0.2
SALience_DECAY = 0.05
MIN_ANALYSIS_INTERVAL = 0.2

# -------------------- ROS Compatibility --------------------
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    pass


# -------------------- Logging --------------------
def _log(level: str, node: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node}: [{level}] {msg}", file=sys.stdout)


# -------------------- Node --------------------
class BodyAwarenessNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = "body_awareness_node"
        self.ros_enabled = ros_enabled

        self.analysis_interval = 0.2
        self.llm_threshold = 0.5
        self.context_window_s = 5.0

        self.last_analysis_ts = 0.0
        self.cumulative_salience = 0.0

        # -------------------- Evolver Metrics --------------------
        self.metrics = {
            "cycles": 0,
            "llm_calls": 0,
            "fallback_used": 0,
            "avg_salience": 0.0,
            "anomalies_detected": 0
        }

        # -------------------- History --------------------
        self.joint_states: Deque[Dict[str, Any]] = deque(maxlen=20)
        self.force_states: Deque[Dict[str, Any]] = deque(maxlen=20)
        self.tactile_states: Deque[Dict[str, Any]] = deque(maxlen=20)
        self.health_states: Deque[Dict[str, Any]] = deque(maxlen=10)

        # -------------------- Async --------------------
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_task: Optional[asyncio.Future] = None

        # -------------------- DB --------------------
        self.db_path = "/tmp/sentience_db/body_awareness_log.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS awareness_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                state TEXT,
                severity REAL,
                notes TEXT
            )
        """)
        self.conn.commit()

        # -------------------- ROS / Dynamic --------------------
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            rospy.Timer(rospy.Duration(self.analysis_interval), self._analysis_wrapper)
        else:
            self.shutdown_flag = threading.Event()
            threading.Thread(target=self._dynamic_loop, daemon=True).start()

        _log("INFO", self.node_name, "Body awareness node online.")

    # -------------------- Async Loop --------------------
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # -------------------- Dynamic Loop --------------------
    def _dynamic_loop(self):
        while not self.shutdown_flag.is_set():
            self._analysis_wrapper(None)
            time.sleep(self.analysis_interval)

    # -------------------- Salience --------------------
    def _add_salience(self, delta: float):
        delta = max(-MAX_SALIENCE_DELTA, min(MAX_SALIENCE_DELTA, delta))
        self.cumulative_salience = min(1.0, max(0.0, self.cumulative_salience + delta))

    def _decay_salience(self):
        self.cumulative_salience = max(0.0, self.cumulative_salience - SALience_DECAY)

    # -------------------- Analysis Wrapper --------------------
    def _analysis_wrapper(self, event: Any):
        now = time.time()
        if now - self.last_analysis_ts < MIN_ANALYSIS_INTERVAL:
            return
        self.last_analysis_ts = now

        if self.active_task and not self.active_task.done():
            return

        self.active_task = asyncio.run_coroutine_threadsafe(
            self._analyze_async(), self.loop
        )

    # -------------------- Core Analysis --------------------
    async def _analyze_async(self):
        self.metrics["cycles"] += 1
        self._decay_salience()

        sal = self.cumulative_salience
        self.metrics["avg_salience"] = (
            (self.metrics["avg_salience"] * (self.metrics["cycles"] - 1)) + sal
        ) / self.metrics["cycles"]

        anomaly = sal > self.llm_threshold
        severity = min(1.0, sal)

        if anomaly:
            self.metrics["anomalies_detected"] += 1

        # LLM only if truly needed
        if anomaly and sal > 0.7:
            self.metrics["llm_calls"] += 1
            notes = "High salience physical anomaly detected."
        else:
            self.metrics["fallback_used"] += 1
            notes = "Normal physical state."

        self._store_state(
            "anomaly" if anomaly else "normal",
            severity,
            notes
        )

        self.cumulative_salience = 0.0

    # -------------------- Storage --------------------
    def _store_state(self, state: str, severity: float, notes: str):
        self.cur.execute(
            "INSERT INTO awareness_log VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                datetime.now().isoformat(),
                state,
                severity,
                notes
            )
        )
        self.conn.commit()

    # -------------------- Sensor Inputs --------------------
    def joint_state_callback(self, data: Dict[str, Any]):
        self.joint_states.append(data)
        self._add_salience(0.1)

    def force_callback(self, data: Dict[str, Any]):
        self.force_states.append(data)
        self._add_salience(0.2)

    def tactile_callback(self, data: Dict[str, Any]):
        self.tactile_states.append(data)
        self._add_salience(0.15)

    def health_callback(self, data: Dict[str, Any]):
        self.health_states.append(data)
        self._add_salience(0.3)

    # -------------------- Shutdown --------------------
    def shutdown(self):
        if hasattr(self, "shutdown_flag"):
            self.shutdown_flag.set()
        self.conn.close()
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        _log("INFO", self.node_name, "Shutdown complete.")


# -------------------- Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = BodyAwarenessNode(ros_enabled=args.ros_enabled)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.shutdown()
