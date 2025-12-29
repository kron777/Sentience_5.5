#!/usr/bin/env python3
"""
SimulatedThinkingNode – UPDATED / STABILIZED

Purpose:
- Deterministic internal simulation engine
- No unsafe eval
- Clear prediction-style outputs
- Compatible with upstream decision / planning nodes
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import random
import sqlite3
import argparse
from datetime import datetime
from typing import Callable, Dict, Any, Optional, Deque, List

import threading
from collections import deque

# ---------------------------------------------------------------------
# ROS (optional)
# ---------------------------------------------------------------------
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

    SimulationResults = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    SimulationResults = ROSMsgFallback


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# ---------------------------------------------------------------------
# Simulated Thinking Node
# ---------------------------------------------------------------------
class SimulatedThinkingNode:
    """
    Internal simulation / prediction node.

    Responsibilities:
    - Run hypothetical scenarios
    - Produce outcome predictions
    - No sensory interpretation
    - No policy decisions
    """

    def __init__(self, db_root: str = "/tmp/sentience_db", ros_enabled: bool = False):
        self.node_name = "simulated_thinking_node"
        self.ros_enabled = ros_enabled

        # -----------------------------------------------------------------
        # DB
        # -----------------------------------------------------------------
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "simulated_thinking.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # -----------------------------------------------------------------
        # State
        # -----------------------------------------------------------------
        self.scenarios: List[Callable[[], Dict[str, Any]]] = []
        self.pending_actions: Deque[Dict[str, Any]] = deque(maxlen=32)
        self.simulation_history: Deque[Dict[str, Any]] = deque(maxlen=128)

        self.last_run_ts = 0.0
        self.run_interval = 5.0

        # -----------------------------------------------------------------
        # ROS
        # -----------------------------------------------------------------
        self.pub_results = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_results = rospy.Publisher(
                "/simulation_results", SimulationResults, queue_size=10
            )
            rospy.Timer(rospy.Duration(self.run_interval), self._ros_timer)

        else:
            self._shutdown_flag = threading.Event()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        log("INFO", self.node_name, "Initialized")

    # -----------------------------------------------------------------
    # DB
    # -----------------------------------------------------------------
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                scenario_name TEXT,
                outcome_json TEXT
            )
        """)
        self.conn.commit()

    # -----------------------------------------------------------------
    # Scenario API
    # -----------------------------------------------------------------
    def add_scenario(self, name: str, fn: Callable[[], Dict[str, Any]]):
        """
        Register a scenario function.

        fn MUST return a dict:
        {
            "prediction": str,
            "confidence": float,
            "notes": str
        }
        """
        self.scenarios.append(fn)
        log("INFO", self.node_name, f"Scenario added: {name}")

    # -----------------------------------------------------------------
    # Core simulation
    # -----------------------------------------------------------------
    def run_simulations(self) -> Dict[str, Any]:
        results = {}
        now = time.time()

        for idx, scenario in enumerate(self.scenarios):
            sid = f"scenario_{idx}"
            try:
                outcome = scenario()
                results[sid] = outcome
                self._persist(sid, outcome)
            except Exception as e:
                results[sid] = {
                    "prediction": "error",
                    "confidence": 0.0,
                    "notes": str(e)
                }

        self.simulation_history.append({
            "timestamp": now,
            "results": results
        })

        self.publish(results)
        return results

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def _persist(self, name: str, outcome: Dict[str, Any]):
        self.conn.execute(
            """
            INSERT INTO simulations
            (id, timestamp, scenario_name, outcome_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                time.time(),
                name,
                json.dumps(outcome),
            ),
        )
        self.conn.commit()

    # -----------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------
    def publish(self, results: Dict[str, Any]):
        payload = {
            "timestamp": time.time(),
            "node": self.node_name,
            "results": results,
        }

        if ROS_AVAILABLE and self.ros_enabled and self.pub_results:
            self.pub_results.publish(String(data=json.dumps(payload)))
        else:
            log("INFO", self.node_name, f"Results: {json.dumps(payload)}")

    # -----------------------------------------------------------------
    # Looping
    # -----------------------------------------------------------------
    def _loop(self):
        while not self._shutdown_flag.is_set():
            now = time.time()
            if now - self.last_run_ts >= self.run_interval:
                self.last_run_ts = now
                self.run_simulations()
            time.sleep(0.2)

    def _ros_timer(self, _):
        self.run_simulations()

    # -----------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------
    def shutdown(self):
        log("INFO", self.node_name, "Shutdown")
        if hasattr(self, "_shutdown_flag"):
            self._shutdown_flag.set()
        self.conn.close()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Shutdown requested")

    def run(self):
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.spin()
        else:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        self.shutdown()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = SimulatedThinkingNode(ros_enabled=args.ros_enabled)

    # Example scenarios
    node.add_scenario(
        "resource_pressure",
        lambda: {
            "prediction": "system_load_increase",
            "confidence": round(random.uniform(0.6, 0.9), 2),
            "notes": "CPU + memory trend projection"
        },
    )

    node.add_scenario(
        "decision_outcome",
        lambda: {
            "prediction": "action_success",
            "confidence": 0.82,
            "notes": "Based on historical patterns"
        },
    )

    node.run()
