#!/usr/bin/env python3
"""
Compassion Modulator Node
Evolver-compatible, learning-enabled version
"""

import sqlite3
import os
import json
import time
import uuid
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque

import asyncio
import aiohttp
import threading

# =========================
# Optional ROS Integration
# =========================
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    pass


# =========================
# Logging Helpers
# =========================
def _log(level: str, node: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# =========================
# Compassion Modulator Node
# =========================
class CompassionModulatorNode:
    NODE_TYPE = "affective_modulation"
    LEARNING_VERSION = "1.0"

    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = "compassion_modulator_node"
        self.ros_enabled = ros_enabled

        # -------------------------
        # Core Parameters
        # -------------------------
        self.default_compassion_level = 0.5
        self.ethical_compassion_bias = 0.3
        self.llm_compassion_threshold = 0.6
        self.recent_context_window_s = 30.0
        self.update_interval = 0.5

        # -------------------------
        # Internal State
        # -------------------------
        self.compassion_level: float = self.default_compassion_level
        self.suffering_map: Dict[str, float] = {}
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=50)

        # Learning / evolution metrics
        self.modulation_history: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.cumulative_suffering_salience = 0.0
        self.last_evolution_tick = time.time()

        # -------------------------
        # Database
        # -------------------------
        db_root = "/tmp/sentience_db"
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "compassion_log.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS compassion_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                agent_id TEXT,
                suffering_score REAL,
                compassion_level REAL,
                reasoning TEXT
            )
        """)
        self.conn.commit()

        # -------------------------
        # Async Loop
        # -------------------------
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._run_async_loop, daemon=True
        )
        self._async_thread.start()

        # -------------------------
        # Execution Thread
        # -------------------------
        self._shutdown_flag = threading.Event()
        self._execution_thread = threading.Thread(
            target=self._dynamic_loop, daemon=True
        )
        self._execution_thread.start()

        _log("INFO", self.node_name, "Initialized (learning-enabled)")

    # =========================
    # Async Loop
    # =========================
    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    # =========================
    # Core Update Logic
    # =========================
    async def _update_suffering_async(
        self, agent_id: str, suffering_score: float, context: Optional[Dict[str, Any]]
    ):
        suffering_score = max(0.0, min(1.0, suffering_score))
        self.suffering_map[agent_id] = suffering_score

        self.cumulative_suffering_salience += suffering_score * 0.5
        self.cumulative_suffering_salience = min(1.0, self.cumulative_suffering_salience)

        # Simple compassionate adjustment
        avg_suffering = sum(self.suffering_map.values()) / len(self.suffering_map)
        new_compassion = min(
            1.0,
            max(self.default_compassion_level, avg_suffering * (1.2 + self.ethical_compassion_bias)),
        )

        delta = new_compassion - self.compassion_level
        self.compassion_level = new_compassion

        record = {
            "timestamp": time.time(),
            "agent_id": agent_id,
            "suffering": suffering_score,
            "compassion": self.compassion_level,
            "delta": delta,
        }
        self.modulation_history.append(record)

        self._log_to_db(agent_id, suffering_score, self.compassion_level, "adaptive update")

    def update_suffering(self, agent_id: str, suffering_score: float, context: Optional[Dict[str, Any]] = None):
        asyncio.run_coroutine_threadsafe(
            self._update_suffering_async(agent_id, suffering_score, context),
            self._async_loop,
        )

    # =========================
    # Evolver Hooks (IMPORTANT)
    # =========================
    def get_evolution_state(self) -> Dict[str, Any]:
        """Snapshot used by evolver.py"""
        return {
            "node": self.node_name,
            "type": self.NODE_TYPE,
            "version": self.LEARNING_VERSION,
            "compassion_level": self.compassion_level,
            "avg_suffering": (
                sum(self.suffering_map.values()) / len(self.suffering_map)
                if self.suffering_map else 0.0
            ),
            "history_len": len(self.modulation_history),
            "ethical_bias": self.ethical_compassion_bias,
            "timestamp": time.time(),
        }

    def apply_evolution_update(self, update: Dict[str, Any]) -> None:
        """Called by evolver.py"""
        if "ethical_compassion_bias" in update:
            self.ethical_compassion_bias = float(
                max(0.0, min(1.0, update["ethical_compassion_bias"]))
            )

        if "default_compassion_level" in update:
            self.default_compassion_level = float(
                max(0.0, min(1.0, update["default_compassion_level"]))
            )

        _log("INFO", self.node_name, f"Evolution update applied: {update}")

    # =========================
    # Persistence
    # =========================
    def _log_to_db(self, agent_id, suffering, compassion, reasoning):
        self.cursor.execute(
            """
            INSERT INTO compassion_log VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                datetime.now().isoformat(),
                agent_id,
                suffering,
                compassion,
                reasoning,
            ),
        )
        self.conn.commit()

    # =========================
    # Runtime Loop
    # =========================
    def _dynamic_loop(self):
        while not self._shutdown_flag.is_set():
            time.sleep(self.update_interval)

    # =========================
    # Shutdown
    # =========================
    def shutdown(self):
        self._shutdown_flag.set()
        self.conn.close()
        self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        _log("INFO", self.node_name, "Shutdown complete")

    def run(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = CompassionModulatorNode(ros_enabled=args.ros_enabled)
    node.run()
