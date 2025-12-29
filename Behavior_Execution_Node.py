#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import argparse
import asyncio
import aiohttp
import threading
from collections import deque
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any, Optional

# =============================
# Safety / Stability Constants
# =============================
MAX_CONFIDENCE = 1.0
MIN_CONFIDENCE = 0.0
ACTION_TIMEOUT_S = 1.0
MIN_ACTION_INTERVAL = 0.3

# =============================
# Logging helpers
# =============================
def _log(level: str, node: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)

# =============================
# Behavior Execution Node
# =============================
class BehaviorExecutionNode:
    """
    Executes actions based on upstream cognitive state.
    LLM is advisory only. Ethics & stability are enforced locally.
    """

    def __init__(self, config_file_path: Optional[str] = None):
        self.node_name = "behavior_execution_node"

        # -------------------------
        # Config
        # -------------------------
        self.update_interval = 0.5
        self.max_action_history = 50
        self.llm_endpoint = "http://localhost:8080/phi2"
        self.ethical_bias = 0.2

        self.db_path = os.path.expanduser("~/sentience_behavior_log.db")

        # -------------------------
        # Internal state
        # -------------------------
        self.last_action_ts = 0.0
        self.current_action = None

        self.latest_states: Dict[str, Dict[str, Any]] = {}
        self.action_history = deque(maxlen=self.max_action_history)
        self.log_buffer = deque(maxlen=100)

        # Evolver metrics
        self.metrics = {
            "cycles": 0,
            "llm_calls": 0,
            "ethical_vetoes": 0,
            "fallbacks": 0,
            "avg_confidence": 0.0,
        }

        # -------------------------
        # DB setup
        # -------------------------
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS behavior_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                action_type TEXT,
                target TEXT,
                parameters TEXT,
                confidence REAL,
                notes TEXT
            )
        """)
        self.conn.commit()

        # -------------------------
        # Async loop
        # -------------------------
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._run_async_loop, daemon=True
        )
        self._async_thread.start()
        self._async_session = None
        self.active_task: Optional[asyncio.Future] = None

        # -------------------------
        # Execution thread
        # -------------------------
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._execution_loop, daemon=True
        )
        self._thread.start()

        _log("INFO", self.node_name, "BehaviorExecutionNode initialized.")

    # =============================
    # Async setup
    # =============================
    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_until_complete(self._create_session())
        self._async_loop.run_forever()

    async def _create_session(self):
        self._async_session = aiohttp.ClientSession()

    async def _close_session(self):
        if self._async_session:
            await self._async_session.close()
            self._async_session = None

    # =============================
    # Public input
    # =============================
    def receive_state(self, state_type: str, data: Dict[str, Any]):
        self.latest_states[state_type] = data

    # =============================
    # Core execution
    # =============================
    def _execution_loop(self):
        while not self._shutdown.is_set():
            self.execute_cycle()
            time.sleep(self.update_interval)

    def execute_cycle(self):
        now = time.time()
        if now - self.last_action_ts < MIN_ACTION_INTERVAL:
            return

        if self.active_task and not self.active_task.done():
            return

        self.metrics["cycles"] += 1
        self.last_action_ts = now

        self.active_task = asyncio.run_coroutine_threadsafe(
            self._compute_action(), self._async_loop
        )

        try:
            result = self.active_task.result(timeout=ACTION_TIMEOUT_S)
            if result:
                self._apply_action(result)
        except Exception:
            self.metrics["fallbacks"] += 1

    # =============================
    # Action computation
    # =============================
    async def _compute_action(self):
        directive = self.latest_states.get("directive", {})
        reasoning = self.latest_states.get("reasoning", {})
        emotion = self.latest_states.get("emotion", {})
        value_drift = self.latest_states.get("value_drift", {})

        # Hard ethical gate
        if value_drift.get("drift_score", 0.0) > 0.6:
            self.metrics["ethical_vetoes"] += 1
            return None

        # LLM advisory call
        payload = {
            "directive": directive,
            "reasoning": reasoning,
            "emotion": emotion,
            "recent_actions": list(self.action_history)[-5:],
            "ethical_bias": self.ethical_bias,
        }

        try:
            async with self._async_session.post(
                self.llm_endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=0.8),
            ) as resp:
                if resp.status != 200:
                    self.metrics["fallbacks"] += 1
                    return None
                self.metrics["llm_calls"] += 1
                data = await resp.json()
        except Exception:
            self.metrics["fallbacks"] += 1
            return None

        action = data.get("action_type", "none")
        target = data.get("target", "none")
        params = data.get("parameters", {})
        confidence = float(data.get("confidence", 0.5))

        confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))

        return {
            "action": action,
            "target": target,
            "parameters": params,
            "confidence": confidence,
        }

    # =============================
    # Apply + log
    # =============================
    def _apply_action(self, action_data: Dict[str, Any]):
        action_id = str(uuid4())
        ts = datetime.utcnow().isoformat() + "Z"

        self.current_action = action_data
        self.action_history.append(action_data)

        self.metrics["avg_confidence"] = (
            sum(a["confidence"] for a in self.action_history)
            / len(self.action_history)
        )

        self.cursor.execute(
            """
            INSERT INTO behavior_log
            (id, timestamp, action_type, target, parameters, confidence, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                action_id,
                ts,
                action_data["action"],
                action_data["target"],
                json.dumps(action_data["parameters"]),
                action_data["confidence"],
                "executed",
            ),
        )
        self.conn.commit()

        _log(
            "INFO",
            self.node_name,
            f"Action={action_data['action']} "
            f"Target={action_data['target']} "
            f"Confidence={action_data['confidence']:.2f}",
        )

    # =============================
    # Evolver hook
    # =============================
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "node": self.node_name,
            "timestamp": time.time(),
            "metrics": dict(self.metrics),
        }

    # =============================
    # Shutdown
    # =============================
    def shutdown(self):
        self._shutdown.set()
        try:
            asyncio.run_coroutine_threadsafe(
                self._close_session(), self._async_loop
            ).result(2)
        except Exception:
            pass
        self.conn.close()
        _log("INFO", self.node_name, "Shutdown complete.")


# =============================
# Standalone
# =============================
if __name__ == "__main__":
    node = BehaviorExecutionNode()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.shutdown()
