#!/usr/bin/env python3
"""
World_Model_Node – Sentience 5.5 compliant (UPDATED)

Role:
- Maintain a coherent, current representation of reality
- Integrate perception, memory, attention, and prediction
- NO prediction inference or judgment
- NO motivational or emotional bias
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import sqlite3
import asyncio
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# World Model Node                                                             #
# --------------------------------------------------------------------------- #
class WorldModelNode:
    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "world_model_node"
        self.db_path = os.path.join(db_root, "world_model.db")
        os.makedirs(db_root, exist_ok=True)

        # --- Internal State ---
        self.recent_context_window_s = 30.0
        self.sensory_data: Dict[str, Any] = {}

        self.recent_sensory_qualia: Deque[Dict[str, Any]] = deque(maxlen=50)
        self.recent_memory_responses: Deque[Dict[str, Any]] = deque(maxlen=20)
        self.recent_attention_states: Deque[Dict[str, Any]] = deque(maxlen=10)
        self.recent_prediction_states: Deque[Dict[str, Any]] = deque(maxlen=10)
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=10)

        # --- Async setup ---
        self._shutdown_flag = threading.Event()
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._run_async_loop, daemon=True
        )
        self._async_thread.start()
        self.active_task: Optional[asyncio.Future] = None

        # --- Database ---
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        log("INFO", self.node_name, "WorldModelNode initialized")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS world_states (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                state_json TEXT
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Async loop                                                         #
    # ------------------------------------------------------------------ #
    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    def _shutdown_async_loop(self):
        if self._async_loop.is_running():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)

    # ------------------------------------------------------------------ #
    # History management                                                 #
    # ------------------------------------------------------------------ #
    def _prune_history(self):
        now = time.time()
        for dq in [
            self.recent_sensory_qualia,
            self.recent_memory_responses,
            self.recent_attention_states,
            self.recent_prediction_states,
            self.recent_cognitive_directives,
        ]:
            while dq and now - float(dq[0].get("timestamp", now)) > self.recent_context_window_s:
                dq.popleft()

    # ------------------------------------------------------------------ #
    # Input callbacks                                                    #
    # ------------------------------------------------------------------ #
    def ingest_sensory_qualia(self, data: Dict[str, Any]):
        self.recent_sensory_qualia.append(data)

    def ingest_memory_response(self, data: Dict[str, Any]):
        self.recent_memory_responses.append(data)

    def ingest_attention_state(self, data: Dict[str, Any]):
        self.recent_attention_states.append(data)

    def ingest_prediction_state(self, data: Dict[str, Any]):
        """
        IMPORTANT:
        PredictionState is consumed as-is.
        This node does NOT judge or validate predictions.
        """
        self.recent_prediction_states.append(data)

    def ingest_cognitive_directive(self, data: Dict[str, Any]):
        self.recent_cognitive_directives.append(data)

    # ------------------------------------------------------------------ #
    # World model update                                                 #
    # ------------------------------------------------------------------ #
    def _run_world_model_update_wrapper(self):
        if self.active_task and not self.active_task.done():
            return

        self.active_task = asyncio.run_coroutine_threadsafe(
            self.update_world_model_async(),
            self._async_loop
        )

    async def update_world_model_async(self):
        self._prune_history()

        world_state = {
            "timestamp": time.time(),
            "sensory_snapshot": list(self.recent_sensory_qualia),
            "memory_snapshot": list(self.recent_memory_responses),
            "attention_snapshot": list(self.recent_attention_states),
            "prediction_snapshot": list(self.recent_prediction_states),
            "directives_snapshot": list(self.recent_cognitive_directives),
        }

        self.persist_world_state(world_state)

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def persist_world_state(self, state: Dict[str, Any]):
        try:
            self.conn.execute(
                """
                INSERT INTO world_states (id, timestamp, state_json)
                VALUES (?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    state["timestamp"],
                    json.dumps(state)
                )
            )
            self.conn.commit()
            log("INFO", self.node_name, "World state persisted")
        except Exception as e:
            log("ERROR", self.node_name, f"Failed to persist world state: {e}")

    # ------------------------------------------------------------------ #
    # Runtime                                                            #
    # ------------------------------------------------------------------ #
    def run(self):
        try:
            while not self._shutdown_flag.is_set():
                self._run_world_model_update_wrapper()
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        log("INFO", self.node_name, "Shutting down WorldModelNode")
        self._shutdown_flag.set()
        self._shutdown_async_loop()
        self.conn.close()


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    node = WorldModelNode()
    node.run()
