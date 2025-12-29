#!/usr/bin/env python3
"""
SystemIntegrationNode – Sentience 5.5 (Corrected)

Role:
- Aggregate system health
- Coordinate nodes
- Issue stabilizing directives
- Maintain character traits
- NO prediction, NO hallucinated planning
"""

from __future__ import annotations

import os
import json
import time
import sys
import argparse
import sqlite3
import asyncio
import aiohttp
import threading
import random
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# System Integration Node                                                     #
# --------------------------------------------------------------------------- #
class SystemIntegrationNode:
    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "system_integration_node"

        # Paths
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "system_log.db")
        self.traits_path = os.path.join(db_root, "character_traits.json")

        # Core state
        self.update_interval = 1.0
        self.ethical_compassion_bias = 0.2
        self.max_event_history = 50

        # Sensory snapshot (passive only)
        self.sensory_data: Dict[str, Any] = {}

        # Node status registry
        self.node_status: Dict[str, Dict[str, Any]] = {}
        self.event_history = deque(maxlen=self.max_event_history)

        # Trait system
        self.character_traits = self._load_character_traits()
        self.trait_update_buffer = deque(maxlen=50)

        # Database
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

        # Async loop
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.session: Optional[aiohttp.ClientSession] = None

        log("INFO", self.node_name, "SystemIntegrationNode online (stability-first).")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_log (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                system_health REAL,
                directive_type TEXT,
                target_node TEXT,
                confidence REAL
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Async Loop                                                         #
    # ------------------------------------------------------------------ #
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._init_async())
        self.loop.run_forever()

    async def _init_async(self):
        self.session = aiohttp.ClientSession()

    # ------------------------------------------------------------------ #
    # Status Updates                                                     #
    # ------------------------------------------------------------------ #
    def update_node_status(self, node_name: str, status: str):
        self.node_status[node_name] = {
            "status": status,
            "last_updated": time.time()
        }

    # ------------------------------------------------------------------ #
    # Health Evaluation                                                  #
    # ------------------------------------------------------------------ #
    def evaluate_system_health(self) -> float:
        if not self.node_status:
            return 0.0
        healthy = sum(1 for n in self.node_status.values() if n["status"] == "running")
        return healthy / len(self.node_status)

    # ------------------------------------------------------------------ #
    # Directive Logic (NO prediction)                                    #
    # ------------------------------------------------------------------ #
    def generate_directive(self, health: float) -> Dict[str, Any]:
        if health < 0.5:
            return {
                "directive_type": "stabilize",
                "target_node": "self_correction_node",
                "confidence": 0.9
            }
        if health < 0.8:
            return {
                "directive_type": "observe",
                "target_node": "attention_node",
                "confidence": 0.6
            }
        return {
            "directive_type": "none",
            "target_node": "none",
            "confidence": 0.3
        }

    # ------------------------------------------------------------------ #
    # Main Tick                                                          #
    # ------------------------------------------------------------------ #
    def tick(self):
        health = self.evaluate_system_health()
        directive = self.generate_directive(health)

        self.event_history.append({
            "timestamp": time.time(),
            "health": health,
            **directive
        })

        self.cursor.execute(
            "INSERT INTO system_log VALUES (?, ?, ?, ?, ?, ?)",
            (
                os.urandom(8).hex(),
                time.time(),
                health,
                directive["directive_type"],
                directive["target_node"],
                directive["confidence"]
            )
        )
        self.conn.commit()

        log(
            "INFO",
            self.node_name,
            f"Health={health:.2f} Directive={directive['directive_type']}"
        )

    # ------------------------------------------------------------------ #
    # Traits                                                             #
    # ------------------------------------------------------------------ #
    def _load_character_traits(self) -> Dict[str, Any]:
        if os.path.exists(self.traits_path):
            try:
                with open(self.traits_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "personality": {
                "reliability": {"value": 0.7},
                "empathy": {"value": 0.8}
            }
        }

    # ------------------------------------------------------------------ #
    # Run / Shutdown                                                     #
    # ------------------------------------------------------------------ #
    def run(self):
        try:
            while True:
                self.tick()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        log("INFO", self.node_name, "Shutting down.")
        if self.session:
            asyncio.run_coroutine_threadsafe(self.session.close(), self.loop)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)
        self.conn.close()


# --------------------------------------------------------------------------- #
# Entry                                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    node = SystemIntegrationNode()
    node.update_node_status("sensory_qualia_node", "running")
    node.update_node_status("memory_node", "running")
    node.update_node_status("emotion_node", "paused")
    node.run()
