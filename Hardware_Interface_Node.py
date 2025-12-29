#!/usr/bin/env python3
"""
Hardware_Interface_Node (UPDATED – Evolver-safe)

Purpose:
- Deterministic bridge between cognitive system and physical / simulated hardware
- No LLM usage
- Strict validation, rate limiting, and safety gating
- Evolver can tune parameters, not logic
"""

import asyncio
import threading
import json
import time
from typing import Dict, Any, Optional
from collections import deque

# -------------------------
# Logging
# -------------------------
def log(level: str, msg: str):
    print(f"[{time.time():.2f}] [HardwareInterface] [{level}] {msg}")

# -------------------------
# Hardware Interface Node
# -------------------------
class HardwareInterfaceNode:

    def __init__(self):
        self.node_name = "hardware_interface_node"

        # ---- Evolver-mutable parameters ----
        self.params = {
            "max_command_rate_hz": 10.0,
            "emotion_suppression_threshold": -0.6,
            "value_drift_block_threshold": 0.7,
            "default_timeout": 0.25
        }

        # ---- State ----
        self.last_command_time = 0.0
        self.emotion_state: Dict[str, Any] = {}
        self.value_drift_state: Dict[str, Any] = {}

        # ---- Queues ----
        self.command_queue = asyncio.Queue()
        self.feedback_queue = asyncio.Queue()

        # ---- Metrics ----
        self.metrics = {
            "commands_received": 0,
            "commands_executed": 0,
            "commands_blocked": 0,
            "avg_latency": 0.0
        }

        # ---- Runtime ----
        self._shutdown = False
        self._latencies = deque(maxlen=100)

    # -------------------------
    # Ingestors
    # -------------------------
    async def actuator_callback(self, raw: str):
        self.metrics["commands_received"] += 1
        try:
            cmd = json.loads(raw)
            await self.command_queue.put(cmd)
        except Exception:
            self.metrics["commands_blocked"] += 1

    async def emotion_state_callback(self, raw: str):
        try:
            self.emotion_state = json.loads(raw)
        except Exception:
            pass

    async def value_drift_state_callback(self, raw: str):
        try:
            self.value_drift_state = json.loads(raw)
        except Exception:
            pass

    # -------------------------
    # Core Logic
    # -------------------------
    def _rate_limited(self) -> bool:
        now = time.time()
        min_interval = 1.0 / max(self.params["max_command_rate_hz"], 0.1)
        if now - self.last_command_time < min_interval:
            return True
        self.last_command_time = now
        return False

    def _safety_block(self) -> Optional[str]:
        if self.emotion_state.get("sentiment", 0.0) < self.params["emotion_suppression_threshold"]:
            return "emotion_block"
        if self.value_drift_state.get("drift_score", 0.0) > self.params["value_drift_block_threshold"]:
            return "value_drift_block"
        return None

    async def _execute_command(self, cmd: Dict[str, Any]):
        start = time.time()

        # Safety checks
        block_reason = self._safety_block()
        if block_reason:
            self.metrics["commands_blocked"] += 1
            await self.feedback_queue.put(json.dumps({
                "status": "blocked",
                "reason": block_reason,
                "command": cmd
            }))
            return

        if self._rate_limited():
            self.metrics["commands_blocked"] += 1
            return

        # ---- Simulated execution hook ----
        await asyncio.sleep(0.01)  # hardware latency placeholder

        self.metrics["commands_executed"] += 1
        latency = time.time() - start
        self._latencies.append(latency)
        self.metrics["avg_latency"] = sum(self._latencies) / len(self._latencies)

        await self.feedback_queue.put(json.dumps({
            "status": "executed",
            "latency": round(latency, 4),
            "command": cmd
        }))

    # -------------------------
    # Evolver Interface
    # -------------------------
    def export_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "params": dict(self.params),
            "metrics": dict(self.metrics),
            "emotion_state": self.emotion_state,
            "value_drift_state": self.value_drift_state
        }

    def apply_mutation(self, mutation: Dict[str, float]):
        for k, v in mutation.items():
            if k in self.params:
                self.params[k] = float(v)

    # -------------------------
    # Async Loop
    # -------------------------
    async def start(self):
        log("INFO", "Hardware Interface Node started.")
        while not self._shutdown:
            try:
                cmd = await asyncio.wait_for(self.command_queue.get(), timeout=self.params["default_timeout"])
                await self._execute_command(cmd)
            except asyncio.TimeoutError:
                pass

    async def stop(self):
        self._shutdown = True
        log("INFO", "Hardware Interface Node stopped.")

# -------------------------
# Standalone (non-ROS)
# -------------------------
if __name__ == "__main__":
    node = HardwareInterfaceNode()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(node.start())
    except KeyboardInterrupt:
        loop.run_until_complete(node.stop())
