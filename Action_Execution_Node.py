#!/usr/bin/env python3
import os
import json
import time
import uuid
import sys
import argparse
import random
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque

import asyncio
import aiohttp
import threading

# Optional ROS Integration
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    pass


# ---------------- Logging ----------------
def _log(node, level, msg):
    print(f"[{datetime.now().isoformat()}] {node}: [{level}] {msg}", file=sys.stdout)


# ---------------- Node ----------------
class ActionExecutionNode:
    def __init__(self, ros_enabled: bool = False):
        self.node_name = "action_execution_node"
        self.ros_enabled = ros_enabled

        # --- Salience & learning ---
        self.cumulative_safety_salience = 0.0
        self.salience_decay_rate = 0.05     # <<< MOD >>> decay per cycle
        self.llm_trigger_threshold = 0.7

        self.execution_history: Deque[Dict[str, Any]] = deque(maxlen=50)
        self.llm_accuracy_window: Deque[bool] = deque(maxlen=20)  # <<< MOD >>>

        # --- Async LLM ---
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._session = None

        _log(self.node_name, "INFO", "Action Execution Node online (stabilised mode).")

    # ---------------- Async ----------------
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _ensure_session(self):
        if not self._session:
            self._session = aiohttp.ClientSession()

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        LLM returns ADVISORY safety opinion only.
        """
        await self._ensure_session()
        try:
            async with self._session.post(
                "http://localhost:8000/v1/chat/completions",
                json={
                    "model": "phi-2",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 300,
                },
                timeout=aiohttp.ClientTimeout(total=20),
            ) as r:
                r.raise_for_status()
                data = await r.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            _log(self.node_name, "WARN", f"LLM unavailable: {e}")
            return {"is_safe": True, "confidence": 0.5, "reason": "LLM unavailable"}

    # ---------------- Core Logic ----------------
    def _decay_salience(self):
        # <<< MOD >>> prevent runaway triggering
        self.cumulative_safety_salience = max(
            0.0, self.cumulative_safety_salience - self.salience_decay_rate
        )

    def _rule_based_safety(self, action: Dict[str, Any]) -> bool:
        # Simple deterministic guard
        if action.get("force", 0) > 0.9:
            return False
        if action.get("ethical_score", 1.0) < 0.5:
            return False
        return True

    def _llm_weight(self) -> float:
        """
        <<< MOD >>>
        Weight LLM advice based on recent correctness.
        """
        if not self.llm_accuracy_window:
            return 0.5
        return sum(self.llm_accuracy_window) / len(self.llm_accuracy_window)

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._decay_salience()  # <<< MOD >>>

        rule_safe = self._rule_based_safety(action)

        llm_advice = None
        llm_weight = 0.0

        if self.cumulative_safety_salience >= self.llm_trigger_threshold:
            prompt = f"Assess safety of action:\n{json.dumps(action, indent=2)}"
            llm_advice = asyncio.run_coroutine_threadsafe(
                self._call_llm(prompt), self._loop
            ).result(timeout=25)
            llm_weight = self._llm_weight()  # <<< MOD >>>

        # <<< MOD >>> combine instead of override
        if llm_advice:
            llm_safe = llm_advice.get("is_safe", True)
            combined_safe = (
                rule_safe if llm_weight < 0.5 else (rule_safe and llm_safe)
            )
        else:
            combined_safe = rule_safe

        success = False
        if combined_safe:
            success = self._perform_action(action)

        # <<< MOD >>> learning signal
        if llm_advice:
            self.llm_accuracy_window.append(success == llm_advice.get("is_safe", True))

        record = {
            "timestamp": time.time(),
            "action": action,
            "rule_safe": rule_safe,
            "llm_used": bool(llm_advice),
            "llm_weight": llm_weight,
            "success": success,
        }
        self.execution_history.append(record)

        return record

    def _perform_action(self, action: Dict[str, Any]) -> bool:
        """
        Placeholder for real execution.
        """
        # Simulate execution outcome
        return random.random() > 0.1

    # ---------------- Shutdown ----------------
    def shutdown(self):
        if self._session:
            asyncio.run_coroutine_threadsafe(self._session.close(), self._loop)
        self._loop.call_soon_threadsafe(self._loop.stop)
        _log(self.node_name, "INFO", "Shutdown complete.")


# ---------------- Main ----------------
if __name__ == "__main__":
    node = ActionExecutionNode()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.shutdown()
