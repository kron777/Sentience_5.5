#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid
import sys
import argparse
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional

import asyncio
import aiohttp
import threading

# =============================
# Safety / Stability Constants
# =============================
MAX_SALIENCE = 1.0
MIN_SALIENCE = 0.0
MAX_SEVERITY = 1.0
MIN_SEVERITY = 0.0
LLM_TIMEOUT_S = 1.2
MIN_ANALYSIS_INTERVAL = 0.5

# =============================
# Logging helpers
# =============================
def _log(level: str, node: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)

# =============================
# Bias Mitigation Node
# =============================
class BiasMitigationNode:
    """
    Detects and mitigates cognitive bias.
    LLM is advisory only. Final authority is rule-based and bounded.
    """

    def __init__(self, config_file_path: Optional[str] = None):
        self.node_name = "bias_mitigation_node"

        # -------------------------
        # Config
        # -------------------------
        self.mitigation_interval = 1.0
        self.llm_trigger_salience = 0.6
        self.recent_context_window_s = 20.0

        self.llm_model = "phi-2"
        self.llm_url = "http://localhost:8000/v1/chat/completions"

        self.db_path = os.path.expanduser("~/sentience_bias_log.db")

        # -------------------------
        # Internal State
        # -------------------------
        self.cumulative_salience = 0.0
        self.last_analysis_ts = 0.0

        self.current_state = {
            "timestamp": time.time(),
            "bias_type": "none",
            "detected_severity": 0.0,
            "mitigation_status": "idle",
        }

        self.recent_internal_narratives = deque(maxlen=10)
        self.recent_interactions = deque(maxlen=10)
        self.recent_memory = deque(maxlen=5)
        self.recent_reflections = deque(maxlen=5)
        self.recent_directives = deque(maxlen=5)

        self.sensory_data = {
            "vision": None,
            "sound": None,
            "instructions": None,
        }

        # Evolver metrics
        self.metrics = {
            "cycles": 0,
            "llm_calls": 0,
            "rule_triggers": 0,
            "llm_failures": 0,
            "avg_severity": 0.0,
        }

        # -------------------------
        # DB Setup
        # -------------------------
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS bias_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                bias_type TEXT,
                detected_severity REAL,
                mitigation_status TEXT,
                reasoning TEXT,
                context_json TEXT
            )
        """)
        self.conn.commit()

        # -------------------------
        # Async Setup
        # -------------------------
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._run_async_loop, daemon=True
        )
        self._async_thread.start()
        self._session: Optional[aiohttp.ClientSession] = None
        self.active_task: Optional[asyncio.Future] = None

        # -------------------------
        # Execution Thread
        # -------------------------
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._execution_loop, daemon=True
        )
        self._thread.start()

        _log("INFO", self.node_name, "BiasMitigationNode initialized.")

    # =============================
    # Async loop
    # =============================
    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_until_complete(self._create_session())
        self._async_loop.run_forever()

    async def _create_session(self):
        self._session = aiohttp.ClientSession()

    async def _close_session(self):
        if self._session:
            await self._session.close()
            self._session = None

    # =============================
    # Input APIs
    # =============================
    def receive_internal_narrative(self, data: Dict[str, Any]):
        self.recent_internal_narratives.append(data)
        self._bump_salience(data.get("salience_score", 0.1))

    def receive_interaction(self, data: Dict[str, Any]):
        self.recent_interactions.append(data)
        self._bump_salience(data.get("urgency_score", 0.1))

    def receive_memory(self, data: Dict[str, Any]):
        self.recent_memory.append(data)
        self._bump_salience(0.2)

    def receive_reflection(self, data: Dict[str, Any]):
        self.recent_reflections.append(data)
        if data.get("consistency_score", 1.0) < 0.7:
            self._bump_salience(0.3)

    def receive_sensory(self, sensor: str, payload: Any):
        self.sensory_data[sensor] = payload
        self._bump_salience(0.2)

    # =============================
    # Core loop
    # =============================
    def _execution_loop(self):
        while not self._shutdown.is_set():
            self._analysis_cycle()
            time.sleep(self.mitigation_interval)

    def _analysis_cycle(self):
        now = time.time()
        if now - self.last_analysis_ts < MIN_ANALYSIS_INTERVAL:
            return

        self.metrics["cycles"] += 1
        self.last_analysis_ts = now

        if self.cumulative_salience < self.llm_trigger_salience:
            self._apply_simple_rules()
            return

        if self.active_task and not self.active_task.done():
            return

        self.active_task = asyncio.run_coroutine_threadsafe(
            self._llm_bias_analysis(), self._async_loop
        )

        try:
            result = self.active_task.result(timeout=LLM_TIMEOUT_S)
            if result:
                self._apply_llm_result(result)
        except Exception:
            self.metrics["llm_failures"] += 1
            self._apply_simple_rules()

        self.cumulative_salience = 0.0

    # =============================
    # Salience handling
    # =============================
    def _bump_salience(self, delta: float):
        self.cumulative_salience = max(
            MIN_SALIENCE,
            min(MAX_SALIENCE, self.cumulative_salience + delta),
        )

    # =============================
    # Simple rule fallback
    # =============================
    def _apply_simple_rules(self):
        self.metrics["rule_triggers"] += 1

        bias_type = "none"
        severity = 0.0

        if self.recent_internal_narratives and self.recent_memory:
            narrative = self.recent_internal_narratives[-1]
            if "always" in narrative.get("narrative_text", "").lower():
                bias_type = "confirmation_bias"
                severity = 0.5

        self._update_state(bias_type, severity, "detected" if severity > 0 else "idle", "rule-based")

    # =============================
    # LLM analysis (advisory)
    # =============================
    async def _llm_bias_analysis(self) -> Optional[Dict[str, Any]]:
        if not self._session:
            return None

        prompt = {
            "context": {
                "narratives": list(self.recent_internal_narratives),
                "interactions": list(self.recent_interactions),
                "memory": list(self.recent_memory),
                "reflections": list(self.recent_reflections),
                "sensory": self.sensory_data,
            }
        }

        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": json.dumps(prompt)}],
            "temperature": 0.3,
            "max_tokens": 300,
        }

        try:
            async with self._session.post(
                self.llm_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=LLM_TIMEOUT_S),
            ) as resp:
                if resp.status != 200:
                    return None
                self.metrics["llm_calls"] += 1
                data = await resp.json()
        except Exception:
            return None

        try:
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return None

    # =============================
    # State update + logging
    # =============================
    def _update_state(self, bias_type: str, severity: float, status: str, source: str):
        severity = max(MIN_SEVERITY, min(MAX_SEVERITY, severity))

        self.current_state = {
            "timestamp": time.time(),
            "bias_type": bias_type,
            "detected_severity": severity,
            "mitigation_status": status,
        }

        self.metrics["avg_severity"] = (
            self.metrics["avg_severity"] * 0.9 + severity * 0.1
        )

        self.cursor.execute(
            """
            INSERT INTO bias_log
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                datetime.utcnow().isoformat() + "Z",
                bias_type,
                severity,
                status,
                source,
                json.dumps(self.current_state),
            ),
        )
        self.conn.commit()

        _log(
            "INFO",
            self.node_name,
            f"Bias={bias_type} Severity={severity:.2f} Source={source}",
        )

    def _apply_llm_result(self, result: Dict[str, Any]):
        self._update_state(
            result.get("bias_type", "none"),
            float(result.get("detected_severity", 0.0)),
            result.get("mitigation_status", "monitor"),
            "llm",
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
    node = BiasMitigationNode()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.shutdown()
