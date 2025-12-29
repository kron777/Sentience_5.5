#!/usr/bin/env python3
"""
Experience_Motivation_Node
--------------------------
Purpose:
Synthesizes motivation state from experience, emotion, performance,
world change, and directives.

This node is LEARNING-CAPABLE via evolver.py but NOT autonomous.
"""

import os
import json
import time
import uuid
import random
import sqlite3
import threading
from collections import deque
from typing import Dict, Any, Optional

# =========================
# Utility logging
# =========================
def log(node, level, msg):
    print(f"[{time.time():.2f}] {node} [{level}] {msg}")

# =========================
# Experience Motivation Node
# =========================
class ExperienceMotivationNode:

    def __init__(self):
        self.node_name = "experience_motivation_node"

        # ---- Core tunable parameters (EVOLVER MUTATES THESE) ----
        self.params = {
            "drive_gain": 1.0,
            "emotion_weight": 0.6,
            "performance_weight": 0.7,
            "world_change_weight": 0.5,
            "directive_weight": 0.9,
            "decay_rate": 0.02,
            "goal_switch_threshold": 0.55
        }

        # ---- State ----
        self.current_goal = "explore"
        self.drive_level = 0.1
        self.active_goals = {"explore": 0.1}

        # ---- History buffers ----
        self.emotion_history = deque(maxlen=10)
        self.performance_history = deque(maxlen=10)
        self.world_history = deque(maxlen=10)
        self.directive_history = deque(maxlen=10)

        # ---- Learning metrics ----
        self.metrics = {
            "goal_stability": 0.0,
            "drive_variance": 0.0,
            "emotion_correlation": 0.0,
            "reward_alignment": 0.0,
            "adaptation_pressure": 0.0
        }

        # ---- Persistence ----
        self.db_path = "/tmp/sentience_db/motivation_log.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

        # ---- Thread loop ----
        self._shutdown = False
        threading.Thread(target=self._loop, daemon=True).start()

        log(self.node_name, "INFO", "Motivation node online (learning-enabled).")

    # =========================
    # Database
    # =========================
    def _init_db(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS motivation (
                id TEXT,
                timestamp REAL,
                goal TEXT,
                drive REAL,
                metrics TEXT
            )
        """)
        self.conn.commit()

    # =========================
    # Inputs
    # =========================
    def ingest_emotion(self, mood: str, intensity: float):
        self.emotion_history.append((mood, intensity))

    def ingest_performance(self, score: float):
        self.performance_history.append(score)

    def ingest_world_change(self, magnitude: float):
        self.world_history.append(magnitude)

    def ingest_directive(self, goal: str, urgency: float):
        self.directive_history.append((goal, urgency))

    # =========================
    # Core motivation synthesis
    # =========================
    def _synthesize_drive(self):
        emotion_signal = sum(i for _, i in self.emotion_history) * self.params["emotion_weight"]
        performance_signal = sum(self.performance_history) * self.params["performance_weight"]
        world_signal = sum(self.world_history) * self.params["world_change_weight"]
        directive_signal = sum(u for _, u in self.directive_history) * self.params["directive_weight"]

        raw_drive = (
            emotion_signal +
            performance_signal +
            world_signal +
            directive_signal
        ) * self.params["drive_gain"]

        # decay
        self.drive_level = max(0.0, self.drive_level * (1 - self.params["decay_rate"]))
        self.drive_level += raw_drive
        self.drive_level = min(1.0, self.drive_level)

    def _maybe_switch_goal(self):
        if self.drive_level > self.params["goal_switch_threshold"]:
            if self.directive_history:
                self.current_goal = self.directive_history[-1][0]

    # =========================
    # Learning Metrics
    # =========================
    def _update_metrics(self):
        self.metrics["goal_stability"] = len(set(self.active_goals)) / max(1, len(self.directive_history))
        self.metrics["drive_variance"] = abs(self.drive_level - 0.5)
        self.metrics["emotion_correlation"] = sum(i for _, i in self.emotion_history) / max(1, len(self.emotion_history))
        self.metrics["reward_alignment"] = sum(self.performance_history) / max(1, len(self.performance_history))
        self.metrics["adaptation_pressure"] = sum(self.world_history) / max(1, len(self.world_history))

    # =========================
    # Evolver Interface
    # =========================
    def export_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "goal": self.current_goal,
            "drive": self.drive_level,
            "params": dict(self.params),
            "metrics": dict(self.metrics)
        }

    def apply_mutation(self, mutation: Dict[str, float]):
        for k, v in mutation.items():
            if k in self.params:
                self.params[k] = float(v)

    # =========================
    # Main loop
    # =========================
    def _loop(self):
        while not self._shutdown:
            self._synthesize_drive()
            self._maybe_switch_goal()
            self._update_metrics()
            self._persist()
            time.sleep(1.0)

    def _persist(self):
        self.cursor.execute(
            "INSERT INTO motivation VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                time.time(),
                self.current_goal,
                self.drive_level,
                json.dumps(self.metrics)
            )
        )
        self.conn.commit()

    def shutdown(self):
        self._shutdown = True
        self.conn.close()
        log(self.node_name, "INFO", "Motivation node shutdown.")

# =========================
# Standalone
# =========================
if __name__ == "__main__":
    node = ExperienceMotivationNode()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        node.shutdown()
