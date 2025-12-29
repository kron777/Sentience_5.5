#!/usr/bin/env python3
"""
TemporalNarrativeNode – Sentience 5.5

Purpose:
- Maintain a coherent temporal self-narrative
- Integrate memory, emotion, and goals
- Produce a stable story of experience over time
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import sqlite3
import argparse
import threading
import random
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional, Deque

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)

# ---------------------------------------------------------------------
# Temporal Narrative Node
# ---------------------------------------------------------------------
class TemporalNarrativeNode:
    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "temporal_narrative_node"

        # -----------------------------
        # Config
        # -----------------------------
        self.narrative_window = 10
        self.ethical_compassion_bias = 0.2

        # -----------------------------
        # Internal State
        # -----------------------------
        self.memory_events: Deque[Dict[str, Any]] = deque(maxlen=self.narrative_window)
        self.emotional_trace: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.goal_history: Deque[Dict[str, Any]] = deque(maxlen=5)

        self.sensory_snapshot: Dict[str, Any] = {}
        self.cumulative_salience: float = 0.0

        # -----------------------------
        # Database
        # -----------------------------
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "temporal_narrative.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS narratives (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                theme TEXT,
                memory_json TEXT,
                emotion TEXT,
                goals_json TEXT,
                sensory_json TEXT
            )
        """)
        self.conn.commit()

        # -----------------------------
        # Runtime
        # -----------------------------
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        log("INFO", self.node_name, "Temporal Narrative Node online")

    # -----------------------------------------------------------------
    # Core loop
    # -----------------------------------------------------------------
    def _loop(self):
        while not self._shutdown.is_set():
            self.generate_narrative()
            time.sleep(6.0)

    # -----------------------------------------------------------------
    # Narrative generation
    # -----------------------------------------------------------------
    def generate_narrative(self):
        theme = self._infer_theme()
        mood = self._summarize_emotion()

        story = {
            "timestamp": time.time(),
            "narrative_theme": theme,
            "recent_memory": list(self.memory_events),
            "mood_trend": mood,
            "goal_trail": list(self.goal_history),
        }

        self._persist(story)
        log("INFO", self.node_name, f"Narrative generated → {theme}")

    # -----------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------
    def _infer_theme(self) -> str:
        if any("failure" in e["event"] for e in self.memory_events):
            return "resilience"
        if any(g["status"] == "completed" for g in self.goal_history):
            return "progress"
        if self.emotional_trace:
            return "reflection"
        return "continuity"

    def _summarize_emotion(self) -> str:
        if not self.emotional_trace:
            return "neutral"
        moods = {}
        for e in self.emotional_trace:
            moods[e["mood"]] = moods.get(e["mood"], 0) + 1
        dominant = max(moods, key=moods.get)
        if dominant in ("sad", "anxious"):
            dominant += " (resilient)"
        return dominant

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def _persist(self, story: Dict[str, Any]):
        self.cursor.execute("""
            INSERT INTO narratives
            (id, timestamp, theme, memory_json, emotion, goals_json, sensory_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            story["timestamp"],
            story["narrative_theme"],
            json.dumps(story["recent_memory"]),
            story["mood_trend"],
            json.dumps(story["goal_trail"]),
            json.dumps(self.sensory_snapshot),
        ))
        self.conn.commit()

    # -----------------------------------------------------------------
    # Public inputs (safe, minimal)
    # -----------------------------------------------------------------
    def ingest_memory(self, event: str):
        self.memory_events.append({"event": event, "timestamp": time.time()})

    def ingest_emotion(self, mood: str, intensity: float):
        self.emotional_trace.append({
            "mood": mood,
            "intensity": intensity,
            "timestamp": time.time()
        })

    def ingest_goal(self, goal: str, status: str):
        self.goal_history.append({
            "goal": goal,
            "status": status,
            "timestamp": time.time()
        })

    # -----------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------
    def shutdown(self):
        self._shutdown.set()
        self.conn.close()
        log("INFO", self.node_name, "Temporal Narrative Node shutdown")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    node = TemporalNarrativeNode()

    try:
        while True:
            # simple live simulation
            node.ingest_memory(random.choice(["learning", "challenge", "success"]))
            node.ingest_emotion(random.choice(["neutral", "reflective", "hopeful"]), random.random())
            node.ingest_goal("self_continuity", random.choice(["in_progress", "completed"]))
            time.sleep(4)
    except KeyboardInterrupt:
        node.shutdown()
