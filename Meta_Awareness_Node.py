#!/usr/bin/env python3
"""
MetaAwarenessNode – Updated (Sentience 5.5 compliant)

Purpose:
- Monitor confidence, contradiction, and internal coherence
- Detect cognitive dissonance & uncertainty
- Emit self-reflection directives (never actions)
- Deterministic + evolver-visible
- ROS-optional, HTTP-first, async-safe
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import sqlite3
import argparse
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from collections import deque

from aiohttp import web

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# Meta-Awareness Node                                                         #
# --------------------------------------------------------------------------- #
class MetaAwarenessNode:
    """
    Second-order monitoring of internal cognition.
    Does NOT decide actions.
    """

    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "meta_awareness_node"
        self.db_path = os.path.join(db_root, "meta_awareness.db")
        os.makedirs(db_root, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # --- state ---
        self.confidence_level: float = 1.0
        self.dissonance: float = 0.0
        self.last_emotion = {"mood": "neutral", "intensity": 0.0}

        self.confidence_threshold = 0.4
        self.dissonance_threshold = 0.6

        # rolling history for evolver
        self.history = deque(maxlen=100)

        # queues
        self.state_queue: asyncio.Queue[str] = asyncio.Queue()
        self.directive_queue: asyncio.Queue[str] = asyncio.Queue()

        log("INFO", self.node_name, "MetaAwarenessNode initialized")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_awareness (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                confidence REAL,
                dissonance REAL,
                mood TEXT,
                emotion_intensity REAL
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Input handlers                                                     #
    # ------------------------------------------------------------------ #
    def ingest_prediction(self, confidence: float, accurate: bool):
        if not accurate:
            self.dissonance += abs(self.confidence_level - confidence)
        self.confidence_level = confidence

    def ingest_emotion(self, mood: str, intensity: float):
        self.last_emotion = {"mood": mood, "intensity": intensity}
        if intensity > 0.7:
            self.dissonance += 0.1

    def ingest_narrative(self, theme: str):
        if "conflict" in theme.lower():
            self.dissonance += 0.2

    # ------------------------------------------------------------------ #
    # Core evaluation                                                    #
    # ------------------------------------------------------------------ #
    def evaluate(self) -> Dict[str, Any]:
        state = {
            "timestamp": time.time(),
            "confidence": round(self.confidence_level, 3),
            "dissonance": round(self.dissonance, 3),
            "mood": self.last_emotion["mood"],
            "emotion_intensity": round(self.last_emotion["intensity"], 3),
        }

        self._log_state(state)
        self.history.append(state)

        if (
            state["confidence"] < self.confidence_threshold
            or state["dissonance"] > self.dissonance_threshold
        ):
            self._emit_reflection_directive(state)

        # decay dissonance
        self.dissonance = max(0.0, self.dissonance * 0.8)

        return state

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def _log_state(self, state: Dict[str, Any]):
        self.conn.execute(
            """
            INSERT INTO meta_awareness
            (id, timestamp, confidence, dissonance, mood, emotion_intensity)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                state["timestamp"],
                state["confidence"],
                state["dissonance"],
                state["mood"],
                state["emotion_intensity"],
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Directives                                                         #
    # ------------------------------------------------------------------ #
    def _emit_reflection_directive(self, state: Dict[str, Any]):
        directive = {
            "type": "self_reflection",
            "reason": "low_confidence_or_high_dissonance",
            "context": state,
            "timestamp": time.time(),
        }
        self.directive_queue.put_nowait(json.dumps(directive))
        log("WARN", self.node_name, f"Reflection triggered: {json.dumps(directive)}")

    # ------------------------------------------------------------------ #
    # HTTP API                                                           #
    # ------------------------------------------------------------------ #
    async def handle_prediction(self, request: web.Request) -> web.Response:
        d = await request.json()
        self.ingest_prediction(d.get("confidence", 1.0), d.get("accurate", True))
        return web.json_response({"status": "ok"})

    async def handle_emotion(self, request: web.Request) -> web.Response:
        d = await request.json()
        self.ingest_emotion(d.get("mood", "neutral"), d.get("intensity", 0.0))
        return web.json_response({"status": "ok"})

    async def handle_narrative(self, request: web.Request) -> web.Response:
        d = await request.json()
        self.ingest_narrative(d.get("main_theme", ""))
        return web.json_response({"status": "ok"})

    async def handle_state(self, request: web.Request) -> web.Response:
        state = self.evaluate()
        return web.json_response(state)

    async def handle_directive(self, request: web.Request) -> web.Response:
        try:
            msg = await asyncio.wait_for(self.directive_queue.get(), timeout=30)
            return web.json_response(json.loads(msg))
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ------------------------------------------------------------------ #
    # App builder                                                        #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/meta/prediction", self.handle_prediction),
            web.post("/meta/emotion", self.handle_emotion),
            web.post("/meta/narrative", self.handle_narrative),
            web.get("/meta/state", self.handle_state),
            web.get("/meta/directive", self.handle_directive),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(description="Sentience 5.5 – MetaAwarenessNode")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()

    node = MetaAwarenessNode()

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        log("INFO", node.node_name, f"MetaAwarenessNode running on :{args.port}")
        await asyncio.Event().wait()
    else:
        log("ERROR", node.node_name, "Use --serve")


if __name__ == "__main__":
    asyncio.run(amain())
