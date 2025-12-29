#!/usr/bin/env python3
"""
MotivationNode – Sentience 5.5 compliant

Purpose:
- Maintain internal motivation level (0.0–1.0)
- Select *intent focus* (goal orientation)
- React to system integration + monitoring signals
- NO action authority
- Deterministic, inspectable, evolver-visible
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
from typing import Dict, Any
from aiohttp import web


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# Motivation Node                                                             #
# --------------------------------------------------------------------------- #
class MotivationNode:
    """
    Maintains motivational state and intent focus.
    """

    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "motivation_node"
        self.db_path = os.path.join(db_root, "motivation.db")
        os.makedirs(db_root, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # internal state
        self.motivation_level: float = 0.5
        self.goal: str = "maintenance"

        # async channels
        self.state_queue: asyncio.Queue[str] = asyncio.Queue()

        log("INFO", self.node_name, "MotivationNode initialized")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS motivation (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                motivation REAL,
                goal TEXT,
                source TEXT
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Motivation update logic                                            #
    # ------------------------------------------------------------------ #
    def update_from_integration(self, data: Dict[str, Any]):
        """
        Input schema (example):
        {
            "decision_confidence": 0.72,
            "system_status": "normal" | "alert"
        }
        """

        confidence = float(data.get("decision_confidence", 0.5))
        status = data.get("system_status", "normal")

        # deterministic motivation shaping
        if status == "alert":
            self.motivation_level = max(0.0, self.motivation_level - 0.15)
            self.goal = "recovery"
        elif confidence >= 0.75:
            self.motivation_level = min(1.0, self.motivation_level + 0.1)
            self.goal = "optimization"
        elif confidence <= 0.4:
            self.motivation_level = max(0.0, self.motivation_level - 0.1)
            self.goal = "reflection"
        else:
            self.motivation_level = 0.5
            self.goal = "maintenance"

        self._persist("integration")
        self._emit_state()

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def _persist(self, source: str):
        self.conn.execute(
            """
            INSERT INTO motivation
            (id, timestamp, motivation, goal, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                time.time(),
                self.motivation_level,
                self.goal,
                source,
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # State emission                                                     #
    # ------------------------------------------------------------------ #
    def _emit_state(self):
        state = {
            "timestamp": time.time(),
            "motivation_level": round(self.motivation_level, 3),
            "goal": self.goal,
        }
        self.state_queue.put_nowait(json.dumps(state))
        log("INFO", self.node_name, f"State updated: {json.dumps(state)}")

    # ------------------------------------------------------------------ #
    # HTTP API                                                           #
    # ------------------------------------------------------------------ #
    async def handle_update(self, request: web.Request) -> web.Response:
        data = await request.json()
        self.update_from_integration(data)
        return web.json_response({"status": "ok"})

    async def handle_state(self, request: web.Request) -> web.Response:
        return web.json_response({
            "motivation_level": round(self.motivation_level, 3),
            "goal": self.goal,
            "timestamp": time.time(),
        })

    async def handle_stream(self, request: web.Request) -> web.Response:
        try:
            msg = await asyncio.wait_for(self.state_queue.get(), timeout=30)
            return web.json_response(json.loads(msg))
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ------------------------------------------------------------------ #
    # App builder                                                        #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/motivation/update", self.handle_update),
            web.get("/motivation/state", self.handle_state),
            web.get("/motivation/stream", self.handle_stream),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(description="Sentience 5.5 – MotivationNode")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()

    node = MotivationNode()

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        log("INFO", node.node_name, f"MotivationNode running on :{args.port}")
        await asyncio.Event().wait()
    else:
        log("ERROR", node.node_name, "Use --serve")


if __name__ == "__main__":
    asyncio.run(amain())
