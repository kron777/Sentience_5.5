#!/usr/bin/env python3
"""
SelfAwarenessNode – Sentience 5.5 compliant (cleaned + aligned)

Role:
- Track internal coherence, uptime, and anomaly signals
- No ethical bias, no interpretation, no affect
- Pure state reporting for downstream reasoning / prediction nodes
- ROS-free by default, HTTP-first
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
from typing import Dict, Any, List
from datetime import datetime
from aiohttp import web


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# Self-Awareness Node                                                         #
# --------------------------------------------------------------------------- #
class SelfAwarenessNode:
    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "self_awareness_node"
        self.start_time = time.time()

        # Internal state (strictly descriptive)
        self.state: Dict[str, Any] = {
            "coherence": 1.0,
            "uptime": 0.0,
            "anomaly_count": 0,
            "last_anomaly": None,
        }

        # Database
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "self_awareness.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        log("INFO", self.node_name, "SelfAwarenessNode initialized")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS self_awareness (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                coherence REAL,
                uptime REAL,
                anomaly_count INTEGER,
                last_anomaly TEXT
            )
        """)
        self.conn.commit()

    def persist_state(self):
        self.conn.execute(
            """
            INSERT INTO self_awareness
            (id, timestamp, coherence, uptime, anomaly_count, last_anomaly)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                time.time(),
                self.state["coherence"],
                self.state["uptime"],
                self.state["anomaly_count"],
                json.dumps(self.state["last_anomaly"])
                if self.state["last_anomaly"] else None,
            )
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # State Updates                                                      #
    # ------------------------------------------------------------------ #
    def update_from_integration(self, payload: Dict[str, Any]):
        """
        Expect:
        {
            "components": {
                "<name>": {"confidence": float}
            }
        }
        """
        components = payload.get("components", {})
        confidences = [
            v["confidence"]
            for v in components.values()
            if isinstance(v, dict) and "confidence" in v
        ]

        if confidences:
            self.state["coherence"] = sum(confidences) / len(confidences)
        else:
            self.state["coherence"] = 0.0

    def update_from_monitoring(self, payload: Dict[str, Any]):
        """
        Expect:
        {
            "status": "normal" | "alert",
            "message": optional
        }
        """
        if payload.get("status") == "alert":
            self.state["anomaly_count"] += 1
            self.state["last_anomaly"] = {
                "time": time.time(),
                "message": payload.get("message", "unspecified"),
            }

    def tick(self):
        self.state["uptime"] = time.time() - self.start_time
        self.persist_state()

    # ------------------------------------------------------------------ #
    # HTTP Handlers                                                      #
    # ------------------------------------------------------------------ #
    async def integration_handler(self, request: web.Request) -> web.Response:
        payload = await request.json()
        self.update_from_integration(payload)
        return web.json_response({"status": "ok"})

    async def monitoring_handler(self, request: web.Request) -> web.Response:
        payload = await request.json()
        self.update_from_monitoring(payload)
        return web.json_response({"status": "ok"})

    async def state_handler(self, request: web.Request) -> web.Response:
        self.tick()
        return web.json_response({
            "timestamp": time.time(),
            "state": self.state
        })

    async def health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "node": self.node_name,
            "time": time.time()
        })

    # ------------------------------------------------------------------ #
    # App                                                                #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/self_awareness/integration", self.integration_handler),
            web.post("/self_awareness/monitoring", self.monitoring_handler),
            web.get("/self_awareness/state", self.state_handler),
            web.get("/self_awareness/health", self.health_handler),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(description="Sentience 5.5 – Self Awareness Node")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8096)
    args = parser.parse_args()

    node = SelfAwarenessNode()

    if not args.serve:
        log("ERROR", node.node_name, "Use --serve")
        return

    app = node.build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    log("INFO", node.node_name, f"Serving on :{args.port}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(amain())
