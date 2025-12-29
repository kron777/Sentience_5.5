#!/usr/bin/env python3
"""
StillnessNode – Sentience 5.5 compliant

Role:
- Provide deterministic cognitive pauses
- Coordinate system-wide quiescence
- No emotion, no bias, no interpretation
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import sqlite3
import argparse
import asyncio
from datetime import datetime
from typing import Optional
from aiohttp import web


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# Stillness Node                                                              #
# --------------------------------------------------------------------------- #
class StillnessNode:
    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "stillness_node"
        self.db_path = os.path.join(db_root, "stillness.db")
        os.makedirs(db_root, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # State
        self.is_still: bool = False
        self.stillness_until: Optional[float] = None

        log("INFO", self.node_name, "StillnessNode initialized")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stillness_events (
                id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                duration REAL,
                trigger TEXT
            )
        """)
        self.conn.commit()

    def _log_event(self, start: float, end: float, trigger: str):
        self.conn.execute(
            """
            INSERT INTO stillness_events
            (id, start_time, end_time, duration, trigger)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                start,
                end,
                end - start,
                trigger
            )
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Core Stillness Logic                                                #
    # ------------------------------------------------------------------ #
    async def enter_stillness(self, duration: float, trigger: str):
        if self.is_still:
            return

        start = time.time()
        self.is_still = True
        self.stillness_until = start + duration

        log("INFO", self.node_name, f"Entering stillness for {duration:.2f}s (trigger={trigger})")

        await asyncio.sleep(duration)

        end = time.time()
        self.is_still = False
        self.stillness_until = None

        self._log_event(start, end, trigger)
        log("INFO", self.node_name, "Exiting stillness")

    # ------------------------------------------------------------------ #
    # HTTP Handlers                                                      #
    # ------------------------------------------------------------------ #
    async def trigger_handler(self, request: web.Request) -> web.Response:
        data = await request.json()
        duration = float(data.get("duration", 1.0))
        trigger = data.get("trigger", "external")

        asyncio.create_task(self.enter_stillness(duration, trigger))

        return web.json_response({
            "status": "accepted",
            "duration": duration,
            "trigger": trigger
        })

    async def status_handler(self, request: web.Request) -> web.Response:
        return web.json_response({
            "node": self.node_name,
            "is_still": self.is_still,
            "until": self.stillness_until,
            "timestamp": time.time()
        })

    async def health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "node": self.node_name
        })

    # ------------------------------------------------------------------ #
    # App                                                                #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/stillness/trigger", self.trigger_handler),
            web.get("/stillness/status", self.status_handler),
            web.get("/stillness/health", self.health_handler),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(description="Sentience 5.5 – Stillness Node")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()

    node = StillnessNode()

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        log("INFO", node.node_name, f"Serving on :{args.port}")
        await asyncio.Event().wait()
    else:
        log("ERROR", node.node_name, "Use --serve")


if __name__ == "__main__":
    asyncio.run(amain())
