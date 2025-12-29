#!/usr/bin/env python3
"""
SensoryQualiaNode – Sentience 5.5 compliant

Role:
- Capture raw system-level sensory signals
- No interpretation, no cognition
- Persist + expose instantaneous qualia snapshots
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import psutil
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
# Sensory Qualia Node                                                         #
# --------------------------------------------------------------------------- #
class SensoryQualiaNode:
    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "sensory_qualia_node"
        self.db_path = os.path.join(db_root, "sensory_qualia.db")
        os.makedirs(db_root, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        log("INFO", self.node_name, "SensoryQualiaNode initialized")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sensory_qualia (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                cpu_percent REAL,
                mem_percent REAL,
                net_sent_bytes REAL,
                net_recv_bytes REAL,
                disk_percent REAL
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Qualia Collection                                                  #
    # ------------------------------------------------------------------ #
    def collect_qualia(self) -> Dict[str, Any]:
        net = psutil.net_io_counters()

        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "net_sent_bytes": net.bytes_sent,
            "net_recv_bytes": net.bytes_recv,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def persist_qualia(self, qualia: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO sensory_qualia
            (id, timestamp, cpu_percent, mem_percent, net_sent_bytes, net_recv_bytes, disk_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                time.time(),
                qualia["cpu_percent"],
                qualia["memory_percent"],
                qualia["net_sent_bytes"],
                qualia["net_recv_bytes"],
                qualia["disk_percent"],
            )
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # HTTP Handlers                                                      #
    # ------------------------------------------------------------------ #
    async def snapshot_handler(self, request: web.Request) -> web.Response:
        qualia = self.collect_qualia()
        self.persist_qualia(qualia)

        payload = {
            "timestamp": time.time(),
            "qualia": qualia
        }

        log("INFO", self.node_name, "Qualia snapshot captured")
        return web.json_response(payload)

    async def health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({
            "node": self.node_name,
            "status": "ok",
            "time": time.time()
        })

    # ------------------------------------------------------------------ #
    # App Builder                                                        #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.get("/sensory/qualia/snapshot", self.snapshot_handler),
            web.get("/sensory/qualia/health", self.health_handler),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(description="Sentience 5.5 – Sensory Qualia Node")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()

    node = SensoryQualiaNode()

    if not args.serve:
        log("ERROR", node.node_name, "Use --serve to start HTTP service")
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
