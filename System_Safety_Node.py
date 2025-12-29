#!/usr/bin/env python3
"""
SystemSafetyNode – Sentience 5.5 compliant

Role:
- Final authority on system safety
- Consumes health + prediction signals
- Emits hard or soft safety states
- NO interpretation, NO emotion, NO narrative
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
from aiohttp import web


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# System Safety Node                                                          #
# --------------------------------------------------------------------------- #
class SystemSafetyNode:
    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "system_safety_node"
        self.db_path = os.path.join(db_root, "system_safety.db")
        os.makedirs(db_root, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # Safety thresholds (HARD)
        self.cpu_hard = 95.0
        self.mem_hard = 95.0
        self.cpu_warn = 90.0
        self.mem_warn = 90.0

        self.current_status = {
            "safe": True,
            "level": "normal",   # normal | warning | critical
            "shutdown": False
        }

        log("INFO", self.node_name, "SystemSafetyNode initialized")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS safety_log (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                level TEXT,
                safe BOOLEAN,
                shutdown BOOLEAN,
                details_json TEXT
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Core Safety Logic                                                   #
    # ------------------------------------------------------------------ #
    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cpu = payload.get("cpu_usage", 0.0)
        mem = payload.get("memory_usage", 0.0)
        cpu_pred = payload.get("predicted_cpu_usage", 0.0)
        mem_pred = payload.get("predicted_memory_usage", 0.0)

        status = {
            "safe": True,
            "level": "normal",
            "shutdown": False
        }

        # Hard violations (immediate shutdown)
        if cpu >= self.cpu_hard or mem >= self.mem_hard:
            status.update({
                "safe": False,
                "level": "critical",
                "shutdown": True
            })

        # Predictive violations (warning)
        elif cpu_pred >= self.cpu_warn or mem_pred >= self.mem_warn:
            status.update({
                "safe": False,
                "level": "warning",
                "shutdown": False
            })

        self.current_status = status
        self._persist(status, payload)
        return status

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def _persist(self, status: Dict[str, Any], payload: Dict[str, Any]):
        self.conn.execute(
            """
            INSERT INTO safety_log
            (id, timestamp, level, safe, shutdown, details_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                time.time(),
                status["level"],
                status["safe"],
                status["shutdown"],
                json.dumps(payload)
            )
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # HTTP Handlers                                                       #
    # ------------------------------------------------------------------ #
    async def safety_check_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            result = self.evaluate(data)

            log("INFO", self.node_name, f"Safety evaluated: {result['level']}")
            return web.json_response({
                "timestamp": time.time(),
                "status": result
            })
        except Exception as e:
            log("ERROR", self.node_name, str(e))
            return web.json_response({"error": str(e)}, status=400)

    async def status_handler(self, request: web.Request) -> web.Response:
        return web.json_response({
            "timestamp": time.time(),
            "status": self.current_status
        })

    # ------------------------------------------------------------------ #
    # App                                                                #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/safety/evaluate", self.safety_check_handler),
            web.get("/safety/status", self.status_handler),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(description="Sentience 5.5 – System Safety Node")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()

    node = SystemSafetyNode()

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
