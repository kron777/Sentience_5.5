#!/usr/bin/env python3
"""
OptimizationNode – Sentience 5.5 compliant

Purpose:
- Observe system performance signals
- Detect inefficiencies, overload, or coherence loss
- Emit optimization *suggestions only*
- No execution authority
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
# Optimization Node                                                           #
# --------------------------------------------------------------------------- #
class OptimizationNode:
    """
    Produces optimization recommendations based on system state.
    """

    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "optimization_node"
        self.db_path = os.path.join(db_root, "optimization.db")
        os.makedirs(db_root, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        self.suggestion_queue: asyncio.Queue[str] = asyncio.Queue()
        self.latest_suggestion: Dict[str, Any] | None = None

        log("INFO", self.node_name, "OptimizationNode initialized")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS optimization (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                category TEXT,
                action TEXT,
                priority TEXT,
                reason TEXT,
                source TEXT
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Core analysis                                                       #
    # ------------------------------------------------------------------ #
    def analyze(self, payload: Dict[str, Any], source: str):
        """
        Expected inputs (examples):

        Monitoring:
        { "status": "alert", "avg_confidence": 0.42 }

        Memory:
        { "total_entries": 112, "capacity": 120 }

        Meta-awareness:
        { "coherence": 0.38 }
        """

        suggestion = None

        # --- Performance degradation ---
        if source == "monitoring":
            if payload.get("status") == "alert":
                suggestion = self._suggest(
                    category="performance",
                    action="reallocate_resources",
                    priority="high",
                    reason="System monitoring reports alert state",
                    source=source,
                )

        # --- Memory pressure ---
        elif source == "memory":
            total = payload.get("total_entries", 0)
            capacity = payload.get("capacity", 100)
            if total / max(capacity, 1) > 0.85:
                suggestion = self._suggest(
                    category="memory",
                    action="prune_old_entries",
                    priority="medium",
                    reason=f"Memory usage high ({total}/{capacity})",
                    source=source,
                )

        # --- Coherence loss ---
        elif source == "meta_awareness":
            coherence = payload.get("coherence", 1.0)
            if coherence < 0.5:
                suggestion = self._suggest(
                    category="coherence",
                    action="recalibrate_internal_models",
                    priority="high",
                    reason=f"Low coherence detected ({coherence:.2f})",
                    source=source,
                )

        if suggestion:
            self._emit(suggestion)

    # ------------------------------------------------------------------ #
    # Suggestion construction                                            #
    # ------------------------------------------------------------------ #
    def _suggest(self, category: str, action: str, priority: str, reason: str, source: str):
        suggestion = {
            "timestamp": time.time(),
            "category": category,
            "action": action,
            "priority": priority,
            "reason": reason,
            "source": source,
        }
        self._persist(suggestion)
        self.latest_suggestion = suggestion
        return suggestion

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def _persist(self, s: Dict[str, Any]):
        self.conn.execute(
            """
            INSERT INTO optimization
            (id, timestamp, category, action, priority, reason, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                s["timestamp"],
                s["category"],
                s["action"],
                s["priority"],
                s["reason"],
                s["source"],
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Emission                                                           #
    # ------------------------------------------------------------------ #
    def _emit(self, suggestion: Dict[str, Any]):
        self.suggestion_queue.put_nowait(json.dumps(suggestion))
        log("INFO", self.node_name, f"Optimization suggested: {json.dumps(suggestion)}")

    # ------------------------------------------------------------------ #
    # HTTP API                                                           #
    # ------------------------------------------------------------------ #
    async def handle_input(self, request: web.Request) -> web.Response:
        data = await request.json()
        source = request.match_info["source"]
        self.analyze(data, source)
        return web.json_response({"status": "received"})

    async def handle_latest(self, request: web.Request) -> web.Response:
        if self.latest_suggestion:
            return web.json_response(self.latest_suggestion)
        return web.json_response({"status": "none"})

    async def handle_stream(self, request: web.Request) -> web.Response:
        try:
            msg = await asyncio.wait_for(self.suggestion_queue.get(), timeout=30)
            return web.json_response(json.loads(msg))
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ------------------------------------------------------------------ #
    # App builder                                                        #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/optimization/input/{source}", self.handle_input),
            web.get("/optimization/latest", self.handle_latest),
            web.get("/optimization/stream", self.handle_stream),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(description="Sentience 5.5 – OptimizationNode")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()

    node = OptimizationNode()

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        log("INFO", node.node_name, f"OptimizationNode running on :{args.port}")
        await asyncio.Event().wait()
    else:
        log("ERROR", node.node_name, "Use --serve")


if __name__ == "__main__":
    asyncio.run(amain())
