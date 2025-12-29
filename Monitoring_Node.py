#!/usr/bin/env python3
"""
MonitoringNode – Updated (Sentience 5.5 compliant)

Purpose:
- Observe system health & node-level performance
- Detect degradation, instability, and regressions
- Emit alerts only (never actions)
- Deterministic metrics, evolver-visible
- ROS-free, HTTP-first, async-safe
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
from typing import Dict, Any, List
from collections import deque

from aiohttp import web

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# Monitoring Node                                                             #
# --------------------------------------------------------------------------- #
class MonitoringNode:
    """
    System-wide performance and stability monitor.
    """

    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "monitoring_node"
        self.db_path = os.path.join(db_root, "monitoring.db")
        os.makedirs(db_root, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # thresholds
        self.alert_threshold = 0.3

        # rolling buffers
        self.recent_metrics = deque(maxlen=100)
        self.alert_queue: asyncio.Queue[str] = asyncio.Queue()

        log("INFO", self.node_name, "MonitoringNode initialized")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS monitoring (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                source TEXT,
                metric_json TEXT,
                score REAL
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Metric ingestion                                                   #
    # ------------------------------------------------------------------ #
    def ingest_metric(self, source: str, metric: Dict[str, Any]) -> None:
        score = float(metric.get("confidence", metric.get("score", 1.0)))
        entry = {
            "timestamp": time.time(),
            "source": source,
            "metric": metric,
            "score": score,
        }
        self.recent_metrics.append(entry)
        self._persist(entry)

    # ------------------------------------------------------------------ #
    # Evaluation                                                         #
    # ------------------------------------------------------------------ #
    def evaluate(self) -> Dict[str, Any]:
        if not self.recent_metrics:
            return {"status": "idle"}

        window = list(self.recent_metrics)[-10:]
        avg_score = sum(e["score"] for e in window) / len(window)

        state = {
            "timestamp": time.time(),
            "average_score": round(avg_score, 3),
            "samples": len(window),
        }

        if avg_score < self.alert_threshold:
            self._emit_alert(avg_score)

        return state

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def _persist(self, entry: Dict[str, Any]):
        self.conn.execute(
            """
            INSERT INTO monitoring
            (id, timestamp, source, metric_json, score)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                entry["timestamp"],
                entry["source"],
                json.dumps(entry["metric"]),
                entry["score"],
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Alerts                                                             #
    # ------------------------------------------------------------------ #
    def _emit_alert(self, score: float):
        alert = {
            "status": "alert",
            "reason": "performance_degradation",
            "average_score": round(score, 3),
            "threshold": self.alert_threshold,
            "timestamp": time.time(),
        }
        self.alert_queue.put_nowait(json.dumps(alert))
        log("WARN", self.node_name, f"Alert emitted: {json.dumps(alert)}")

    # ------------------------------------------------------------------ #
    # HTTP API                                                           #
    # ------------------------------------------------------------------ #
    async def handle_metric(self, request: web.Request) -> web.Response:
        d = await request.json()
        self.ingest_metric(d.get("source", "unknown"), d.get("metric", {}))
        return web.json_response({"status": "ok"})

    async def handle_state(self, request: web.Request) -> web.Response:
        return web.json_response(self.evaluate())

    async def handle_alert(self, request: web.Request) -> web.Response:
        try:
            msg = await asyncio.wait_for(self.alert_queue.get(), timeout=30)
            return web.json_response(json.loads(msg))
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ------------------------------------------------------------------ #
    # App builder                                                        #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/monitor/metric", self.handle_metric),
            web.get("/monitor/state", self.handle_state),
            web.get("/monitor/alert", self.handle_alert),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(description="Sentience 5.5 – MonitoringNode")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args()

    node = MonitoringNode()

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        log("INFO", node.node_name, f"MonitoringNode running on :{args.port}")
        await asyncio.Event().wait()
    else:
        log("ERROR", node.node_name, "Use --serve")


if __name__ == "__main__":
    asyncio.run(amain())
