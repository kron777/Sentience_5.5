#!/usr/bin/env python3
"""
PerformanceMonitorNode – Sentience 5.5 compliant

Role:
- Collect raw system performance metrics
- Zero interpretation, zero bias
- Serve metrics to PerformanceMetricsNode
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import psutil
import sqlite3
import argparse
import asyncio
import signal
from datetime import datetime
from typing import Dict, Any, Optional
from aiohttp import web


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(
        f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}",
        file=sys.stdout,
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Performance Monitor Node                                                    #
# --------------------------------------------------------------------------- #
class PerformanceMonitorNode:
    SCHEMA_VERSION = 1

    def __init__(self, db_root: str = "/tmp/sentience_db", interval: float = 1.0):
        self.node_name = "performance_monitor_node"
        self.interval = interval
        self.db_path = os.path.join(db_root, "performance_monitor.db")
        os.makedirs(db_root, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_db()

        self._last_net = psutil.net_io_counters()
        self._latest_snapshot: Optional[Dict[str, Any]] = None
        self._shutdown = asyncio.Event()

        log("INFO", self.node_name, "PerformanceMonitorNode initialized")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_metrics (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                metric_name TEXT,
                value REAL,
                unit TEXT
            )
        """)
        self.conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION)),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Metric collection                                                  #
    # ------------------------------------------------------------------ #
    def collect_metrics(self) -> Dict[str, Dict[str, float]]:
        now = time.time()

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent

        net = psutil.net_io_counters()
        sent_rate = net.bytes_sent - self._last_net.bytes_sent
        recv_rate = net.bytes_recv - self._last_net.bytes_recv
        self._last_net = net

        metrics = {
            "cpu_percent": {"value": cpu, "unit": "%"},
            "memory_percent": {"value": mem, "unit": "%"},
            "disk_percent": {"value": disk, "unit": "%"},
            "net_bytes_sent_per_s": {"value": sent_rate, "unit": "bytes/s"},
            "net_bytes_recv_per_s": {"value": recv_rate, "unit": "bytes/s"},
        }

        self._latest_snapshot = {
            "timestamp": now,
            "metrics": metrics,
        }

        return metrics

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def persist_metrics(self, metrics: Dict[str, Dict[str, float]]):
        ts = time.time()
        rows = [
            (
                str(uuid.uuid4()),
                ts,
                name,
                payload["value"],
                payload["unit"],
            )
            for name, payload in metrics.items()
        ]
        self.conn.executemany(
            """
            INSERT INTO raw_metrics
            (id, timestamp, metric_name, value, unit)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Background sampler                                                 #
    # ------------------------------------------------------------------ #
    async def sampler_loop(self):
        log("INFO", self.node_name, "Background sampler started")
        while not self._shutdown.is_set():
            metrics = self.collect_metrics()
            self.persist_metrics(metrics)
            await asyncio.sleep(self.interval)
        log("INFO", self.node_name, "Background sampler stopped")

    # ------------------------------------------------------------------ #
    # HTTP Handlers                                                      #
    # ------------------------------------------------------------------ #
    async def snapshot_handler(self, request: web.Request) -> web.Response:
        metrics = self.collect_metrics()
        self.persist_metrics(metrics)
        return web.json_response(self._latest_snapshot)

    async def latest_handler(self, request: web.Request) -> web.Response:
        if not self._latest_snapshot:
            return web.json_response({"status": "no_data"}, status=404)
        return web.json_response(self._latest_snapshot)

    async def health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "node": self.node_name,
            "schema_version": self.SCHEMA_VERSION,
            "time": time.time(),
        })

    # ------------------------------------------------------------------ #
    # App                                                                #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.get("/performance/monitor/snapshot", self.snapshot_handler),
            web.get("/performance/monitor/latest", self.latest_handler),
            web.get("/performance/monitor/health", self.health_handler),
        ])
        return app

    # ------------------------------------------------------------------ #
    # Shutdown                                                           #
    # ------------------------------------------------------------------ #
    async def shutdown(self):
        log("INFO", self.node_name, "Shutdown requested")
        self._shutdown.set()
        await asyncio.sleep(0)
        self.conn.close()


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(
        description="Sentience 5.5 – Performance Monitor Node"
    )
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    node = PerformanceMonitorNode(interval=args.interval)

    if not args.serve:
        log("ERROR", node.node_name, "Use --serve")
        return

    app = node.build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    sampler_task = asyncio.create_task(node.sampler_loop())

    log("INFO", node.node_name, f"Serving on :{args.port}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(
            sig, lambda: asyncio.create_task(node.shutdown())
        )

    await node._shutdown.wait()
    sampler_task.cancel()


if __name__ == "__main__":
    asyncio.run(amain())
