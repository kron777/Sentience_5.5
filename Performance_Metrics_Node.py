#!/usr/bin/env python3
"""
PerformanceMetricsNode – Sentience 5.5 compliant

Role:
- Aggregate raw system metrics
- Compute deterministic KPIs
- Detect degradation and suboptimal states
- Publish factual performance reports only
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import sqlite3
import argparse
from datetime import datetime
from typing import Dict, Any, List
from aiohttp import web
import asyncio


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# Performance Metrics Node                                                    #
# --------------------------------------------------------------------------- #
class PerformanceMetricsNode:
    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "performance_metrics_node"
        self.db_path = os.path.join(db_root, "performance_metrics.db")
        os.makedirs(db_root, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # Rolling metric buffer
        self.system_metrics: List[Dict[str, Any]] = []

        # Last report
        self.last_report: Dict[str, Any] | None = None

        log("INFO", self.node_name, "PerformanceMetricsNode initialized")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_reports (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                overall_score REAL,
                suboptimal BOOLEAN,
                kpis_json TEXT,
                degradation_flags_json TEXT
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Metric ingestion                                                   #
    # ------------------------------------------------------------------ #
    def ingest_metric(self, metric: Dict[str, Any]):
        """
        Expected metric format:
        {
          "timestamp": float,
          "name": str,
          "value": float
        }
        """
        self.system_metrics.append(metric)

        # Keep last 60 seconds
        cutoff = time.time() - 60.0
        self.system_metrics = [
            m for m in self.system_metrics
            if m.get("timestamp", 0) >= cutoff
        ]

    # ------------------------------------------------------------------ #
    # KPI computation                                                    #
    # ------------------------------------------------------------------ #
    def compute_kpis(self) -> Dict[str, float]:
        latency = []
        errors = []
        cpu = []

        for m in self.system_metrics:
            name = m.get("name")
            value = m.get("value", 0)

            if "latency" in name:
                latency.append(value)
            elif "error" in name:
                errors.append(value)
            elif "cpu" in name:
                cpu.append(value)

        kpis = {
            "latency_avg_ms": sum(latency) / len(latency) if latency else 0.0,
            "error_rate": sum(errors),
            "cpu_util_avg": sum(cpu) / len(cpu) if cpu else 0.0,
            "metric_count": len(self.system_metrics),
        }
        return kpis

    # ------------------------------------------------------------------ #
    # Scoring                                                            #
    # ------------------------------------------------------------------ #
    def score_performance(self, kpis: Dict[str, float]) -> Dict[str, Any]:
        score = 1.0
        flags = []

        if kpis["latency_avg_ms"] > 120:
            score -= 0.25
            flags.append("high_latency")

        if kpis["error_rate"] > 0:
            score -= 0.4
            flags.append("errors_present")

        if kpis["cpu_util_avg"] > 0.85:
            score -= 0.2
            flags.append("cpu_pressure")

        score = max(0.0, min(1.0, score))

        return {
            "overall_score": score,
            "suboptimal": score < 0.7,
            "flags": flags,
        }

    # ------------------------------------------------------------------ #
    # Report generation                                                  #
    # ------------------------------------------------------------------ #
    def generate_report(self) -> Dict[str, Any]:
        kpis = self.compute_kpis()
        assessment = self.score_performance(kpis)

        report = {
            "timestamp": time.time(),
            "overall_score": assessment["overall_score"],
            "suboptimal": assessment["suboptimal"],
            "kpis": kpis,
            "degradation_flags": assessment["flags"],
        }

        self._persist(report)
        self.last_report = report

        log("INFO", self.node_name, f"Performance report generated: score={report['overall_score']:.2f}")
        return report

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def _persist(self, report: Dict[str, Any]):
        self.conn.execute(
            """
            INSERT INTO performance_reports
            (id, timestamp, overall_score, suboptimal, kpis_json, degradation_flags_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                report["timestamp"],
                report["overall_score"],
                report["suboptimal"],
                json.dumps(report["kpis"]),
                json.dumps(report["degradation_flags"]),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # HTTP API                                                           #
    # ------------------------------------------------------------------ #
    async def ingest_handler(self, request: web.Request) -> web.Response:
        metric = await request.json()
        self.ingest_metric(metric)
        return web.json_response({"status": "accepted"})

    async def report_handler(self, request: web.Request) -> web.Response:
        report = self.generate_report()
        return web.json_response(report)

    async def latest_handler(self, request: web.Request) -> web.Response:
        return web.json_response(self.last_report or {"status": "none"})

    # ------------------------------------------------------------------ #
    # App                                                                #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/performance/metric", self.ingest_handler),
            web.get("/performance/report", self.report_handler),
            web.get("/performance/latest", self.latest_handler),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(description="Sentience 5.5 – Performance Metrics Node")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8096)
    args = parser.parse_args()

    node = PerformanceMetricsNode()

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        log("INFO", node.node_name, f"Running on :{args.port}")
        await asyncio.Event().wait()
    else:
        log("ERROR", node.node_name, "Use --serve")


if __name__ == "__main__":
    asyncio.run(amain())
