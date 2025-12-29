#!/usr/bin/env python3
"""
SelfCorrectionNode – Sentience 5.5 compliant (Safe Mode)

Role:
- Detect internal issues
- Propose corrective directives (text-only)
- Require audit approval
- NEVER self-modify code automatically
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import asyncio
import sqlite3
import argparse
import random
from datetime import datetime
from typing import Dict, Any
from aiohttp import web, ClientSession

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# Self-Correction Node                                                        #
# --------------------------------------------------------------------------- #
class SelfCorrectionNode:
    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "self_correction_node"
        self.db_path = os.path.join(db_root, "self_correction.db")
        os.makedirs(db_root, exist_ok=True)

        self.pending_directives: asyncio.Queue = asyncio.Queue()
        self.audit_queue: asyncio.Queue = asyncio.Queue()

        self.ethical_compassion_bias = 0.2
        self.session: ClientSession | None = None

        self._init_db()
        log("INFO", self.node_name, "SelfCorrectionNode initialized (SAFE MODE)")

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS self_corrections (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                issue TEXT,
                directive TEXT,
                audit_status TEXT,
                notes TEXT
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Core Logic                                                         #
    # ------------------------------------------------------------------ #
    async def generate_directive(self, issue: str) -> str:
        """
        Generates a SAFE, TEXT-ONLY corrective suggestion.
        """
        await asyncio.sleep(0.1)  # placeholder for LLM latency

        directive = (
            f"Observed issue: {issue}\n"
            f"Suggested action:\n"
            f"- Investigate root cause\n"
            f"- Apply minimal, reversible change\n"
            f"- Add monitoring\n\n"
            f"Compassionate note: treat this as learning, not failure."
        )
        return directive

    async def handle_issue(self, issue: str):
        directive = await self.generate_directive(issue)
        payload = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "issue": issue,
            "directive": directive,
            "audit_status": "pending",
        }
        await self.pending_directives.put(payload)
        log("INFO", self.node_name, "Directive generated and queued for audit")

    async def handle_audit_result(self, directive_id: str, approved: bool):
        status = "approved" if approved else "rejected"
        self.conn.execute(
            """
            UPDATE self_corrections
            SET audit_status = ?
            WHERE id = ?
            """,
            (status, directive_id),
        )
        self.conn.commit()

        log("INFO", self.node_name, f"Audit result: {directive_id} → {status}")

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def store_directive(self, payload: Dict[str, Any]):
        self.conn.execute(
            """
            INSERT INTO self_corrections
            (id, timestamp, issue, directive, audit_status, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                payload["timestamp"],
                payload["issue"],
                payload["directive"],
                payload["audit_status"],
                "",
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # HTTP Handlers                                                      #
    # ------------------------------------------------------------------ #
    async def issue_handler(self, request: web.Request) -> web.Response:
        data = await request.json()
        issue = data.get("issue", "unspecified anomaly")
        await self.handle_issue(issue)
        return web.json_response({"status": "queued", "issue": issue})

    async def next_directive_handler(self, request: web.Request) -> web.Response:
        try:
            directive = await asyncio.wait_for(self.pending_directives.get(), timeout=10)
            self.store_directive(directive)
            return web.json_response(directive)
        except asyncio.TimeoutError:
            return web.json_response({"status": "none"})

    async def audit_handler(self, request: web.Request) -> web.Response:
        data = await request.json()
        await self.handle_audit_result(
            data["directive_id"], data.get("approved", False)
        )
        return web.json_response({"status": "recorded"})

    async def health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "node": self.node_name,
            "time": time.time(),
        })

    # ------------------------------------------------------------------ #
    # App                                                               #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/self_correction/issue", self.issue_handler),
            web.get("/self_correction/next", self.next_directive_handler),
            web.post("/self_correction/audit", self.audit_handler),
            web.get("/self_correction/health", self.health_handler),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(description="Sentience 5.5 – Self Correction Node")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()

    node = SelfCorrectionNode()

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
