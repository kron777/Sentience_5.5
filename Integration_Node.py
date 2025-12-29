#!/usr/bin/env python3
"""
IntegrationNode – Updated (Sentience 5.5 compliant)

Applied updates:
- Deterministic integration cycle (no implicit overwrites)
- Timestamped, versioned integration frames
- Evolver-compatible (clear inputs/outputs, no hidden coupling)
- Async-safe queue handling
- Explicit schema normalization
- No ROS, no magic globals
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Dict, Any, Optional

from aiohttp import web

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("Integration_Node")

# --------------------------------------------------------------------------- #
# Integration Node                                                            #
# --------------------------------------------------------------------------- #
class IntegrationNode:
    """
    Central integration hub.
    Collects node outputs → normalizes → integrates → emits atomic frame.
    """

    def __init__(self) -> None:
        self.node_outputs: Dict[str, Dict[str, Any]] = {}
        self.integration_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.integration_version: int = 0

    # ---------------- HTTP subscriber endpoints ---------------- #
    async def _handle_input(self, key: str, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        self.node_outputs[key] = {
            "data": data,
            "timestamp": time.time(),
        }
        logger.info("Received %s input", key)
        self._integrate()
        return web.json_response({"status": "accepted", "node": key})

    async def handle_decision(self, request: web.Request) -> web.Response:
        return await self._handle_input("decision", request)

    async def handle_learning(self, request: web.Request) -> web.Response:
        return await self._handle_input("learning", request)

    async def handle_communication(self, request: web.Request) -> web.Response:
        return await self._handle_input("communication", request)

    async def handle_monitoring(self, request: web.Request) -> web.Response:
        return await self._handle_input("monitoring", request)

    async def handle_adaptation(self, request: web.Request) -> web.Response:
        return await self._handle_input("adaptation", request)

    # ---------------- Core integration logic ---------------- #
    def _integrate(self) -> None:
        if not self.node_outputs:
            return

        self.integration_version += 1

        frame: Dict[str, Any] = {
            "version": self.integration_version,
            "timestamp": time.time(),
            "status": "integrated",
            "components": {},
            "final_action": {
                "action": "wait",
                "priority": "low",
            },
        }

        # normalize components
        for key, payload in self.node_outputs.items():
            frame["components"][key] = payload["data"]

        # decision dominates action
        decision = self.node_outputs.get("decision", {}).get("data", {})
        frame["final_action"]["action"] = decision.get("action", "wait")

        # priority resolution
        priorities = []
        for payload in self.node_outputs.values():
            pr = payload["data"].get("priority")
            if pr:
                priorities.append(pr)

        if priorities:
            frame["final_action"]["priority"] = max(
                priorities, key=lambda p: ["low", "medium", "high", "critical"].index(p)
                if p in ["low", "medium", "high", "critical"] else 0
            )

        self.integration_queue.put_nowait(frame)
        logger.info("Integrated frame v%s emitted", self.integration_version)

    # ---------------- Output endpoint ---------------- #
    async def handle_integration_output(self, request: web.Request) -> web.Response:
        try:
            frame = await asyncio.wait_for(self.integration_queue.get(), timeout=30)
            return web.json_response(frame)
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ---------------- App builder ---------------- #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/decision_making_output", self.handle_decision),
            web.post("/learning_output", self.handle_learning),
            web.post("/communication_output", self.handle_communication),
            web.post("/monitoring_output", self.handle_monitoring),
            web.post("/adaptation_output", self.handle_adaptation),
            web.get("/integration_output", self.handle_integration_output),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentience 5.5 – IntegrationNode")
    p.add_argument("--serve", action="store_true", help="Run HTTP service")
    p.add_argument("--port", type=int, default=8087, help="HTTP port")
    return p


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
async def amain() -> None:
    args = build_parser().parse_args()
    node = IntegrationNode()

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        logger.info("IntegrationNode running on :%d", args.port)
        await asyncio.Event().wait()
    else:
        logger.error("Nothing to do – use --serve")


if __name__ == "__main__":
    asyncio.run(amain())
