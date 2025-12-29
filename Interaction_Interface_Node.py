#!/usr/bin/env python3
"""
InteractionInterfaceNode – Updated (Sentience 5.5 compliant)

Applied updates:
- Deterministic interaction frames
- Versioned + timestamped responses
- Evolver-compatible (explicit input/output contract)
- Async-safe queue usage
- Clear action → response mapping (no hidden logic)
- ROS-free, HTTP-first
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Dict, Any, List

from aiohttp import web

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("Interaction_Node")

# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
class InteractionInterfaceNode:
    """
    Receives control actions and emits interaction responses as atomic frames.
    """

    def __init__(self) -> None:
        self.interaction_log: List[Dict[str, Any]] = []
        self.response_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.response_version: int = 0

    # ---------------- HTTP input ---------------- #
    async def handle_control_output(self, request: web.Request) -> web.Response:
        try:
            control_data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        logger.info("Received control output")
        self._process_interaction(control_data)
        return web.json_response({"status": "accepted"})

    # ---------------- Core logic ---------------- #
    def _process_interaction(self, control_data: Dict[str, Any]) -> None:
        self.response_version += 1

        action = control_data.get("action", "idle")

        response_text = "acknowledged"
        if action == "respond_emotionally":
            response_text = "emotional_response_emitted"
        elif action == "execute_task":
            response_text = "task_execution_started"
        elif action == "wait":
            response_text = "standing_by"

        frame: Dict[str, Any] = {
            "version": self.response_version,
            "timestamp": time.time(),
            "action": action,
            "response": response_text,
            "status": "ok",
        }

        self.interaction_log.append(frame)
        self.response_queue.put_nowait(frame)

        logger.info("Interaction frame v%s emitted", self.response_version)

    # ---------------- Output ---------------- #
    async def handle_interaction_response(self, request: web.Request) -> web.Response:
        try:
            frame = await asyncio.wait_for(self.response_queue.get(), timeout=30)
            return web.json_response(frame)
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ---------------- App builder ---------------- #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/control_output", self.handle_control_output),
            web.get("/interaction_response", self.handle_interaction_response),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentience 5.5 – InteractionInterfaceNode")
    p.add_argument("--serve", action="store_true", help="Run HTTP service")
    p.add_argument("--port", type=int, default=8088, help="HTTP port")
    p.add_argument("--test", action="store_true", help="Run test interaction")
    return p


# --------------------------------------------------------------------------- #
# Test injector                                                               #
# --------------------------------------------------------------------------- #
async def inject_test_message(node: InteractionInterfaceNode) -> None:
    await asyncio.sleep(0.2)
    node._process_interaction({"action": "respond_emotionally"})
    frame = await node.response_queue.get()
    print(json.dumps(frame, indent=2))


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
async def amain() -> None:
    args = build_parser().parse_args()
    node = InteractionInterfaceNode()

    if args.test:
        await inject_test_message(node)
        return

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        logger.info("InteractionInterfaceNode running on :%d", args.port)
        await asyncio.Event().wait()
    else:
        logger.error("Nothing to do – use --serve or --test")


if __name__ == "__main__":
    asyncio.run(amain())
