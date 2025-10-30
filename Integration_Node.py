#!/usr/bin/env python3
"""
IntegrationNode – ROS-free, asyncio-first
Same topics & JSON shapes, but HTTP/CLI instead of rospy.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import aiohttp
from aiohttp import web

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("Integration-Free")

# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
class IntegrationNode:
    """
    Async, ROS-free integrator.
    Subscribers -> HTTP POST endpoints
    Publisher -> GET /integration_output
    """

    def __init__(self) -> None:
        self.node_outputs: Dict[str, Dict] = {}
        self.integration_queue: asyncio.Queue[str] = asyncio.Queue()

    # ---------- subscriber callbacks (HTTP handlers) ---------- #
    async def handle_decision(self, request: web.Request) -> web.Response:
        data = await request.json()
        self.node_outputs["decision_making"] = data
        logger.info("Received decision data")
        self.integrate_outputs()
        return web.json_response({"status": "received"})

    async def handle_learning(self, request: web.Request) -> web.Response:
        data = await request.json()
        self.node_outputs["learning"] = data
        logger.info("Received learning data")
        self.integrate_outputs()
        return web.json_response({"status": "received"})

    async def handle_communication(self, request: web.Request) -> web.Response:
        data = await request.json()
        self.node_outputs["communication"] = data
        logger.info("Received communication data")
        self.integrate_outputs()
        return web.json_response({"status": "received"})

    async def handle_monitoring(self, request: web.Request) -> web.Response:
        data = await request.json()
        self.node_outputs["monitoring"] = data
        logger.info("Received monitoring data")
        self.integrate_outputs()
        return web.json_response({"status": "received"})

    async def handle_adaptation(self, request: web.Request) -> web.Response:
        data = await request.json()
        self.node_outputs["adaptation"] = data
        logger.info("Received adaptation data")
        self.integrate_outputs()
        return web.json_response({"status": "received"})

    # ---------- integration logic (unchanged) ---------- #
    def integrate_outputs(self) -> None:
        if not self.node_outputs:
            logger.warning("No outputs received for integration")
            return

        try:
            integrated_response = {"status": "integrated", "components": {}}

            if "decision_making" in self.node_outputs:
                integrated_response["components"]["decision"] = self.node_outputs["decision_making"]
            if "learning" in self.node_outputs:
                integrated_response["components"]["suggestion"] = self.node_outputs["learning"].get("suggestion", "none")
            if "communication" in self.node_outputs:
                integrated_response["components"]["last_message"] = self.node_outputs["communication"].get("message", {})
            if "monitoring" in self.node_outputs:
                integrated_response["components"]["system_status"] = self.node_outputs["monitoring"].get("status", "unknown")
            if "adaptation" in self.node_outputs:
                integrated_response["components"]["strategy"] = self.node_outputs["adaptation"].get("strategy", "default")

            priority = max(
                (self.node_outputs.get(n, {}).get("priority", "low") for n in self.node_outputs),
                default="low",
            )
            integrated_response["final_action"] = {
                "priority": priority,
                "action": self.node_outputs.get("decision_making", {}).get("action", "wait"),
            }

            # push to queue so GET /integration_output can stream it
            self.integration_queue.put_nowait(json.dumps(integrated_response))
            logger.info("Integrated response published: %s", json.dumps(integrated_response))
        except Exception as e:
            logger.error("Error integrating outputs: %s", e)

    # ---------- publisher (GET stream) ---------- #
    async def handle_integration_output(self, request: web.Request) -> web.Response:
        """
        Long-poll style: wait for next integrated message and return it.
        """
        try:
            msg = await asyncio.wait_for(self.integration_queue.get(), timeout=30)
            return web.json_response(json.loads(msg))
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ---------- app builder ---------- #
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
    p = argparse.ArgumentParser(description="Sentience 5.0 – IntegrationNode ROS-free")
    p.add_argument("--serve", action="store_true", help="run HTTP service")
    p.add_argument("--port", type=int, default=8087, help="HTTP port")
    p.add_argument("--test", action="store_true", help="inject test messages and show integration")
    return p


# --------------------------------------------------------------------------- #
# Test injector (optional)                                                    #
# --------------------------------------------------------------------------- #
async def inject_test_messages(node: IntegrationNode) -> None:
    await asyncio.sleep(0.5)
    test_msgs = [
        ("decision_making_output", {"action": "move", "priority": "high"}),
        ("learning_output", {"suggestion": "reduce_speed"}),
        ("communication_output", {"message": "user said hello"}),
        ("monitoring_output", {"status": "ok"}),
        ("adaptation_output", {"strategy": "eco"}),
    ]
    for topic, payload in test_msgs:
        node.node_outputs[topic.split("_output")[0]] = payload
        node.integrate_outputs()
        await asyncio.sleep(0.2)

    # wait for integration output
    try:
        integrated = await asyncio.wait_for(node.integration_queue.get(), timeout=5)
        print("Test integration result:")
        print(json.dumps(json.loads(integrated), indent=2))
    except asyncio.TimeoutError:
        print("No integration output received")


# --------------------------------------------------------------------------- #
# Entry-point                                                               #
# --------------------------------------------------------------------------- #
async def amain() -> None:
    args = build_parser().parse_args()
    node = IntegrationNode()

    if args.test:
        await inject_test_messages(node)
        return

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        logger.info("Integration HTTP service on :%d", args.port)
        await asyncio.Event().wait()  # run forever
    else:
        logger.error("Nothing to do – use --test or --serve")


if __name__ == "__main__":
    asyncio.run(amain())
