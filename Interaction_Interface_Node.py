#!/usr/bin/env python3
"""
InteractionInterfaceNode – ROS-free, asyncio-first
Same topics & JSON shapes, but HTTP/CLI instead of rospy.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List

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
logger = logging.getLogger("Interaction-Free")

# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
class InteractionInterfaceNode:
    """
    Async, ROS-free interaction handler.
    Subscriber -> HTTP POST /control_output
    Publisher  -> GET /interaction_response
    """

    def __init__(self) -> None:
        self.interaction_log: List[Dict] = []
        self.response_queue: asyncio.Queue[str] = asyncio.Queue()

    # ---------- subscriber callback (HTTP handler) ---------- #
    async def handle_control_output(self, request: web.Request) -> web.Response:
        try:
            control_data = await request.json()
            logger.info("Received control data")
            self.handle_interaction(control_data)
            return web.json_response({"status": "received"})
        except Exception as e:
            logger.error("Error processing control data: %s", e)
            return web.json_response({"status": "error", "error": str(e)}, status=400)

    # ---------- interaction logic (unchanged) ---------- #
    def handle_interaction(self, control_data: Dict) -> None:
        try:
            action = control_data.get("action", "idle")
            response = {"timestamp": time.time(), "action": action, "response": "acknowledged"}

            if action == "respond_emotionally":
                response["response"] = "Expressing appropriate emotion"
            elif action == "execute_task":
                response["response"] = "Task execution in progress"

            self.interaction_log.append(response)
            self.response_queue.put_nowait(json.dumps(response))
            logger.info("Handled interaction: %s", json.dumps(response))
        except Exception as e:
            logger.error("Error handling interaction: %s", e)

    # ---------- publisher (GET stream) ---------- #
    async def handle_interaction_response(self, request: web.Request) -> web.Response:
        try:
            msg = await asyncio.wait_for(self.response_queue.get(), timeout=30)
            return web.json_response(json.loads(msg))
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ---------- app builder ---------- #
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
    p = argparse.ArgumentParser(description="Sentience 5.0 – InteractionInterfaceNode ROS-free")
    p.add_argument("--serve", action="store_true", help="run HTTP service")
    p.add_argument("--port", type=int, default=8088, help="HTTP port")
    p.add_argument("--test", action="store_true", help="inject test message and show response")
    return p


# --------------------------------------------------------------------------- #
# Test injector (optional)                                                    #
# --------------------------------------------------------------------------- #
async def inject_test_message(node: InteractionInterfaceNode) -> None:
    await asyncio.sleep(0.5)
    test_payload = {"action": "respond_emotionally"}
    node.handle_interaction(test_payload)
    await asyncio.sleep(0.2)
    try:
        response = await asyncio.wait_for(node.response_queue.get(), timeout=5)
        print("Test interaction response:")
        print(json.dumps(json.loads(response), indent=2))
    except asyncio.TimeoutError:
        print("No response received")


# --------------------------------------------------------------------------- #
# Entry-point                                                               #
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
        logger.info("Interaction HTTP service on :%d", args.port)
        await asyncio.Event().wait()  # run forever
    else:
        logger.error("Nothing to do – use --test or --serve")


if __name__ == "__main__":
    asyncio.run(amain())
