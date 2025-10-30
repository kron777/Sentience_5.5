#!/usr/bin/env python3
"""
LearningUpdateNode – ROS-free, asyncio-first
Same topics & JSON shapes, but HTTP/CLI instead of rospy.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
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
logger = logging.getLogger("LearningUpdate-Free")

# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
class LearningUpdateNode:
    """
    Async, ROS-free learning-rate updater.
    Subscribers -> HTTP POST endpoints
    Publisher -> GET /learning_update
    """

    def __init__(self) -> None:
        self.learning_rate = 0.1
        self.update_queue: asyncio.Queue[str] = asyncio.Queue()

    # ---------- subscriber handlers (HTTP POST) ---------- #
    async def handle_feedback(self, request: web.Request) -> web.Response:
        try:
            feedback_data = await request.json()
            logger.info("Received feedback data")
            self.update_learning_from_feedback(feedback_data)
            return web.json_response({"status": "received"})
        except Exception as e:
            logger.error("Error processing feedback: %s", e)
            return web.json_response({"status": "error", "error": str(e)}, status=400)

    async def handle_optimization(self, request: web.Request) -> web.Response:
        try:
            optimization_data = await request.json()
            logger.info("Received optimization data")
            self.update_learning_from_optimization(optimization_data)
            return web.json_response({"status": "received"})
        except Exception as e:
            logger.error("Error processing optimization: %s", e)
            return web.json_response({"status": "error", "error": str(e)}, status=400)

    async def handle_memory(self, request: web.Request) -> web.Response:
        try:
            memory_data = await request.json()
            logger.info("Received memory data")
            self.adjust_learning_rate(memory_data)
            return web.json_response({"status": "received"})
        except Exception as e:
            logger.error("Error processing memory: %s", e)
            return web.json_response({"status": "error", "error": str(e)}, status=400)

    # ---------- core logic (unchanged) ---------- #
    def update_learning_from_feedback(self, feedback_data: Dict[str, Any]) -> None:
        try:
            if feedback_data.get("success", False):
                self.learning_rate = min(0.2, self.learning_rate + 0.01)
            else:
                self.learning_rate = max(0.05, self.learning_rate - 0.01)
            update = {"learning_rate": self.learning_rate, "source": "feedback"}
            self.publish_update(update)
        except Exception as e:
            logger.error("Error updating from feedback: %s", e)

    def update_learning_from_optimization(self, optimization_data: Dict[str, Any]) -> None:
        try:
            if optimization_data.get("priority", "low") == "high":
                self.learning_rate = min(0.2, self.learning_rate + 0.02)
            update = {"learning_rate": self.learning_rate, "source": "optimization"}
            self.publish_update(update)
        except Exception as e:
            logger.error("Error updating from optimization: %s", e)

    def adjust_learning_rate(self, memory_data: Dict[str, Any]) -> None:
        try:
            if memory_data.get("total_entries", 0) > 80:
                self.learning_rate = max(0.05, self.learning_rate - 0.01)
            update = {"learning_rate": self.learning_rate, "source": "memory"}
            self.publish_update(update)
        except Exception as e:
            logger.error("Error adjusting learning rate: %s", e)

    def publish_update(self, update: Dict[str, Any]) -> None:
        try:
            payload = json.dumps(update)
            self.update_queue.put_nowait(payload)
            logger.info("Published learning update: %s", payload)
        except Exception as e:
            logger.error("Error publishing update: %s", e)

    # ---------- publisher (GET stream) ---------- #
    async def handle_learning_update(self, request: web.Request) -> web.Response:
        try:
            msg = await asyncio.wait_for(self.update_queue.get(), timeout=30)
            return web.json_response(json.loads(msg))
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ---------- app builder ---------- #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes(
            [
                web.post("/feedback_input", self.handle_feedback),
                web.post("/optimization_suggestions", self.handle_optimization),
                web.post("/memory_status", self.handle_memory),
                web.get("/learning_update", self.handle_learning_update),
            ]
        )
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentience 5.0 – LearningUpdateNode ROS-free")
    p.add_argument("--serve", action="store_true", help="run HTTP service")
    p.add_argument("--port", type=int, default=8090, help="HTTP port")
    p.add_argument("--test", action="store_true", help="inject test messages and show updates")
    return p


# --------------------------------------------------------------------------- #
# Test injector (optional)                                                    #
# --------------------------------------------------------------------------- #
async def inject_test_messages(node: LearningUpdateNode) -> None:
    await asyncio.sleep(0.5)
    test_msgs = [
        ("/feedback_input", {"success": True}),
        ("/optimization_suggestions", {"priority": "high"}),
        ("/memory_status", {"total_entries": 85}),
    ]
    for topic, payload in test_msgs:
        async with aiohttp.ClientSession() as session:
            await session.post(f"http://localhost:8090{topic}", json=payload)
        await asyncio.sleep(0.2)

    # wait for updates
    for _ in range(3):
        try:
            update = await asyncio.wait_for(node.update_queue.get(), timeout=5)
            print("Learning update:")
            print(json.dumps(json.loads(update), indent=2))
        except asyncio.TimeoutError:
            print("No update received")


# --------------------------------------------------------------------------- #
# Entry-point                                                               #
# --------------------------------------------------------------------------- #
async def amain() -> None:
    args = build_parser().parse_args()
    node = LearningUpdateNode()

    if args.test:
        await inject_test_messages(node)
        return

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        logger.info("LearningUpdate HTTP service on :%d", args.port)
        await asyncio.Event().wait()  # run forever
    else:
        logger.error("Nothing to do – use --test or --serve")


if __name__ == "__main__":
    asyncio.run(amain())
