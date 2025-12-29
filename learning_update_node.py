#!/usr/bin/env python3
"""
LearningUpdateNode – Updated (Sentience 5.5 compliant)

Applied updates:
- Deterministic, versioned learning updates
- Bounded learning-rate dynamics (no runaway drift)
- Evolver-compatible metrics export
- Explicit update provenance
- Async-safe queues, no silent overwrites
- ROS-free, HTTP-first
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Dict, Any

from aiohttp import web

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("LearningUpdate_Node")

# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
class LearningUpdateNode:
    """
    Maintains and evolves a global learning-rate signal.
    """

    def __init__(self) -> None:
        self.learning_rate: float = 0.1
        self.min_lr: float = 0.05
        self.max_lr: float = 0.2

        self.update_version: int = 0
        self.update_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        # simple stability metric for evolver
        self.recent_deltas: list[float] = []

    # ---------------- HTTP inputs ---------------- #
    async def handle_feedback(self, request: web.Request) -> web.Response:
        data = await self._safe_json(request)
        if data is None:
            return web.json_response({"error": "invalid_json"}, status=400)

        self._apply_feedback(data)
        return web.json_response({"status": "accepted"})

    async def handle_optimization(self, request: web.Request) -> web.Response:
        data = await self._safe_json(request)
        if data is None:
            return web.json_response({"error": "invalid_json"}, status=400)

        self._apply_optimization(data)
        return web.json_response({"status": "accepted"})

    async def handle_memory(self, request: web.Request) -> web.Response:
        data = await self._safe_json(request)
        if data is None:
            return web.json_response({"error": "invalid_json"}, status=400)

        self._apply_memory(data)
        return web.json_response({"status": "accepted"})

    async def _safe_json(self, request: web.Request) -> Dict[str, Any] | None:
        try:
            return await request.json()
        except Exception:
            logger.warning("Invalid JSON received")
            return None

    # ---------------- Core logic ---------------- #
    def _emit_update(self, source: str, delta: float) -> None:
        self.update_version += 1
        self.learning_rate = max(self.min_lr, min(self.max_lr, self.learning_rate))

        self.recent_deltas.append(delta)
        if len(self.recent_deltas) > 10:
            self.recent_deltas.pop(0)

        frame = {
            "version": self.update_version,
            "timestamp": time.time(),
            "learning_rate": self.learning_rate,
            "delta": delta,
            "source": source,
            "stability": round(1.0 - (sum(abs(d) for d in self.recent_deltas) / len(self.recent_deltas)), 3)
            if self.recent_deltas else 1.0,
        }

        self.update_queue.put_nowait(frame)
        logger.info("Learning update v%s emitted (lr=%.3f)", self.update_version, self.learning_rate)

    def _apply_feedback(self, feedback: Dict[str, Any]) -> None:
        delta = 0.01 if feedback.get("success") else -0.01
        self.learning_rate += delta
        self._emit_update("feedback", delta)

    def _apply_optimization(self, optimization: Dict[str, Any]) -> None:
        delta = 0.02 if optimization.get("priority") == "high" else 0.0
        self.learning_rate += delta
        self._emit_update("optimization", delta)

    def _apply_memory(self, memory: Dict[str, Any]) -> None:
        delta = -0.01 if memory.get("total_entries", 0) > 80 else 0.0
        self.learning_rate += delta
        self._emit_update("memory", delta)

    # ---------------- Output ---------------- #
    async def handle_learning_update(self, request: web.Request) -> web.Response:
        try:
            frame = await asyncio.wait_for(self.update_queue.get(), timeout=30)
            return web.json_response(frame)
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ---------------- App builder ---------------- #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/feedback_input", self.handle_feedback),
            web.post("/optimization_suggestions", self.handle_optimization),
            web.post("/memory_status", self.handle_memory),
            web.get("/learning_update", self.handle_learning_update),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentience 5.5 – LearningUpdateNode")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8090)
    return p


async def amain() -> None:
    args = build_parser().parse_args()
    node = LearningUpdateNode()

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        logger.info("LearningUpdateNode running on :%d", args.port)
        await asyncio.Event().wait()
    else:
        logger.error("Use --serve")


if __name__ == "__main__":
    asyncio.run(amain())
