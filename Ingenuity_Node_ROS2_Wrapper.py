#!/usr/bin/env python3
"""
IngenuityNode – ROS-free, asyncio-first
Same API as the original ROS wrapper, but HTTP/CLI instead of rclpy.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

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
logger = logging.getLogger("Ingenuity-ROS-Free")

# --------------------------------------------------------------------------- #
# Core logic (unchanged)                                                      #
# --------------------------------------------------------------------------- #
class IngenuityNodeCore:
    def __init__(self) -> None:
        self.generated_nodes: Dict[str, Any] = {}

    def generate_node_code(self, node_name: str, specification: str) -> str:
        return f'''
class {node_name}:
    def __init__(self):
        self.name = "{node_name}"

    def run(self):
        print("Running {node_name}: {specification}")
'''

    def validate_code(self, code_str: str) -> bool:
        try:
            compile(code_str, "<string>", "exec")
            return True
        except SyntaxError:
            return False

    def integrate_node(self, code_str: str, node_name: str) -> Tuple[bool, str]:
        if not self.validate_code(code_str):
            return False, "Syntax error in generated code"
        module = types.ModuleType(node_name)
        try:
            exec(code_str, module.__dict__)
            node_cls = getattr(module, node_name)
            instance = node_cls()
            self.generated_nodes[node_name] = instance
            return True, f"Node {node_name} integrated"
        except Exception as e:
            logger.exception("Integration failed")
            return False, str(e)


# --------------------------------------------------------------------------- #
# HTTP service                                                                #
# --------------------------------------------------------------------------- #
class IngenuityService:
    def __init__(self, core: IngenuityNodeCore) -> None:
        self.core = core

    # ---------- routes ---------- #
    async def handle_create_node(self, request: web.Request) -> web.Response:
        body = await request.json()
        node_name = body.get("node_name", "DemoNode")
        spec = body.get("specification", "Demo node created by IngenuityNode.")
        code = self.core.generate_node_code(node_name, spec)
        success, msg = self.core.integrate_node(code, node_name)
        return web.json_response({"success": success, "message": msg})

    async def handle_run_node(self, request: web.Request) -> web.Response:
        name = request.match_info["node"]
        instance = self.core.generated_nodes.get(name)
        if instance:
            instance.run()
            return web.json_response({"success": True, "message": f"Ran {name}"})
        return web.json_response({"success": False, "message": f"{name} not found"}, status=404)

    async def handle_status(self, request: web.Request) -> web.Response:
        return web.json_response({"nodes": list(self.core.generated_nodes.keys())})

    # ---------- app builder ---------- #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/create_node", self.handle_create_node),
            web.post("/run/{node}", self.handle_run_node),
            web.get("/status", self.handle_status),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentience 5.0 – IngenuityNode ROS-free")
    p.add_argument("--serve", action="store_true", help="run HTTP service")
    p.add_argument("--port", type=int, default=8085, help="HTTP port")
    p.add_argument("--create", nargs=2, metavar=("NAME", "SPEC"), help="one-shot CLI create & integrate")
    return p


# --------------------------------------------------------------------------- #
# Entry-point                                                               #
# --------------------------------------------------------------------------- #
async def amain() -> None:
    args = build_parser().parse_args()
    core = IngenuityNodeCore()
    service = IngenuityService(core)

    if args.create:
        name, spec = args.create
        code = core.generate_node_code(name, spec)
        success, msg = core.integrate_node(code, name)
        logger.info("Create %s: %s", name, msg)
        if success:
            core.generated_nodes[name].run()
        return

    if args.serve:
        app = service.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        logger.info("Ingenuity HTTP service on :%d", args.port)
        await asyncio.Event().wait()  # run forever
    else:
        logger.error("Nothing to do – use --create or --serve")


if __name__ == "__main__":
    asyncio.run(amain())
