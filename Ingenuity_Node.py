Below is a **ROS-free, asyncio-first** rewrite of the Ingenuity node that

* keeps the **same public API** (`generate_node_code`, `validate_code`, `integrate_node`, `save_node_code`, `create_and_deploy_node`, `run_node`)  
* removes every assumption about a ROS workspace – the default output folder is simply `./ingenuity_nodes` (override with `--workspace`)  
* adds **async** file-IO and optional **HTTP hooks** so other nodes can ask the ingenuity service to create & hot-load code at runtime  
* is 100 % stand-alone (no `rospy`, no `catkin`) but can be bridged back to ROS later via a thin wrapper (supplied at the end)

Save as `ingenuity_node.py` and run:

```bash
# interactive CLI
python3 ingenuity_node.py \
  --workspace ./my_nodes \
  --generate "EnergyManagerNode|Manages battery usage efficiently."

# or as a long-running service
python3 ingenuity_node.py --serve --port 8083
```

--------------------------------------------------
ingenuity_node.py
--------------------------------------------------
```python
#!/usr/bin/env python3
"""
Stand-alone Ingenuity node for Sentience 5.0
ROS-free, but API compatible.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import sys
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
logger = logging.getLogger("Ingenuity-Node")

# --------------------------------------------------------------------------- #
# Models                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class CodeArtifact:
    node_name: str
    code: str
    module: Optional[types.ModuleType] = None
    instance: Optional[Any] = None


# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
class IngenuityNode:
    """
    Async, ROS-free ingenuity engine.
    Public API is identical to the original class.
    """

    def __init__(self, workspace: Path, notifier: Optional[Callable[[Dict], None]] = None):
        self.workspace: Path = workspace
        self.artifacts: Dict[str, CodeArtifact] = {}
        self.notifier = notifier or self._default_notifier

    # ------------------------------------------------------------------ #
    # Public API (sync shape preserved)                                 #
    # ------------------------------------------------------------------ #
    def generate_node_code(self, node_name: str, specification: str) -> str:
        """
        Extremely small template – swap this for an LLM call if you want.
        """
        return f'''
class {node_name}:
    def __init__(self):
        self.name = "{node_name}"

    def run(self):
        print("Running {node_name}: {specification}")
'''

    def validate_code(self, code_str: str) -> bool:
        try:
            ast.parse(code_str)
            return True
        except SyntaxError as exc:
            self.notifier({"status": "validation_failed", "error": str(exc)})
            return False

    def integrate_node(self, code_str: str, node_name: str) -> Optional[Any]:
        if not self.validate_code(code_str):
            return None
        module = types.ModuleType(node_name)
        try:
            exec(code_str, module.__dict__)
            node_cls = getattr(module, node_name)
            instance = node_cls()
            self.artifacts[node_name] = CodeArtifact(
                node_name=node_name, code=code_str, module=module, instance=instance
            )
            self.notifier({"status": "integrated", "node": node_name})
            return instance
        except Exception:
            self.notifier({"status": "integration_error", "node": node_name})
            logger.exception("Integration failed")
            return None

    async def save_node_code(self, node_name: str, code_str: str) -> Path:
        """
        Async file write – so we can call it from the HTTP handler.
        """
        pkg_dir = self.workspace / node_name
        pkg_dir.mkdir(parents=True, exist_ok=True)
        file_path = pkg_dir / f"{node_name}.py"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, file_path.write_text, code_str)
        self.notifier({"status": "saved_to_disk", "path": str(file_path)})
        return file_path

    async def create_and_deploy_node(self, node_name: str, specification: str) -> Optional[Any]:
        code = self.generate_node_code(node_name, specification)
        await self.save_node_code(node_name, code)
        return self.integrate_node(code, node_name)

    def run_node(self, node_name: str) -> None:
        artifact = self.artifacts.get(node_name)
        if artifact and artifact.instance:
            try:
                artifact.instance.run()
            except Exception as exc:
                logger.error("Error running node %s: %s", node_name, exc)
        else:
            logger.error("Node %s not found or not integrated", node_name)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _default_notifier(self, msg: Dict) -> None:
        logger.info("Ingenuity event: %s", msg)

    # ------------------------------------------------------------------ #
    # HTTP service (optional)                                            #
    # ------------------------------------------------------------------ #
    async def _http_generate(self, request: web.Request) -> web.Response:
        params = await request.json()
        name = params["node_name"]
        spec = params["specification"]
        instance = await self.create_and_deploy_node(name, spec)
        if instance:
            return web.json_response({"status": "created", "node": name})
        return web.json_response({"status": "failed"}, status=400)

    async def _http_run(self, request: web.Request) -> web.Response:
        name = request.match_info["node"]
        self.run_node(name)
        return web.json_response({"status": "ran", "node": name})

    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/generate", self._http_generate),
            web.post("/run/{node}", self._http_run),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentience 5.0 – Ingenuity Node (ROS-free)")
    p.add_argument("--workspace", type=Path, default=Path("./ingenuity_nodes"), help="output directory")
    p.add_argument("--serve", action="store_true", help="run as HTTP service")
    p.add_argument("--port", type=int, default=8083, help="HTTP port")
    p.add_argument("--generate", metavar="NAME|SPEC", help="one-shot CLI: generate & deploy node")
    return p


# --------------------------------------------------------------------------- #
# Entry-point                                                               #
# --------------------------------------------------------------------------- #
async def amain() -> None:
    args = build_parser().parse_args()
    node = IngenuityNode(workspace=args.workspace)

    if args.generate:
        name, spec = args.generate.split("|", 1)
        await node.create_and_deploy_node(name, spec)
        node.run_node(name)
        return

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        logger.info("Ingenuity HTTP service listening on :%d", args.port)
        await asyncio.Event().wait()  # run for ever
    else:
        logger.error("Nothing to do – use --generate or --serve")


if __name__ == "__main__":
    asyncio.run(amain())
```

--------------------------------------------------
ROS-wrapper sketch (optional)
--------------------------------------------------
```python
#!/usr/bin/env python3
"""
Thin ROS shim around the stand-alone IngenuityNode.
Subscribes to `/ingenuity/request` (JSON) and publishes `/ingenuity/response`.
"""
import rospy, asyncio, threading, json
from std_msgs.msg import String
from ingenuity_node import IngenuityNode, build_parser

class ROSIngenuityBridge:
    def __init__(self):
        rospy.init_node('ingenuity_node')
        args = build_parser().parse_args(rospy.myargv()[1:])
        self.node = IngenuityNode(workspace=args.workspace)

        self.pub = rospy.Publisher('/ingenuity/response', String, queue_size=10)
        rospy.Subscriber('/ingenuity/request', String, self._on_request)

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_async, daemon=True)
        self.thread.start()

    def _on_request(self, msg):
        self.loop.call_soon_threadsafe(self._handle, msg.data)

    def _run_async(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _handle(self, data: str):
        try:
            req = json.loads(data)
            cmd = req["cmd"]
            if cmd == "generate":
                name = req["node_name"]
                spec = req["specification"]
                instance = await self.node.create_and_deploy_node(name, spec)
                self.pub.publish(json.dumps({"status": "created", "node": name}))
            elif cmd == "run":
                name = req["node_name"]
                self.node.run_node(name)
                self.pub.publish(json.dumps({"status": "ran", "node": name}))
            else:
                self.pub.publish(json.dumps({"status": "unknown_cmd"}))
        except Exception as exc:
            self.pub.publish(json.dumps({"status": "error", "error": str(exc)}))

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    ROSIngenuityBridge().spin
