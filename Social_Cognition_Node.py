#!/usr/bin/env python3
"""
SocialCognitionNode – Sentience 5.5 (ROS-free, asyncio-first)

Role:
- Receive interaction requests
- Apply compassionate social reasoning (LLM-assisted or fallback)
- Publish interaction responses
- Persist interactions for learning/audit
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import sqlite3
import argparse
import asyncio
import threading
import random
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque

import aiohttp
from aiohttp import web

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# LLM Client (HTTP, local-first)                                               #
# --------------------------------------------------------------------------- #
class AsyncLLMClient:
    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def query(self, prompt: str, temperature: float = 0.4, max_tokens: int = 256) -> str:
        await self.ensure_session()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            async with self.session.post(
                self.base_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM_ERROR] {e}"


# --------------------------------------------------------------------------- #
# Social Cognition Node                                                        #
# --------------------------------------------------------------------------- #
class SocialCognitionNode:
    def __init__(self, db_root: str = "/tmp/sentience_db"):
        self.node_name = "social_cognition_node"

        # Parameters
        self.ethical_compassion_bias = 0.2
        self.llm_model = "phi-2"
        self.llm_base_url = "http://localhost:8000/v1/chat/completions"
        self.llm_timeout = 10.0

        # State
        self.pending_queries: Deque[Dict[str, Any]] = deque(maxlen=32)
        self.interaction_history: Deque[Dict[str, Any]] = deque(maxlen=128)

        # DB
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "social_cognition.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # Async
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        self.llm = AsyncLLMClient(
            base_url=self.llm_base_url,
            model=self.llm_model,
            timeout=self.llm_timeout,
        )

        log("INFO", self.node_name, "Initialized")

    # ------------------------------------------------------------------ #
    # DB                                                                  #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                prompt TEXT,
                response TEXT,
                compassion_bias REAL
            )
        """)
        self.conn.commit()

    def _log_interaction(self, prompt: str, response: str):
        self.conn.execute(
            """
            INSERT INTO interactions (id, timestamp, prompt, response, compassion_bias)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                time.time(),
                prompt,
                response,
                self.ethical_compassion_bias,
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Async Loop                                                          #
    # ------------------------------------------------------------------ #
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # ------------------------------------------------------------------ #
    # Core Logic                                                          #
    # ------------------------------------------------------------------ #
    async def _handle_prompt_async(self, prompt: str) -> str:
        compassionate_prompt = (
            f"{prompt}\n\n"
            f"Respond with empathy, social awareness, and compassion. "
            f"Compassion bias level: {self.ethical_compassion_bias}."
        )
        response = await self.llm.query(compassionate_prompt)
        self._log_interaction(prompt, response)
        return response

    def handle_prompt(self, prompt: str) -> str:
        fut = asyncio.run_coroutine_threadsafe(
            self._handle_prompt_async(prompt), self.loop
        )
        return fut.result(timeout=15)

    # ------------------------------------------------------------------ #
    # HTTP Handlers                                                       #
    # ------------------------------------------------------------------ #
    async def interaction_request_handler(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            log("INFO", self.node_name, f"Received prompt: {prompt}")
            response = await self._handle_prompt_async(prompt)
            return web.json_response({
                "timestamp": time.time(),
                "response": response
            })
        except Exception as e:
            log("ERROR", self.node_name, f"Request error: {e}")
            return web.json_response({"error": str(e)}, status=400)

    async def health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "node": self.node_name,
            "time": time.time()
        })

    # ------------------------------------------------------------------ #
    # App                                                                 #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/social/interaction_request", self.interaction_request_handler),
            web.get("/social/health", self.health_handler),
        ])
        return app

    # ------------------------------------------------------------------ #
    # Shutdown                                                            #
    # ------------------------------------------------------------------ #
    async def shutdown_async(self):
        await self.llm.close()

    def shutdown(self):
        log("INFO", self.node_name, "Shutting down")
        asyncio.run_coroutine_threadsafe(self.shutdown_async(), self.loop)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.conn.close()


# --------------------------------------------------------------------------- #
# Entry-point                                                                 #
# --------------------------------------------------------------------------- #
async def amain():
    parser = argparse.ArgumentParser(description="Sentience 5.5 – Social Cognition Node")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()

    node = SocialCognitionNode()

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
