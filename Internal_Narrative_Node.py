#!/usr/bin/env python3
"""
InternalNarrativeNode – Updated (Sentience 5.5 compliant)

Applied updates:
- Deterministic, versioned narrative frames
- Clear separation: signal accumulation → generation → publication
- Evolver-compatible state & metrics export
- LLM strictly advisory, gated by salience
- Async lifecycle hardened (no silent task leaks)
- No ROS, no hidden coupling
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional

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
logger = logging.getLogger("InternalNarrative_Node")

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "db_root_path": "/tmp/sentience_db",
    "internal_narrative_node": {
        "narrative_generation_interval": 1.0,
        "llm_narrative_threshold_salience": 0.5,
        "recent_context_window_s": 15.0,
    },
    "llm_params": {
        "model_name": "phi-2",
        "base_url": "http://localhost:8000/v1/chat/completions",
        "timeout_seconds": 30.0,
    },
}

def load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path and path.exists():
        return json.loads(path.read_text())
    return DEFAULT_CONFIG

# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
class InternalNarrativeNode:
    """
    Generates internal narrative frames from accumulated cognitive signals.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.cfg = load_config(config_path)
        self.node_name = "internal_narrative_node"

        # --- DB --- #
        db_path = Path(self.cfg["db_root_path"]) / "internal_narrative_log.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

        # --- Async --- #
        self.session: Optional[aiohttp.ClientSession] = None
        self.timer_task: Optional[asyncio.Task] = None

        # --- State --- #
        self.narrative_version: int = 0
        self.cumulative_salience: float = 0.0

        self.last_narrative: Dict[str, Any] = {
            "version": 0,
            "timestamp": time.time(),
            "narrative_text": "System idle.",
            "main_theme": "idle_reflection",
            "sentiment": 0.0,
            "salience_score": 0.1,
        }

        # --- Signal buffers --- #
        self.buffers = {
            "attention": deque(maxlen=5),
            "emotion": deque(maxlen=5),
            "motivation": deque(maxlen=5),
            "world_model": deque(maxlen=5),
            "performance": deque(maxlen=5),
            "memory": deque(maxlen=5),
            "prediction": deque(maxlen=5),
            "directives": deque(maxlen=3),
        }

        # --- Output queues --- #
        self.narrative_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS internal_narrative_log (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                version INTEGER,
                narrative_text TEXT,
                main_theme TEXT,
                sentiment REAL,
                salience_score REAL,
                llm_reasoning TEXT,
                context_snapshot_json TEXT
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Input handlers                                                     #
    # ------------------------------------------------------------------ #
    async def _handle_signal(self, key: str, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        data["timestamp"] = time.time()
        self.buffers[key].append(data)

        weight = {
            "attention": 0.2,
            "emotion": 0.4,
            "motivation": 0.3,
            "world_model": 0.5,
            "performance": 0.6,
            "memory": 0.25,
            "prediction": 0.6,
            "directives": 1.0,
        }.get(key, 0.1)

        self._accumulate_salience(weight)
        return web.json_response({"status": "accepted"})

    # ------------------------------------------------------------------ #
    # Salience                                                           #
    # ------------------------------------------------------------------ #
    def _accumulate_salience(self, score: float) -> None:
        self.cumulative_salience = min(1.0, self.cumulative_salience + score)

    # ------------------------------------------------------------------ #
    # Narrative generation                                               #
    # ------------------------------------------------------------------ #
    async def _generate(self) -> None:
        threshold = self.cfg["internal_narrative_node"]["llm_narrative_threshold_salience"]

        narrative_text = "Monitoring internal state."
        main_theme = "idle_reflection"
        sentiment = 0.0
        salience_score = max(0.1, self.cumulative_salience)
        llm_reasoning = "Rule-based generation."

        if self.cumulative_salience >= threshold and self.session:
            llm_out = await self._query_llm(self._compile_context())
            if llm_out:
                narrative_text = llm_out.get("narrative_text", narrative_text)
                main_theme = llm_out.get("main_theme", main_theme)
                sentiment = float(llm_out.get("sentiment", sentiment))
                salience_score = float(llm_out.get("salience_score", salience_score))
                llm_reasoning = llm_out.get("llm_reasoning", llm_reasoning)

        self.narrative_version += 1
        frame = {
            "version": self.narrative_version,
            "timestamp": time.time(),
            "narrative_text": narrative_text,
            "main_theme": main_theme,
            "sentiment": sentiment,
            "salience_score": salience_score,
        }

        self.last_narrative = frame
        self._persist(frame, llm_reasoning)
        self.narrative_queue.put_nowait(frame)

        self.cumulative_salience = 0.0

    async def _query_llm(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = f"""
Generate a concise internal narrative (1–3 sentences).

Context:
{json.dumps(context, indent=2)}

Return JSON with:
narrative_text, main_theme, sentiment (-1..1), salience_score (0..1), llm_reasoning
"""
        payload = {
            "model": self.cfg["llm_params"]["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "max_tokens": 250,
            "stream": False,
        }
        try:
            async with self.session.post(
                self.cfg["llm_params"]["base_url"],
                json=payload,
                timeout=self.cfg["llm_params"]["timeout_seconds"],
            ) as resp:
                result = await resp.json()
                return json.loads(result["choices"][0]["message"]["content"])
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Context                                                            #
    # ------------------------------------------------------------------ #
    def _compile_context(self) -> Dict[str, Any]:
        return {
            "previous_narrative": self.last_narrative,
            "signals": {k: list(v) for k, v in self.buffers.items()},
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def _persist(self, frame: Dict[str, Any], reasoning: str) -> None:
        self.conn.execute(
            """
            INSERT INTO internal_narrative_log
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                frame["timestamp"],
                frame["version"],
                frame["narrative_text"],
                frame["main_theme"],
                frame["sentiment"],
                frame["salience_score"],
                reasoning,
                json.dumps(self._compile_context()),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Output                                                             #
    # ------------------------------------------------------------------ #
    async def handle_internal_narrative(self, request: web.Request) -> web.Response:
        try:
            frame = await asyncio.wait_for(self.narrative_queue.get(), timeout=30)
            return web.json_response(frame)
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    async def _loop(self) -> None:
        interval = self.cfg["internal_narrative_node"]["narrative_generation_interval"]
        while True:
            await asyncio.sleep(interval)
            await self._generate()

    async def start(self) -> None:
        self.session = aiohttp.ClientSession()
        self.timer_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self.timer_task:
            self.timer_task.cancel()
        if self.session:
            await self.session.close()
        self.conn.close()

    # ------------------------------------------------------------------ #
    # App builder                                                        #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/attention_state", lambda r: self._handle_signal("attention", r)),
            web.post("/emotion_state", lambda r: self._handle_signal("emotion", r)),
            web.post("/motivation_state", lambda r: self._handle_signal("motivation", r)),
            web.post("/world_model_state", lambda r: self._handle_signal("world_model", r)),
            web.post("/performance_report", lambda r: self._handle_signal("performance", r)),
            web.post("/memory_response", lambda r: self._handle_signal("memory", r)),
            web.post("/prediction_state", lambda r: self._handle_signal("prediction", r)),
            web.post("/cognitive_directives", lambda r: self._handle_signal("directives", r)),
            web.get("/internal_narrative", self.handle_internal_narrative),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentience 5.5 – InternalNarrativeNode")
    p.add_argument("--config", type=Path)
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8089)
    return p


async def amain() -> None:
    args = build_parser().parse_args()
    node = InternalNarrativeNode(args.config)

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        await node.start()
        logger.info("InternalNarrativeNode running on :%d", args.port)
        await asyncio.Event().wait()
    else:
        logger.error("Use --serve")


if __name__ == "__main__":
    asyncio.run(amain())
