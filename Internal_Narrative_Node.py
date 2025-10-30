#!/usr/bin/env python3
"""
Internal Narrative Node – ROS-free, asyncio-first
Same topics & JSON shapes, but HTTP/CLI instead of rospy.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

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
logger = logging.getLogger("InternalNarrative-Free")

# --------------------------------------------------------------------------- #
# Config & Utils                                                              #
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "db_root_path": "/tmp/sentience_db",
    "default_log_level": "INFO",
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

def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    if path and path.exists():
        return json.loads(path.read_text())
    logger.warning("Config not found – using defaults")
    return DEFAULT_CONFIG

# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
class InternalNarrativeNode:
    """
    Async, ROS-free internal narrative engine.
    Subscribers -> HTTP POST endpoints
    Publishers -> GET /internal_narrative, /error_monitor/report, /cognitive_directives
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.cfg = load_config(config_path)
        self.node_name = "internal_narrative_node"

        # --- DB setup --- #
        db_path = Path(self.cfg["db_root_path"]) / "internal_narrative_log.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

        # --- Async HTTP --- #
        self.session: Optional[aiohttp.ClientSession] = None

        # --- State --- #
        self.last_generated_narrative = {
            "timestamp": str(time.time()),
            "narrative_text": "I am observing my internal states.",
            "main_theme": "idle_reflection",
            "sentiment": 0.0,
            "salience_score": 0.1,
        }
        self.recent_attention_states: deque = deque(maxlen=5)
        self.recent_emotion_states: deque = deque(maxlen=5)
        self.recent_motivation_states: deque = deque(maxlen=5)
        self.recent_world_model_states: deque = deque(maxlen=5)
        self.recent_performance_reports: deque = deque(maxlen=5)
        self.recent_memory_responses: deque = deque(maxlen=5)
        self.recent_prediction_states: deque = deque(maxlen=5)
        self.recent_cognitive_directives: deque = deque(maxlen=3)
        self.cumulative_narrative_salience = 0.0

        # --- Queues for HTTP streaming --- #
        self.narrative_queue: asyncio.Queue[str] = asyncio.Queue()
        self.error_queue: asyncio.Queue[str] = asyncio.Queue()
        self.directive_queue: asyncio.Queue[str] = asyncio.Queue()

        # --- Timer handle --- #
        self.timer_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ #
    # Database                                                           #
    # ------------------------------------------------------------------ #
    def _init_db(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS internal_narrative_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                narrative_text TEXT,
                main_theme TEXT,
                sentiment REAL,
                salience_score REAL,
                llm_reasoning TEXT,
                context_snapshot_json TEXT
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_narrative_timestamp ON internal_narrative_log (timestamp)")
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # HTTP handlers (subscribers)                                        #
    # ------------------------------------------------------------------ #
    async def handle_attention_state(self, request: web.Request) -> web.Response:
        data = await request.json()
        self.recent_attention_states.append(data)
        self._update_cumulative_salience(data.get("priority_score", 0.0) * 0.2)
        logger.debug("Received attention state")
        return web.json_response({"status": "received"})

    async def handle_emotion_state(self, request: web.Request) -> web.Response:
        data = await request.json()
        self.recent_emotion_states.append(data)
        if data.get("mood_intensity", 0.0) > 0.5:
            self._update_cumulative_salience(data.get("mood_intensity", 0.0) * 0.4)
        logger.debug("Received emotion state")
        return web.json_response({"status": "received"})

    async def handle_motivation_state(self, request: web.Request) -> web.Response:
        data = await request.json()
        if isinstance(data.get("active_goals_json"), str):
            try:
                data["active_goals"] = json.loads(data["active_goals_json"])
            except json.JSONDecodeError:
                data["active_goals"] = {}
        self.recent_motivation_states.append(data)
        if data.get("overall_drive_level", 0.0) > 0.4:
            self._update_cumulative_salience(data.get("overall_drive_level", 0.0) * 0.3)
        logger.debug("Received motivation state")
        return web.json_response({"status": "received"})

    async def handle_world_model_state(self, request: web.Request) -> web.Response:
        data = await request.json()
        if isinstance(data.get("changed_entities_json"), str):
            try:
                data["changed_entities"] = json.loads(data["changed_entities_json"])
            except json.JSONDecodeError:
                data["changed_entities"] = []
        self.recent_world_model_states.append(data)
        if data.get("significant_change_flag") or data.get("consistency_score", 1.0) < 0.8:
            self._update_cumulative_salience(0.5)
        logger.debug("Received world model state")
        return web.json_response({"status": "received"})

    async def handle_performance_report(self, request: web.Request) -> web.Response:
        data = await request.json()
        if isinstance(data.get("kpis_json"), str):
            try:
                data["kpis"] = json.loads(data["kpis_json"])
            except json.JSONDecodeError:
                data["kpis"] = {}
        self.recent_performance_reports.append(data)
        if data.get("suboptimal_flag") or data.get("overall_score", 1.0) < 0.7:
            self._update_cumulative_salience(0.6)
        logger.debug("Received performance report")
        return web.json_response({"status": "received"})

    async def handle_memory_response(self, request: web.Request) -> web.Response:
        data = await request.json()
        if isinstance(data.get("memories_json"), str):
            try:
                data["memories"] = json.loads(data["memories_json"])
            except json.JSONDecodeError:
                data["memories"] = []
        self.recent_memory_responses.append(data)
        if data.get("response_code") == 200 and data.get("memories"):
            self._update_cumulative_salience(0.25)
        logger.debug("Received memory response")
        return web.json_response({"status": "received"})

    async def handle_prediction_state(self, request: web.Request) -> web.Response:
        data = await request.json()
        self.recent_prediction_states.append(data)
        if data.get("urgency_flag") or data.get("prediction_confidence", 0.0) > 0.7:
            self._update_cumulative_salience(0.6)
        logger.debug("Received prediction state")
        return web.json_response({"status": "received"})

    async def handle_cognitive_directive(self, request: web.Request) -> web.Response:
        data = await request.json()
        self.recent_cognitive_directives.append(data)
        if data.get("target_node") == self.node_name and data.get("directive_type") == "GenerateInternalNarrative":
            self._update_cumulative_salience(data.get("urgency", 0.0) * 1.0)
            logger.info("Received directive to generate narrative")
        return web.json_response({"status": "received"})

    # ------------------------------------------------------------------ #
    # Salience & pruning                                                 #
    # ------------------------------------------------------------------ #
    def _update_cumulative_salience(self, score: float) -> None:
        self.cumulative_narrative_salience += score
        self.cumulative_narrative_salience = min(1.0, self.cumulative_narrative_salience)

    def _prune_history(self) -> None:
        current_time = time.time()
        for dq in [
            self.recent_attention_states,
            self.recent_emotion_states,
            self.recent_motivation_states,
            self.recent_world_model_states,
            self.recent_performance_reports,
            self.recent_memory_responses,
            self.recent_prediction_states,
            self.recent_cognitive_directives,
        ]:
            while dq and (current_time - float(dq[0].get("timestamp", 0.0))) > self.cfg["internal_narrative_node"]["recent_context_window_s"]:
                dq.popleft()

    # ------------------------------------------------------------------ #
    # Narrative generation (async)                                       #
    # ------------------------------------------------------------------ #
    async def generate_internal_narrative_async(self) -> None:
        self._prune_history()

        narrative_text = "..."
        main_theme = "idle_reflection"
        sentiment = 0.0
        salience_score = 0.1
        llm_reasoning = "Not evaluated by LLM."

        threshold = self.cfg["internal_narrative_node"]["llm_narrative_threshold_salience"]
        if self.cumulative_narrative_salience >= threshold:
            logger.info("Triggering LLM for narrative (salience: %.2f)", self.cumulative_narrative_salience)
            context_for_llm = self._compile_llm_context_for_narrative()
            llm_output = await self._generate_llm_narrative(context_for_llm)
            if llm_output:
                narrative_text = llm_output.get("narrative_text", "No narrative generated.")
                main_theme = llm_output.get("main_theme", "unspecified")
                sentiment = max(-1.0, min(1.0, llm_output.get("sentiment", 0.0)))
                salience_score = max(0.0, min(1.0, llm_output.get("salience_score", 0.1)))
                llm_reasoning = llm_output.get("llm_reasoning", "LLM provided no specific reasoning.")
            else:
                logger.warning("LLM narrative generation failed – falling back to simple rules")
                narrative_text, main_theme, sentiment, salience_score = self._apply_simple_narrative_rules()
                llm_reasoning = "Fallback to simple rules due to LLM failure."
        else:
            logger.debug("Insufficient salience (%.2f) – using simple rules", self.cumulative_narrative_salience)
            narrative_text, main_theme, sentiment, salience_score = self._apply_simple_narrative_rules()
            llm_reasoning = "Fallback to simple rules due to low salience."

        self.last_generated_narrative = {
            "timestamp": str(time.time()),
            "narrative_text": narrative_text,
            "main_theme": main_theme,
            "sentiment": sentiment,
            "salience_score": salience_score,
        }

        self.save_internal_narrative_log(
            id=str(uuid.uuid4()),
            timestamp=self.last_generated_narrative["timestamp"],
            narrative_text=self.last_generated_narrative["narrative_text"],
            main_theme=self.last_generated_narrative["main_theme"],
            sentiment=self.last_generated_narrative["sentiment"],
            salience_score=self.last_generated_narrative["salience_score"],
            llm_reasoning=llm_reasoning,
            context_snapshot_json=json.dumps(self._compile_llm_context_for_narrative()),
        )
        self.publish_internal_narrative()
        self.cumulative_narrative_salience = 0.0

    async def _generate_llm_narrative(self, context_for_llm: Dict) -> Optional[Dict]:
        prompt = f"""
You are the Internal Narrative Module of a robot's cognitive architecture. Generate a concise internal monologue (1-3 sentences) reflecting the robot's current internal state.

Current context:
{json.dumps(context_for_llm, indent=2)}

Respond in JSON:
{{
  "timestamp": "<current time>",
  "narrative_text": "<monologue>",
  "main_theme": "problem_solving|self_assessment|reflection|planning|emotional_processing|environmental_analysis|idle_reflection",
  "sentiment": <float, -1.0 to 1.0>,
  "salience_score": <float, 0.0 to 1.0>,
  "llm_reasoning": "<why this narrative>"
}}
"""
        payload = {
            "model": self.cfg["llm_params"]["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 250,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        try:
            async with self.session.post(
                self.cfg["llm_params"]["base_url"],
                json=payload,
                timeout=self.cfg["llm_params"]["timeout_seconds"],
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                content = result["choices"][0]["message"]["content"]
                data = json.loads(content)
                data["sentiment"] = float(data["sentiment"])
                data["salience_score"] = float(data["salience_score"])
                return data
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

    def _apply_simple_narrative_rules(self) -> tuple[str, str, float, float]:
        current_time = time.time()
        narrative_text = "I am processing data."
        main_theme = "idle_reflection"
        sentiment = 0.0
        salience_score = 0.1

        if self.recent_performance_reports:
            latest = self.recent_performance_reports[-1]
            if current_time - float(latest.get("timestamp", 0.0)) < 2.0 and latest.get("suboptimal_flag") and latest.get("overall_score", 1.0) < 0.6:
                narrative_text = f"My performance is {latest.get('overall_score'):.2f}. I need to improve efficiency."
                main_theme = "self_assessment_problem"
                sentiment = -0.5
                salience_score = 0.7
                return narrative_text, main_theme, sentiment, salience_score

        if self.recent_emotion_states:
            latest = self.recent_emotion_states[-1]
            if current_time - float(latest.get("timestamp", 0.0)) < 1.0 and latest.get("mood_intensity", 0.0) > 0.4:
                mood = latest.get("mood", "neutral")
                if mood == "happy":
                    narrative_text = "I feel quite positive. This state is conducive to productive tasks."
                    main_theme = "emotional_processing"
                    sentiment = 0.6
                    salience_score = 0.4
                elif mood == "frustrated":
                    narrative_text = "I am experiencing some frustration. I should identify the source."
                    main_theme = "emotional_processing_problem_solving"
                    sentiment = -0.6
                    salience_score = 0.5
                return narrative_text, main_theme, sentiment, salience_score

        if self.recent_motivation_states:
            latest = self.recent_motivation_states[-1]
            if current_time - float(latest.get("timestamp", 0.0)) < 2.0 and latest.get("dominant_goal_id") != "none":
                narrative_text = f"My current primary objective is to '{latest.get('dominant_goal_id')}'. I will continue to focus my resources on achieving this."
                main_theme = "planning_goal_focus"
                sentiment = 0.3
                salience_score = 0.3
                return narrative_text, main_theme, sentiment, salience_score

        narrative_text = "My systems are stable. I am observing the incoming data streams and preparing for the next task."
        main_theme = "idle_reflection"
        sentiment = 0.0
        salience_score = 0.1
        return narrative_text, main_theme, sentiment, salience_score

    def _compile_llm_context_for_narrative(self) -> Dict:
        return {
            "current_time": time.time(),
            "previous_narrative": self.last_generated_narrative,
            "recent_cognitive_inputs": {
                "attention_state": list(self.recent_attention_states)[-1] if self.recent_attention_states else "N/A",
                "emotion_state": list(self.recent_emotion_states)[-1] if self.recent_emotion_states else "N/A",
                "motivation_state": list(self.recent_motivation_states)[-1] if self.recent_motivation_states else "N/A",
                "world_model_state": list(self.recent_world_model_states)[-1] if self.recent_world_model_states else "N/A",
                "performance_report": list(self.recent_performance_reports)[-1] if self.recent_performance_reports else "N/A",
                "memory_responses": list(self.recent_memory_responses),
                "prediction_state": list(self.recent_prediction_states)[-1] if self.recent_prediction_states else "N/A",
                "cognitive_directives_for_self": [
                    d for d in self.recent_cognitive_directives if d.get("target_node") == self.node_name
                ],
            },
        }

    # ------------------------------------------------------------------ #
    # Database & publishing                                              #
    # ------------------------------------------------------------------ #
    def save_internal_narrative_log(
        self,
        id: str,
        timestamp: str,
        narrative_text: str,
        main_theme: str,
        sentiment: float,
        salience_score: float,
        llm_reasoning: str,
        context_snapshot_json: str,
    ) -> None:
        try:
            self.conn.execute(
                """
                INSERT INTO internal_narrative_log
                (id, timestamp, narrative_text, main_theme, sentiment, salience_score, llm_reasoning, context_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (id, timestamp, narrative_text, main_theme, sentiment, salience_score, llm_reasoning, context_snapshot_json),
            )
            self.conn.commit()
            logger.debug("Saved narrative log (ID: %s, Theme: %s)", id, main_theme)
        except sqlite3.Error as e:
            logger.error("DB save failed: %s", e)

    def publish_internal_narrative(self) -> None:
        try:
            self.last_generated_narrative["timestamp"] = str(time.time())
            self.narrative_queue.put_nowait(json.dumps(self.last_generated_narrative))
            logger.debug("Published internal narrative (Theme: %s)", self.last_generated_narrative["main_theme"])
        except Exception as e:
            logger.error("Publish failed: %s", e)

    # ------------------------------------------------------------------ #
    # HTTP publishers (GET streams)                                      #
    # ------------------------------------------------------------------ #
    async def handle_internal_narrative(self, request: web.Request) -> web.Response:
        try:
            msg = await asyncio.wait_for(self.narrative_queue.get(), timeout=30)
            return web.json_response(json.loads(msg))
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    async def handle_error_report(self, request: web.Request) -> web.Response:
        try:
            msg = await asyncio.wait_for(self.error_queue.get(), timeout=30)
            return web.json_response(json.loads(msg))
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    async def handle_cognitive_directives(self, request: web.Request) -> web.Response:
        try:
            msg = await asyncio.wait_for(self.directive_queue.get(), timeout=30)
            return web.json_response(json.loads(msg))
        except asyncio.TimeoutError:
            return web.json_response({"status": "timeout"})

    # ------------------------------------------------------------------ #
    Timer / lifecycle                                                  #
    # ------------------------------------------------------------------ #
    async def _timer_loop(self) -> None:
        interval = self.cfg["internal_narrative_node"]["narrative_generation_interval"]
        while True:
            await asyncio.sleep(interval)
            await self.generate_internal_narrative_async()

    async def start(self) -> None:
        self.session = aiohttp.ClientSession()
        self.timer_task = asyncio.create_task(self._timer_loop())
        logger.info("Internal Narrative Node started (ROS-free)")

    async def stop(self) -> None:
        if self.timer_task:
            self.timer_task.cancel()
        if self.session:
            await self.session.close()
        self.conn.close()
        logger.info("Internal Narrative Node shut down cleanly")

    # ------------------------------------------------------------------ #
    # App builder                                                        #
    # ------------------------------------------------------------------ #
    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.post("/attention_state", self.handle_attention_state),
            web.post("/emotion_state", self.handle_emotion_state),
            web.post("/motivation_state", self.handle_motivation_state),
            web.post("/world_model_state", self.handle_world_model_state),
            web.post("/performance_report", self.handle_performance_report),
            web.post("/memory_response", self.handle_memory_response),
            web.post("/prediction_state", self.handle_prediction_state),
            web.post("/cognitive_directives", self.handle_cognitive_directive),
            web.get("/internal_narrative", self.handle_internal_narrative),
            web.get("/error_monitor/report", self.handle_error_report),
            web.get("/cognitive_directives", self.handle_cognitive_directives),
        ])
        return app


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentience 5.0 – InternalNarrativeNode ROS-free")
    p.add_argument("--config", type=Path, help="optional JSON config file")
    p.add_argument("--serve", action="store_true", help="run HTTP service")
    p.add_argument("--port", type=int, default=8089, help="HTTP port")
    p.add_argument("--test", action="store_true", help="inject test messages and show narrative")
    return p


# --------------------------------------------------------------------------- #
# Test injector (optional)                                                    #
# --------------------------------------------------------------------------- #
async def inject_test_messages(node: InternalNarrativeNode) -> None:
    await asyncio.sleep(0.5)
    test_msgs = [
        ("/attention_state", {"priority_score": 0.8}),
        ("/emotion_state", {"mood": "frustrated", "mood_intensity": 0.7}),
        ("/performance_report", {"suboptimal_flag": True, "overall_score": 0.5}),
    ]
    for topic, payload in test_msgs:
        async with aiohttp.ClientSession() as session:
            await session.post(f"http://localhost:8089{topic}", json=payload)
        await asyncio.sleep(0.2)

    # wait for narrative
    try:
        narrative = await asyncio.wait_for(node.narrative_queue.get(), timeout=10)
        print("Test narrative:")
        print(json.dumps(json.loads(narrative), indent=2))
    except asyncio.TimeoutError:
        print("No narrative received")


# --------------------------------------------------------------------------- #
# Entry-point                                                               #
# --------------------------------------------------------------------------- #
async def amain() -> None:
    args = build_parser().parse_args()
    node = InternalNarrativeNode(config_path=args.config)

    if args.test:
        await inject_test_messages(node)
        return

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        await node.start()
        logger.info("Internal Narrative HTTP service on :%d", args.port)
        await asyncio.Event().wait()  # run forever
    else:
        logger.error("Nothing to do – use --test or --serve")


if __name__ == "__main__":
    asyncio.run(amain())
