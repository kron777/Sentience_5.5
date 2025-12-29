#!/usr/bin/env python3
"""
InsightNode – Evolver-integrated, learning-enabled
Generates node suggestions from awareness + conversation + dreaming
Now instrumented for Evolver.py (Sentience 5.5)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, List

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
logger = logging.getLogger("Insight-Node")

# --------------------------------------------------------------------------- #
# Protocols (stable contracts)                                                #
# --------------------------------------------------------------------------- #
class AwarenessProvider(Protocol):
    def get_state_description(self) -> str: ...

class ConversationProvider(Protocol):
    def get_recent_dialogue(self) -> str: ...

class DreamingProvider(Protocol):
    def get_recent_dream(self) -> str: ...

class LLMProvider(Protocol):
    async def query(self, prompt: str) -> str: ...

class EvolverHook(Protocol):
    def record(self, payload: Dict[str, Any]) -> None: ...

# --------------------------------------------------------------------------- #
# Evolver telemetry                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class InsightTelemetry:
    node: str = "InsightNode"
    event_id: str = ""
    timestamp: float = 0.0
    context_size: int = 0
    llm_latency_ms: float = 0.0
    suggestions_count: int = 0
    json_valid: bool = False
    error: Optional[str] = None

# --------------------------------------------------------------------------- #
# InsightNode                                                                 #
# --------------------------------------------------------------------------- #
class InsightNode:
    """
    Evolver-compatible Insight Node.
    - Learning aware
    - Telemetry emitting
    - Deterministic API surface
    """

    def __init__(
        self,
        awareness: AwarenessProvider,
        conversation: ConversationProvider,
        dreaming: DreamingProvider,
        llm: LLMProvider,
        evolver: Optional[EvolverHook] = None,
    ):
        self.awareness = awareness
        self.conversation = conversation
        self.dreaming = dreaming
        self.llm = llm
        self.evolver = evolver

    # ------------------------------------------------------------------ #
    # Context + prompt                                                   #
    # ------------------------------------------------------------------ #
    def gather_context(self) -> str:
        awareness_text = self.awareness.get_state_description()
        conversation_text = self.conversation.get_recent_dialogue()
        dreaming_text = self.dreaming.get_recent_dream()

        return (
            f"Robot Awareness:\n{awareness_text}\n\n"
            f"Recent Conversation:\n{conversation_text}\n\n"
            f"Dreaming Output:\n{dreaming_text}\n"
        )

    def construct_prompt(self, context: str) -> str:
        return (
            "You are an architectural insight engine.\n"
            "Your task is to identify missing or weak cognitive functions.\n\n"
            f"Context:\n{context}\n\n"
            "Propose new functional nodes that would improve the system.\n"
            "Each node must be specific, non-duplicative, and purposeful.\n\n"
            "Return JSON only in this format:\n"
            "{\n"
            "  \"nodes\": [\n"
            "    {\n"
            "      \"name\": \"NodeName\",\n"
            "      \"spec\": \"Clear functional responsibility\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

    # ------------------------------------------------------------------ #
    # Core analysis                                                      #
    # ------------------------------------------------------------------ #
    async def analyze(self) -> Dict[str, Any]:
        telemetry = InsightTelemetry(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
        )

        context = self.gather_context()
        telemetry.context_size = len(context)

        prompt = self.construct_prompt(context)

        start = time.perf_counter()
        raw_response = await self.llm.query(prompt)
        telemetry.llm_latency_ms = (time.perf_counter() - start) * 1000.0

        try:
            parsed = json.loads(raw_response)
            telemetry.json_valid = True
            telemetry.suggestions_count = len(parsed.get("nodes", []))
        except Exception as e:
            parsed = {"nodes": []}
            telemetry.error = str(e)
            logger.warning("InsightNode: invalid JSON from LLM")

        # Emit to evolver
        if self.evolver:
            self.evolver.record(asdict(telemetry))

        return parsed

    # ------------------------------------------------------------------ #
    # HTTP interface                                                     #
    # ------------------------------------------------------------------ #
    async def _http_analyze(self, request: web.Request) -> web.Response:
        result = await self.analyze()
        return web.json_response(result)

    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([web.get("/analyze", self._http_analyze)])
        return app


# --------------------------------------------------------------------------- #
# Evolver-compatible stubs                                                    #
# --------------------------------------------------------------------------- #
class StdoutEvolver:
    """Drop-in Evolver hook (replace with evolver.py collector)."""
    def record(self, payload: Dict[str, Any]) -> None:
        logger.info("EVOLVER_EVENT %s", json.dumps(payload))


# --------------------------------------------------------------------------- #
# Mock providers                                                              #
# --------------------------------------------------------------------------- #
class MockLLM:
    async def query(self, prompt: str) -> str:
        await asyncio.sleep(0.1)
        return json.dumps(
            {
                "nodes": [
                    {
                        "name": "CognitiveLoadBalancerNode",
                        "spec": "Redistributes reasoning effort across nodes to prevent saturation."
                    },
                    {
                        "name": "SelfCritiqueNode",
                        "spec": "Evaluates internal outputs for blind spots and contradictions."
                    }
                ]
            }
        )


class MockAwareness:
    def get_state_description(self) -> str:
        return "System frequently revisits the same conclusions."

class MockConversation:
    def get_recent_dialogue(self) -> str:
        return "User says responses feel repetitive."

class MockDreaming:
    def get_recent_dream(self) -> str:
        return "A structure that notices its own loops and corrects them."


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentience 5.5 – InsightNode")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8084)
    return p


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
async def amain() -> None:
    args = build_parser().parse_args()

    node = InsightNode(
        awareness=MockAwareness(),
        conversation=MockConversation(),
        dreaming=MockDreaming(),
        llm=MockLLM(),
        evolver=StdoutEvolver(),
    )

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        logger.info("InsightNode HTTP service running on :%d", args.port)
        await asyncio.Event().wait()
    else:
        result = await node.analyze()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(amain())
