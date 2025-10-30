#!/usr/bin/env python3
"""
InsightNode – ROS-free, asyncio-first
Generates node suggestions from awareness + conversation + dreaming
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

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
# Protocols (for type-safe mocking or ROS bridging)                          #
# --------------------------------------------------------------------------- #
class AwarenessProvider(Protocol):
    def get_state_description(self) -> str: ...

class ConversationProvider(Protocol):
    def get_recent_dialogue(self) -> str: ...

class DreamingProvider(Protocol):
    def get_recent_dream(self) -> str: ...

class LLMProvider(Protocol):
    async def query(self, prompt: str) -> str: ...

# --------------------------------------------------------------------------- #
# InsightNode                                                                 #
# --------------------------------------------------------------------------- #
class InsightNode:
    """
    ROS-free insight engine.
    Same public API as original.
    """

    def __init__(
        self,
        awareness: AwarenessProvider,
        conversation: ConversationProvider,
        dreaming: DreamingProvider,
        llm: LLMProvider,
    ):
        self.awareness = awareness
        self.conversation = conversation
        self.dreaming = dreaming
        self.llm = llm

    # ------------------------------------------------------------------ #
    # Public API (unchanged signatures)                                  #
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
            f"Given the robot's current situation and context below:\n{context}\n\n"
            "Identify areas where the robot can improve itself by creating new functional nodes. "
            "For each suggested node, provide a name and a short description. "
            "Return the response in valid JSON format as:\n"
            "{\n  \"nodes\": [\n    {\"name\": \"NodeName\", \"spec\": \"Description\"}, ... ]\n}\n"
        )

    async def analyze(self) -> Dict[str, Any]:
        context = self.gather_context()
        prompt = self.construct_prompt(context)
        raw_response = await self.llm.query(prompt)
        try:
            suggestions = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            logger.warning("LLM response not valid JSON – returning empty")
            suggestions = {"nodes": []}
        return suggestions

    # ------------------------------------------------------------------ #
    # HTTP service (optional)                                            #
    # ------------------------------------------------------------------ #
    async def _http_analyze(self, request: web.Request) -> web.Response:
        suggestions = await self.analyze()
        return web.json_response(suggestions)

    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([web.get("/analyze", self._http_analyze)])
        return app


# --------------------------------------------------------------------------- #
# Stand-alone LLM adapters                                                    #
# --------------------------------------------------------------------------- #
class MockLLM:
    """Synchronous mock – wrap for async usage."""

    async def query(self, prompt: str) -> str:
        logger.debug("LLM prompt:\n%s", prompt)
        await asyncio.sleep(0.1)  # simulate network
        return json.dumps(
            {
                "nodes": [
                    {"name": "EnergyManagerNode", "spec": "Manage battery usage efficiently."},
                    {"name": "TaskSchedulerNode", "spec": "Optimise task order to meet deadlines."},
                ]
            }
        )


# --------------------------------------------------------------------------- #
# Mock providers (identical to original)                                      #
# --------------------------------------------------------------------------- #
class MockAwareness:
    def get_state_description(self) -> str:
        return "Battery drains too quickly and task scheduling is inefficient."


class MockConversation:
    def get_recent_dialogue(self) -> str:
        return "User mentioned that tasks are often late and system overheats."


class MockDreaming:
    def get_recent_dream(self) -> str:
        return "Imagined a node balancing power usage dynamically."


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentience 5.0 – InsightNode (ROS-free)")
    p.add_argument("--serve", action="store_true", help="run HTTP service")
    p.add_argument("--port", type=int, default=8084, help="HTTP port")
    return p


# --------------------------------------------------------------------------- #
 Entry-point                                                               #
# --------------------------------------------------------------------------- #
async def amain() -> None:
    args = build_parser().parse_args()

    # wire mocks – replace with real providers via CLI or env vars later
    awareness = MockAwareness()
    conversation = MockConversation()
    dreaming = MockDreaming()
    llm = MockLLM()

    node = InsightNode(awareness, conversation, dreaming, llm)

    if args.serve:
        app = node.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        logger.info("Insight HTTP service on :%d", args.port)
        await asyncio.Event().wait()  # forever
    else:
        suggestions = await node.analyze()
        print("InsightNode Suggestions:")
        for n in suggestions.get("nodes", []):
            print(f"- {n['name']}: {n['spec']}")


if __name__ == "__main__":
    asyncio.run(amain())
