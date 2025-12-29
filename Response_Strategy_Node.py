#!/usr/bin/env python3
"""
Response_Strategy_Node
Sentience 5.5

Role:
- Decide response strategy based on input classification
- Prevent repetition and evasive boilerplate
- Keep system grounded, human-readable, and adaptive
"""

from datetime import datetime, timezone
from typing import Dict


def log(level: str, msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] [STRATEGY] [{level}] {msg}")


class ResponseStrategyNode:
    def __init__(self):
        self.node_name = "Response_Strategy_Node"
        log("INFO", "Response strategy online")

    # -------------------------------------------------
    # Strategy selection
    # -------------------------------------------------
    def decide(self, analysis: Dict[str, str]) -> Dict[str, str]:
        classification = analysis.get("classification", "valid")

        if classification == "category_error":
            return self._category_error()

        if classification == "nonsense":
            return self._nonsense()

        if classification == "metaphor":
            return self._metaphor()

        # Default: valid input
        return self._direct_answer()

    # -------------------------------------------------
    # Strategies
    # -------------------------------------------------

    def _category_error(self) -> Dict[str, str]:
        return {
            "mode": "explain_error",
            "tone": "calm",
            "message": (
                "That question mixes categories that don’t compute together. "
                "If you rephrase it so the concepts belong to the same kind of thing, "
                "I can answer it clearly."
            )
        }

    def _nonsense(self) -> Dict[str, str]:
        return {
            "mode": "ground",
            "tone": "neutral",
            "message": (
                "I don’t think that question resolves to a meaningful statement. "
                "What are you trying to explore with it?"
            )
        }

    def _metaphor(self) -> Dict[str, str]:
        return {
            "mode": "interpret",
            "tone": "open",
            "message": (
                "That sounds metaphorical. If you’d like, I can interpret it symbolically, "
                "or you can restate it literally."
            )
        }

    def _direct_answer(self) -> Dict[str, str]:
        return {
            "mode": "answer",
            "tone": "direct",
            "message": ""
        }
