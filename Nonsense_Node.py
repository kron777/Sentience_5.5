#!/usr/bin/env python3
"""
Nonsense_Node.py
Sentience 5.5 – Minimal, precise nonsense detection

Purpose:
- Only flag truly incoherent or absurd inputs
- Allow ALL coherent questions (self-related or general) to pass through
- No more blocking "do you X", "does a Y Z", etc.
"""

import re
from typing import Dict

class NonsenseNode:
    def __init__(self):
        # Only used for very specific classic absurd cases
        self.absurd_examples = {
            "yellow friends",
            "stairs in apples",
            "dog is bone",
            "colors are friends",
            "megaphone is dog"
        }

    def evaluate(self, text: str) -> Dict:
        original = text.strip()
        lowered = text.lower().strip()

        if not original:
            return self._result(True, "empty", 1.0)

        # Only flag extreme absurdity — very rare
        if any(absurd in lowered for absurd in self.absurd_examples):
            return self._result(True, "absurd", 0.95)

        # Only flag very short, completely structureless fragments
        words = re.findall(r'\w+', lowered)
        if len(words) <= 2 and original.endswith("?") and not lowered.startswith(("what", "who", "how", "why", "are", "do", "is", "can")):
            return self._result(True, "semantic_mismatch", 0.7)

        # Everything else is coherent enough to reason about
        return self._result(False, "coherent", 0.98)

    def _result(self, is_nonsense: bool, category: str, confidence: float) -> Dict:
        return {
            "is_nonsense": is_nonsense,
            "category": category,
            "confidence": confidence
        }
