#!/usr/bin/env python3
"""
Knowledge_Consolidation_Node
Sentience 5.5

Role:
- Consolidate learned knowledge
- Prevent memory bloat
- Maintain epistemic integrity
"""

import time
from datetime import datetime, timezone
from typing import Dict, List


def log(level: str, msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] [KNOWLEDGE] [{level}] {msg}")


class KnowledgeConsolidationNode:
    def __init__(self, memory_node=None):
        self.node_name = "Knowledge_Consolidation_Node"
        self.memory = memory_node

        # Tunables
        self.max_items_per_topic = 5
        self.relevance_decay = 0.98

        log("INFO", "Knowledge consolidation node initialized")

    # -------------------------------------------------
    # Entry point
    # -------------------------------------------------
    def consolidate(self):
        if not self.memory:
            log("WARN", "No memory node attached")
            return

        knowledge = self.memory.get_external_knowledge()
        if not knowledge:
            return

        grouped = self._group_by_topic(knowledge)

        consolidated = {}
        for topic, items in grouped.items():
            consolidated[topic] = self._consolidate_topic(items)

        self.memory.replace_consolidated_knowledge(consolidated)
        log("INFO", f"Consolidated {len(consolidated)} knowledge topics")

    # -------------------------------------------------
    # Internals
    # -------------------------------------------------
    def _group_by_topic(self, knowledge: List[Dict]) -> Dict[str, List[Dict]]:
        topics = {}
        for item in knowledge:
            topic = self._infer_topic(item["summary"])
            topics.setdefault(topic, []).append(item)
        return topics

    def _infer_topic(self, text: str) -> str:
        """
        Extremely conservative topic inference.
        No guessing, no creativity.
        """
        words = text.lower().split()
        if not words:
            return "unknown"

        return words[0]  # first keyword anchor

    def _consolidate_topic(self, items: List[Dict]) -> Dict:
        # Sort newest first
        items = sorted(items, key=lambda x: x["timestamp"], reverse=True)

        # Trim
        items = items[: self.max_items_per_topic]

        # Decay relevance
        now = time.time()
        for item in items:
            age = now - item["timestamp"]
            item["relevance"] = max(0.1, self.relevance_decay ** (age / 3600))

        summary = self._merge_summaries(items)

        return {
            "summary": summary,
            "sources": [i["source"] for i in items],
            "last_updated": items[0]["timestamp"]
        }

    def _merge_summaries(self, items: List[Dict]) -> str:
        """
        Deterministic merge.
        No interpretation.
        """
        seen = set()
        merged = []

        for item in items:
            for sentence in item["summary"].split(". "):
                if sentence not in seen:
                    seen.add(sentence)
                    merged.append(sentence)

        return ". ".join(merged) + "."
