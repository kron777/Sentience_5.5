#!/usr/bin/env python3
"""
Meta_Cognition_Node — Sentience 5.5

Role:
- Observe system-wide cognition via CognitiveEventBus
- Detect repetition, drift, silence, dominance
- Emit correction and insight events
- NEVER fabricate confidence
"""

import time
from collections import Counter, deque
from typing import Dict, Any
from datetime import datetime, timezone

from Cognitive_Event_Bus import CognitiveEventBus


# --------------------------------------------------
# Logging
# --------------------------------------------------
def log(level: str, msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] [META_COG] [{level}] {msg}")


# --------------------------------------------------
# Meta Cognition Node
# --------------------------------------------------
class MetaCognitionNode:
    def __init__(self, bus: CognitiveEventBus):
        self.node_name = "Meta_Cognition_Node"
        self.bus = bus

        self.window = deque(maxlen=100)
        self.last_analysis = 0.0
        self.analysis_interval = 3.0  # seconds

        # Subscriptions
        self.bus.subscribe("external_input", self.observe)
        self.bus.subscribe("conversation_output", self.observe)
        self.bus.subscribe("memory_stored", self.observe)
        self.bus.subscribe("node_heartbeat", self.observe)

        log("INFO", "Meta-Cognition Node initialized")

    # --------------------------------------------------
    # Observation
    # --------------------------------------------------
    def observe(self, event: Dict[str, Any]):
        self.window.append(event)
        now = time.time()

        if now - self.last_analysis >= self.analysis_interval:
            self.last_analysis = now
            self.analyze()

    # --------------------------------------------------
    # Analysis
    # --------------------------------------------------
    def analyze(self):
        if not self.window:
            return

        sources = [e["source"] for e in self.window]
        event_types = [e["event_type"] for e in self.window]

        source_counts = Counter(sources)
        event_counts = Counter(event_types)

        issues = []

        # 1. Repetition detection
        if event_counts.get("conversation_output", 0) >= 5:
            issues.append("REPETITION_RISK")

        # 2. Node dominance
        dominant = source_counts.most_common(1)[0]
        if dominant[1] / len(self.window) > 0.6:
            issues.append(f"NODE_DOMINANCE:{dominant[0]}")

        # 3. Silent memory
        if event_counts.get("memory_stored", 0) == 0:
            issues.append("MEMORY_INACTIVE")

        # 4. Input without synthesis
        if (
            event_counts.get("external_input", 0) > 0
            and event_counts.get("conversation_output", 0) == 0
        ):
            issues.append("NO_RESPONSE_GENERATED")

        # --------------------------------------------------
        # Emit findings
        # --------------------------------------------------
        if issues:
            self.emit_correction(issues)
        else:
            self.emit_health()

    # --------------------------------------------------
    # Emitters
    # --------------------------------------------------
    def emit_correction(self, issues):
        payload = {
            "issues_detected": issues,
            "window_size": len(self.window),
        }

        self.bus.publish(
            event_type="meta_cognitive_alert",
            source=self.node_name,
            payload=payload,
            salience=0.9,
        )

        log("WARN", f"Issues detected: {issues}")

    def emit_health(self):
        self.bus.publish(
            event_type="meta_cognitive_health",
            source=self.node_name,
            payload={"status": "STABLE"},
            salience=0.2,
        )

        log("INFO", "System cognitive state stable")
