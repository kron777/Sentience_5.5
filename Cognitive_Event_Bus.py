#!/usr/bin/env python3
"""
Cognitive_Event_Bus — Sentience 5.5

Role:
- Central internal event bus for ALL cognitive activity
- Deterministic, inspectable, replayable
- No intelligence, no interpretation — pure coordination

Every node:
- Publishes events here
- Subscribes to events here
"""

import time
import threading
from typing import Callable, Dict, List, Any
from collections import deque
from datetime import datetime, timezone


# --------------------------------------------------
# Logging
# --------------------------------------------------
def log(level: str, msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] [EVENT_BUS] [{level}] {msg}")


# --------------------------------------------------
# Cognitive Event Bus
# --------------------------------------------------
class CognitiveEventBus:
    def __init__(self, history_limit: int = 500):
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self.event_history = deque(maxlen=history_limit)
        self._lock = threading.Lock()

        log("INFO", "Cognitive Event Bus initialized")

    # --------------------------------------------------
    # Subscription
    # --------------------------------------------------
    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        with self._lock:
            self.subscribers.setdefault(event_type, []).append(handler)
        log("INFO", f"Subscriber registered for event type: {event_type}")

    # --------------------------------------------------
    # Publish
    # --------------------------------------------------
    def publish(
        self,
        event_type: str,
        source: str,
        payload: Dict[str, Any],
        salience: float = 0.5,
    ):
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "source": source,
            "salience": salience,
            "payload": payload,
        }

        with self._lock:
            self.event_history.append(event)
            handlers = list(self.subscribers.get(event_type, []))

        log("EVENT", f"{event_type} from {source}")

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                log("ERROR", f"Handler failure in {event_type}: {e}")

    # --------------------------------------------------
    # Introspection
    # --------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "registered_event_types": list(self.subscribers.keys()),
                "subscriber_counts": {
                    k: len(v) for k, v in self.subscribers.items()
                },
                "event_history_size": len(self.event_history),
                "recent_events": list(self.event_history)[-10:],
            }
