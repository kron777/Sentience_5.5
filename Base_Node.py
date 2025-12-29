#!/usr/bin/env python3
"""
Base_Node — Sentience 5.5
STEP 2: Mandatory node contract

Every node MUST inherit from this class.
No exceptions.
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime, timezone


def log(node: str, level: str, msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] [{node}] [{level}] {msg}")


class BaseNode(ABC):
    def __init__(
        self,
        node_name: str,
        self_model,
        heartbeat_interval: float = 1.0,
    ):
        self.node_name = node_name
        self.self_model = self_model
        self.heartbeat_interval = heartbeat_interval

        self._alive = True
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Mandatory registration
        self.self_model.register_node(self.node_name)

        log(self.node_name, "INFO", "Node initialized")

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------
    def start(self):
        log(self.node_name, "INFO", "Node starting")
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()
        self.run()

    def shutdown(self):
        log(self.node_name, "INFO", "Node shutting down")
        self._alive = False

    # --------------------------------------------------
    # Heartbeat
    # --------------------------------------------------
    def _heartbeat_loop(self):
        while self._alive:
            self.self_model.heartbeat(self.node_name)
            time.sleep(self.heartbeat_interval)

    # --------------------------------------------------
    # Required behavior
    # --------------------------------------------------
    @abstractmethod
    def run(self):
        """
        Main node execution loop.
        MUST be implemented.
        """
        pass
