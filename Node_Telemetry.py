# node_telemetry.py
import time
import threading


class NodeTelemetry:
    """
    Canonical telemetry object shared by all nodes.
    This is the ONLY source of truth for node health.
    """

    def __init__(self, name: str):
        self.name = name
        self.last_heartbeat = time.time()
        self.status = "INIT"
        self.load = 0.0
        self.errors = 0
        self.lock = threading.Lock()

    def heartbeat(self, load: float = 0.0):
        with self.lock:
            self.last_heartbeat = time.time()
            self.status = "ALIVE"
            self.load = float(load)

    def error(self):
        with self.lock:
            self.errors += 1
            self.status = "ERROR"

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "node": self.name,
                "status": self.status,
                "last_heartbeat": round(self.last_heartbeat, 3),
                "load": round(self.load, 3),
                "errors": self.errors
            }

    def is_responsive(self, timeout: float = 3.0) -> bool:
        with self.lock:
            return (time.time() - self.last_heartbeat) <= timeout
