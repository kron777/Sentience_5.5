#!/usr/bin/env python3
import os
import time
import json
import psutil
import requests
from datetime import datetime, timezone

def log(level, msg):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] [CAPABILITY_PROBE] [{level}] {msg}")

class CapabilityProbeNode:
    def __init__(self, orchestrator):
        self.node_name = "Capability_Probe_Node"
        self.orchestrator = orchestrator
        log("INFO", "Capability_Probe_Node initialized")
        self.emit_snapshot()

    def emit_snapshot(self):
        snapshot = {
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "ollama_online": False,
            "ollama_models": []
        }

        # Probe Ollama safely
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=1)
            if r.status_code == 200:
                snapshot["ollama_online"] = True
                snapshot["ollama_models"] = [
                    m["name"] for m in r.json().get("models", [])
                ]
        except Exception:
            pass

        event = {
            "type": "capability_snapshot",
            "source": self.node_name,
            "payload": snapshot,
            "salience": 0.4
        }

        self.orchestrator.route_event(event)
        log("INFO", json.dumps(snapshot))
