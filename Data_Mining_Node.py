#!/usr/bin/env python3
"""
Data_Mining_Node.py

ROLE:
- Detect patterns, trends, anomalies, correlations
- Operates ONLY on supplied data windows
- Produces structured, verifiable outputs
- NEVER issues decisions, ethics, or control actions

This node is a TOOL NODE, not a cognitive agent.
"""

import os, sys, json, time, uuid, sqlite3, argparse, threading, asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List
from collections import deque

# ---------------- Logging ----------------

def _log(level, node, msg):
    print(f"[{datetime.now().isoformat()}] {node} [{level}] {msg}")

def info(n,m): _log("INFO",n,m)
def warn(n,m): _log("WARN",n,m)
def error(n,m): _log("ERROR",n,m)
def debug(n,m): _log("DEBUG",n,m)

# ---------------- Config ----------------

def load_config(node_name, path=None):
    return {
        "db_root_path": "/tmp/sentience_db",
        "data_mining_node": {
            "mining_interval": 2.0,
            "salience_threshold": 0.5,
            "context_window_s": 30.0,
            "llm_enabled": True
        },
        "llm": {
            "model": "phi-2",
            "url": "http://localhost:8000/v1/chat/completions",
            "timeout": 45
        }
    }

# ---------------- Node ----------------

class DataMiningNode:
    """
    NON-COGNITIVE SPECIALIST NODE
    """

    def __init__(self, config_path=None):
        self.node = "data_mining_node"
        cfg = load_config(self.node, config_path)

        self.params = cfg["data_mining_node"]
        self.llm_cfg = cfg["llm"]

        self.interval = self.params["mining_interval"]
        self.salience_threshold = self.params["salience_threshold"]
        self.context_window = self.params["context_window_s"]
        self.llm_enabled = self.params["llm_enabled"]

        self.queue: Deque[Dict[str,Any]] = deque()
        self.recent_data: Dict[str,Deque] = {
            "memory": deque(maxlen=50),
            "world": deque(maxlen=50),
            "performance": deque(maxlen=50),
            "bias": deque(maxlen=50),
            "ethics": deque(maxlen=50)
        }

        self._setup_db(cfg["db_root_path"])
        self._setup_async()

        info(self.node, "Data Mining Node online (tool-mode)")

        self._shutdown = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    # ---------------- Database ----------------

    def _setup_db(self, root):
        os.makedirs(root, exist_ok=True)
        self.db = sqlite3.connect(os.path.join(root, "data_mining.db"), check_same_thread=False)
        self.cur = self.db.cursor()
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS mining_results (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            analysis_type TEXT,
            input_sources TEXT,
            insight TEXT,
            extracted_json TEXT,
            confidence REAL
        )
        """)
        self.db.commit()

    # ---------------- Async / LLM ----------------

    def _setup_async(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True).start()
        self.session = None

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _llm(self, prompt):
        if not self.llm_enabled:
            return None
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession()
        payload = {
            "model": self.llm_cfg["model"],
            "messages":[{"role":"user","content":prompt}],
            "temperature":0.1,
            "max_tokens":600
        }
        async with self.session.post(self.llm_cfg["url"], json=payload, timeout=self.llm_cfg["timeout"]) as r:
            out = await r.json()
            return out["choices"][0]["message"]["content"]

    # ---------------- Input API ----------------

    def submit_task(self, task: Dict[str,Any]):
        task["id"] = task.get("id", str(uuid.uuid4()))
        task["timestamp"] = time.time()
        self.queue.append(task)
        debug(self.node, f"Task queued: {task['analysis_type']}")

    def ingest(self, source: str, data: Dict[str,Any]):
        if source in self.recent_data:
            self.recent_data[source].append(data)

    # ---------------- Core Loop ----------------

    def _loop(self):
        while not self._shutdown.is_set():
            if self.queue:
                task = self.queue.popleft()
                self._process(task)
            time.sleep(self.interval)

    # ---------------- Mining ----------------

    def _process(self, task: Dict[str,Any]):
        analysis = task.get("analysis_type","general")
        salience = task.get("salience",0.0)

        snapshot = self._collect(task.get("sources",[]))
        insight, extracted, confidence = self._rule_mine(analysis, snapshot)

        if self.llm_enabled and salience >= self.salience_threshold:
            llm_out = self.loop.run_until_complete(
                self._llm(self._llm_prompt(analysis, snapshot))
            )
            if llm_out:
                try:
                    parsed = json.loads(llm_out)
                    insight = parsed.get("insight", insight)
                    extracted = parsed.get("evidence", extracted)
                    confidence = parsed.get("confidence", confidence)
                except Exception:
                    warn(self.node, "LLM output parse failed, using rule output")

        self._store(task, insight, extracted, confidence)
        info(self.node, f"Mining complete [{analysis}] conf={confidence:.2f}")

    # ---------------- Helpers ----------------

    def _collect(self, sources: List[str]):
        now = time.time()
        out = {}
        for s in sources:
            if s in self.recent_data:
                out[s] = [
                    d for d in self.recent_data[s]
                    if now - d.get("timestamp",now) <= self.context_window
                ]
        return out

    def _rule_mine(self, analysis, data):
        if analysis == "trend":
            scores = [d["score"] for d in data.get("performance",[]) if "score" in d]
            if len(scores) >= 3:
                return (
                    "Performance trend detected",
                    {"avg":sum(scores)/len(scores)},
                    0.6
                )
        return ("No strong pattern",{},0.2)

    def _llm_prompt(self, analysis, data):
        return f"""
Analyze the following structured data.
Task: {analysis}
Return JSON only:

{{"insight": "...","evidence": {{...}},"confidence":0.0}}

DATA:
{json.dumps(data,indent=2)}
"""

    def _store(self, task, insight, extracted, confidence):
        self.cur.execute(
            "INSERT INTO mining_results VALUES (?,?,?,?,?,?,?)",
            (
                task["id"],
                datetime.now().isoformat(),
                task.get("analysis_type"),
                json.dumps(task.get("sources",[])),
                insight,
                json.dumps(extracted),
                confidence
            )
        )
        self.db.commit()

    # ---------------- Shutdown ----------------

    def shutdown(self):
        self._shutdown.set()
        if self.session:
            self.loop.run_until_complete(self.session.close())
        self.db.close()

# ---------------- Main ----------------

if __name__ == "__main__":
    n = DataMiningNode()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        n.shutdown()
