#!/usr/bin/env python3
"""
VoiceInteractionNode – Sentience 5.5 (UPDATED)

Changes / fixes:
- Fully ROS-optional, asyncio-first core
- Fixed missing imports and undefined symbols
- Removed implicit globals (sensory_data, AsyncPhi2Client)
- Deterministic async lifecycle
- Clear separation: input → LLM → response → optional TTS
- Safe fallbacks everywhere
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import asyncio
import sqlite3
import argparse
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque

import aiohttp

# --------------------------------------------------------------------------- #
# Optional audio                                                              #
# --------------------------------------------------------------------------- #
try:
    import speech_recognition as sr
    import pyttsx3
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    sr = None
    pyttsx3 = None

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)

# --------------------------------------------------------------------------- #
# Minimal config loader (safe fallback)                                        #
# --------------------------------------------------------------------------- #
def load_config(node_name: str, path: Optional[str] = None) -> Dict[str, Any]:
    return {
        "db_root_path": "/tmp/sentience_db",
        "ethical_compassion_bias": 0.2,
        "llm": {
            "endpoint": "http://localhost:8000/generate",
            "timeout": 20.0
        }
    }

# --------------------------------------------------------------------------- #
# Async LLM client                                                            #
# --------------------------------------------------------------------------- #
class AsyncLLMClient:
    def __init__(self, endpoint: str, timeout: float):
        self.endpoint = endpoint
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def ensure(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

    async def query(self, prompt: str, temperature: float = 0.6, max_tokens: int = 200) -> str:
        await self.ensure()
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            async with self.session.post(self.endpoint, json=payload) as r:
                r.raise_for_status()
                data = await r.json()
                return data.get("response", "")
        except Exception as e:
            return f"[LLM fallback] {prompt[:60]}"

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

# --------------------------------------------------------------------------- #
# Voice Interaction Node                                                       #
# --------------------------------------------------------------------------- #
class VoiceInteractionNode:
    def __init__(self, config_path: Optional[str] = None):
        self.node_name = "voice_interaction_node"
        cfg = load_config(self.node_name, config_path)

        self.ethical_compassion_bias = cfg.get("ethical_compassion_bias", 0.2)

        llm_cfg = cfg.get("llm", {})
        self.llm = AsyncLLMClient(
            llm_cfg.get("endpoint"),
            llm_cfg.get("timeout", 20.0)
        )

        # DB
        db_root = cfg.get("db_root_path", "/tmp/sentience_db")
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "voice_interaction.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        # State
        self.pending: Deque[str] = deque(maxlen=10)
        self.history: Deque[Dict[str, Any]] = deque(maxlen=100)

        # Audio
        self.recognizer = sr.Recognizer() if HAS_AUDIO else None
        self.tts = pyttsx3.init() if HAS_AUDIO else None

        # Async loop
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        log("INFO", self.node_name, "Initialized")

    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                command TEXT,
                response TEXT
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # ------------------------------------------------------------------ #
    def submit_command(self, command: str):
        self.pending.append(command)

    # ------------------------------------------------------------------ #
    async def _generate_response(self, command: str) -> str:
        prompt = (
            f"Human: {command}\n\n"
            f"Respond clearly, kindly, and helpfully. "
            f"Compassion bias={self.ethical_compassion_bias}."
        )
        return await self.llm.query(prompt)

    # ------------------------------------------------------------------ #
    def _persist(self, command: str, response: str):
        self.conn.execute(
            "INSERT INTO interactions VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), time.time(), command, response)
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    def _speak(self, text: str):
        if HAS_AUDIO and self.tts:
            try:
                self.tts.say(text)
                self.tts.runAndWait()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def step(self):
        if not self.pending:
            return

        command = self.pending.popleft()
        future = asyncio.run_coroutine_threadsafe(
            self._generate_response(command), self.loop
        )
        response = future.result(timeout=30)

        self._persist(command, response)
        self.history.append({"command": command, "response": response})

        log("INFO", self.node_name, f"CMD: {command}")
        log("INFO", self.node_name, f"RSP: {response}")

        self._speak(response)

    # ------------------------------------------------------------------ #
    def run(self):
        log("INFO", self.node_name, "Running")
        try:
            while True:
                self.step()
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.shutdown()

    # ------------------------------------------------------------------ #
    def shutdown(self):
        log("INFO", self.node_name, "Shutting down")
        asyncio.run_coroutine_threadsafe(self.llm.close(), self.loop)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=3)
        self.conn.close()
        if HAS_AUDIO and self.tts:
            self.tts.stop()

# --------------------------------------------------------------------------- #
# Entry                                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str)
    args = p.parse_args()

    node = VoiceInteractionNode(args.config)
    node.submit_command("Hello, how are you?")
    node.run()
