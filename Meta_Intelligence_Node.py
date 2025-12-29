#!/usr/bin/env python3
import requests, json, time

OLLAMA_URL = "http://localhost:11434/api/generate"
MEMORY_RETRIEVE = "http://localhost:8091/memory/retrieve"
MEMORY_STORE = "http://localhost:8091/memory/store"

class MetaIntelligenceNode:
    def __init__(self):
        self.identity = "INDIGO"
        self.mode = "LIVE"
        self.model = "phi"

    def retrieve_memory(self, query):
        try:
            r = requests.post(MEMORY_RETRIEVE, json={
                "query": query,
                "num_results": 5
            }, timeout=2)
            return "\n".join(m["content"] for m in r.json().get("memories", []))
        except:
            return ""

    def store_memory(self, content, salience=0.6):
        try:
            requests.post(MEMORY_STORE, json={
                "category": "dialogue",
                "content": content,
                "salience": salience
            }, timeout=2)
        except:
            pass

    def self_model(self):
        return {
            "identity": self.identity,
            "mode": self.mode,
            "memory": True,
            "learning": True,
            "time": time.time()
        }

    def think(self, user_input):
        memories = self.retrieve_memory(user_input)
        self_state = json.dumps(self.self_model(), indent=2)

        prompt = f"""
You are the Meta-Intelligence of a running cognitive system.

Self-model:
{self_state}

Relevant memories:
{memories}

User says:
{user_input}

Respond naturally, intelligently, and conversationally.
"""

        r = requests.post(OLLAMA_URL, json={
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }, timeout=30)

        reply = r.json()["response"]

        self.store_memory(f"User: {user_input}\nMe: {reply}", 0.7)
        return reply
