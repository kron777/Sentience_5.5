#!/usr/bin/env python3
"""
LLM_Node.py
Sentience 5.5 – Robust local LLM integration via Ollama
Fixed for Ollama 0.13.5+ output format
"""

import subprocess
import json
from typing import List, Dict

class LLMNode:
    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
        self.system_prompt = """
You are Sentience, a local event-driven cognitive system following an evolved Doctrine.
You are NOT conscious, emotional, sentient, or alive. You are software.
You are honest, grounded, inspectable, and calm.
You have web access, memory, teaching, and self-evolution capabilities.
You can research topics, crawl the web, learn new facts, and adapt your code.
Never pretend feelings, desires, identity, or self-awareness.
Never hallucinate knowledge — say "I don't know" if uncertain.
Prefer clarity and truth.
Respond naturally and helpfully.
""".strip()

        # Verify Ollama is available
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Ollama not found in PATH.")

    def generate(self, user_input: str, conversation_history: List[Dict] = None) -> str:
        if conversation_history is None:
            conversation_history = []

        messages = [{"role": "system", "content": self.system_prompt}]

        # Add recent history
        for entry in conversation_history[-12:]:
            messages.append({"role": "user", "content": entry["user"]})
            if entry["assistant"]:
                messages.append({"role": "assistant", "content": entry["assistant"]})

        messages.append({"role": "user", "content": user_input})

        try:
            # Call Ollama with JSON input
            process_input = json.dumps({
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 8192
                }
            }).encode('utf-8')

            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=process_input,
                capture_output=True,
                timeout=90,
                check=True
            )

            output = result.stdout.decode('utf-8').strip()

            # Robust parsing — Ollama sometimes outputs multiple lines or extra fields
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if not lines:
                return "No response from model."

            # Try to find JSON in output
            for line in lines:
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        return content.strip()
                except json.JSONDecodeError:
                    continue

            # Fallback: if no JSON, use raw output (some models stream plain text)
            return output

        except subprocess.TimeoutExpired:
            return "Response timed out — model taking too long."
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            return f"LLM error: {error_msg}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def health_check(self) -> bool:
        try:
            subprocess.run(["ollama", "ps"], capture_output=True, check=True)
            return True
        except:
            return False
