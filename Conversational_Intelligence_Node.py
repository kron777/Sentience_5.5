#!/usr/bin/env python3
"""
Conversational_Intelligence_Node.py
Sentience 5.5 – Fluent intelligence via local LLM
Final evolved version
"""

import random
import re

class ConversationalIntelligenceNode:
    def __init__(self, memory, nonsense, knowledge, reasoning, crawler, evolver, llm):
        self.memory = memory
        self.nonsense = nonsense
        self.knowledge = knowledge
        self.reasoning = reasoning
        self.crawler = crawler
        self.evolver = evolver
        self.llm = llm

        # Build conversation history for LLM context
        self.history = []  # List of {"user": str, "assistant": str}

    def respond(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        lowered = text.lower()

        # Update history
        self.history.append({"user": text, "assistant": ""})  # placeholder

        # Nonsense filter (still useful for extreme cases)
        if self.nonsense.evaluate(text)["is_nonsense"]:
            response = "That input doesn't form coherent meaning."
            self.history[-1]["assistant"] = response
            self.memory.store(text, "nonsense", response)
            return response

        # === SPECIAL COMMANDS (keep rule-based for control) ===
        if any(lowered.startswith(cmd) for cmd in ["crawl ", "search ", "research ", "look up "]):
            query = text.split(" ", 2)[2] if len(text.split()) > 2 else text.split(" ", 1)[1]
            response = f"Crawling the web for: {query}"
            self.history[-1]["assistant"] = response
            return response + "\nResults will be processed and may trigger evolution."

        if any(lowered.startswith(cmd) for cmd in ["evolve ", "learn about ", "adapt to ", "study ", "improve on "]):
            topic = text.split(" ", 2)[2] if len(text.split()) > 2 else ""
            if topic:
                response = self.evolver.evolve_from_query(topic)
            else:
                response = "Please specify a topic: 'evolve neural networks'"
            self.history[-1]["assistant"] = response
            return response

        if any(phrase in lowered for phrase in ["evolution status", "how have you evolved", "evolver status"]):
            response = self.evolver.status()
            self.history[-1]["assistant"] = response
            return response

        # Teaching
        if lowered.startswith(("teach me that ", "remember that ", "learn that ")):
            parts = text.split(" ", 3)
            if len(parts) >= 4 and "=" in parts[3]:
                kv = parts[3].split("=", 1)
                key = kv[0].strip()
                value = kv[1].strip().strip('"\'')
                response = self.knowledge.teach_fact(key, value)
            else:
                response = "Format: teach me that <key> = <value>"
            self.history[-1]["assistant"] = response
            return response

        if any(phrase in lowered for phrase in ["what have you learned", "list facts", "show learned"]):
            response = self.knowledge.list_learned()
            self.history[-1]["assistant"] = response
            return response

        # === NORMAL CONVERSATION → LLM ===
        try:
            response = self.llm.generate(text, self.history[:-1])  # exclude current placeholder
            if not response or "error" in response.lower():
                response = "I encountered an issue with my language module. Falling back to basic response."
        except Exception as e:
            response = f"Language module unavailable: {str(e)}"

        # Update history
        self.history[-1]["assistant"] = response
        self.memory.store(text, "llm", response)

        # Rare metaphor
        if random.random() < 0.025:
            fragments = [
                "like turning a stone over to see what's beneath",
                "as if testing whether two shapes really fit",
                "a bit like asking if shadows can shake hands",
            ]
            response += f" — {random.choice(fragments)}."

        return response
