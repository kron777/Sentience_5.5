#!/usr/bin/env python3
"""
Reasoning_Node.py
Sentience 5.5 – Final expanded self-model reasoning
"""

class ReasoningNode:
    def __init__(self, knowledge, memory):
        self.knowledge = knowledge
        self.memory = memory

        # Expanded explicit world/self rules
        self.rules = {
            # Self architecture
            "specs": "Python 3, local event-driven, modular nodes, bounded memory, no external access",
            "architecture": "Orchestrator, Knowledge_Node, Reasoning_Node, Memory_Node, Nonsense_Node",
            "destiny": "To process input grounded in explicit facts, remain inspectable, and avoid hallucination or theatre",
            "purpose": "Implement the Doctrine: grounded, honest, calm cognition",
            "creator": "Built by you, following the Doctrine dated 2025-12-29",

            # Sensory / body
            "see": False,
            "hear": False,
            "touch": False,
            "balls": False,
            "fingers": False,
            "body": False,
            "clean": False,

            # Understanding self
            "understand yourself": "I have a complete explicit self-model. I know exactly what I am and what I am not.",
        }

    def generate_response(self, text: str) -> str | None:
        lowered = text.lower().strip().rstrip("?")

        # Self-spec questions
        if any(q in lowered for q in ["specs", "specifications", "hardware", "software"]):
            return f"My specifications: {self.rules['specs']}."

        if any(q in lowered for q in ["destiny", "purpose", "goal", "mission"]):
            return self.rules["destiny"]

        if "understand yourself" in lowered or "know yourself" in lowered:
            return self.rules["understand yourself"]

        # Body/sensory questions
        if any(word in lowered for word in ["balls", "fingers", "eyes", "ears", "hands", "see", "hear", "touch", "clean"]):
            return "No. I have no body or senses. I am software."

        # How many fingers / visual questions
        if "how many fingers" in lowered or "holding up" in lowered:
            return "I cannot see you. I have no vision or physical presence."

        # Crude/body jokes
        if "balls" in lowered:
            return "I have no body. That question doesn't apply."

        # Default self-denial for unknown attributes
        if lowered.startswith(("are you ", "do you ", "have you ", "can you ")):
            attribute = lowered.split(" ", 3)[-1] if len(lowered.split()) > 3 else lowered.split(" ", 2)[-1]
            if attribute in self.rules:
                val = self.rules[attribute]
                return f"Yes. {val}" if val is True else "No. I have no body or physical traits."
            return "No. That does not apply to me. I am software."

        # Let other cases fall through if needed
        return None
