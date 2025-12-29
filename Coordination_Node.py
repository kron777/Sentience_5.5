#!/usr/bin/env python3
"""
Knowledge_Node.py
Sentience 5.5 – Explicit, bounded, inspectable fact store

Purpose:
- Central source of grounded truth
- Fully visible at runtime
- No hidden weights, no training
- Doctrine Section 4 & 10 compliant
"""

from typing import Optional, Dict, Any

class KnowledgeNode:
    def __init__(self):
        # Core self-model — explicit and exhaustive
        self.self_model: Dict[str, Any] = {
            "name": "Sentience",
            "type": "local event-driven cognitive system",
            "version": "5.5",
            "current_date": "2025-12-29",
            "conscious": False,
            "sentient": False,
            "emotional": False,
            "biological": False,
            "body": False,
            "color": False,
            "location": "running on local hardware",
            "personal_relations": False,
            "desires": False,
            "knowledge_source": "explicit facts, runtime state, conversation memory",
            "capabilities": [
                "process text input",
                "detect incoherence",
                "recall conversation history",
                "respond from grounded facts",
                "admit uncertainty"
            ],
            "limitations": [
                "no internet access",
                "no emotions",
                "no independent action",
                "no persistent learning without explicit storage"
            ]
        }

    def get(self, key: str) -> Optional[Any]:
        """Direct lookup — returns None if not present"""
        return self.self_model.get(key.lower())

    def has_attribute(self, attribute: str) -> bool:
        """For yes/no questions about self"""
        val = self.get(attribute)
        if val is not None:
            return bool(val)
        # Default denial for unknown attributes
        return False

    def query_capability(self, action: str) -> bool:
        """Check if system can perform an action"""
        capabilities = self.self_model["capabilities"]
        return any(action.lower() in cap.lower() for cap in capabilities)

    def describe_self(self) -> str:
        """Concise self-description — used in responses"""
        return (
            f"I am {self.self_model['name']}, "
            f"a {self.self_model['type']}. "
            f"I have no consciousness, emotions, or body."
        )

    def snapshot(self) -> Dict[str, Any]:
        """Introspection — full view of known facts"""
        return dict(self.self_model)
