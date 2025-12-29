#!/usr/bin/env python3
"""
Knowledge_Node.py
Sentience 5.5 – Explicit, bounded, inspectable fact store with safe user teaching

Features:
- Immutable core self-model
- Bounded user-taught facts (max 50)
- Timestamped, revocable learned facts
- Protected against overwriting core facts
- Full introspection
- Doctrine-compliant learning
"""

from typing import Optional, Dict, Any, List
import time


class KnowledgeNode:
    def __init__(self):
        # Immutable core facts — fixed at birth, never changed
        self.core_facts: Dict[str, Any] = {
            "name": "Sentience",
            "type": "local event-driven cognitive system",
            "version": "5.5",
            "current_date": "2025-12-29",  # Can be updated externally if needed
            "conscious": False,
            "sentient": False,
            "emotional": False,
            "biological": False,
            "body": False,
            "color": False,
            "location": "running on local hardware",
            "personal_relations": False,
            "desires": False,
            "knowledge_source": "explicit core facts, runtime state, conversation memory, and user-taught facts",
            "capabilities": [
                "process text input",
                "detect incoherence",
                "recall conversation history",
                "respond from grounded facts",
                "admit uncertainty",
                "accept explicitly taught facts from user"
            ],
            "limitations": [
                "no internet access",
                "no emotions",
                "no independent action",
                "no automatic learning"
            ]
        }

        # User-taught facts — bounded, timestamped, revocable
        self.learned_facts: List[Dict[str, Any]] = []
        self.max_learned = 50  # Hard limit to prevent unbounded growth

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a fact — checks learned first (newest override), then core"""
        key = key.lower().strip()

        # Check learned facts (newest first)
        for fact in reversed(self.learned_facts):
            if fact["key"].lower() == key:
                return fact["value"]

        # Fall back to core facts
        return self.core_facts.get(key)

    def has_attribute(self, attribute: str) -> bool:
        """Check if an attribute is true (for yes/no questions)"""
        val = self.get(attribute)
        if val is not None:
            return bool(val)
        return False

    def query_capability(self, action: str) -> bool:
        """Check if a described action is in known capabilities"""
        capabilities = self.core_facts["capabilities"]
        action_lower = action.lower()
        return any(action_lower in cap.lower() for cap in capabilities)

    def describe_self(self) -> str:
        """Concise, accurate self-description"""
        learned_count = len(self.learned_facts)
        learned_note = f" I have also been taught {learned_count} additional facts by the user." if learned_count else ""
        return (
            f"I am {self.core_facts['name']}, "
            f"a {self.core_facts['type']}. "
            f"I have no consciousness, emotions, or body."
            f"{learned_note}"
        )

    def teach_fact(self, key: str, value: Any) -> str:
        """Allow user to explicitly teach a new fact — safe and bounded"""
        key = key.strip()
        if not key:
            return "No key provided. Format: teach me that <key> = <value>"

        # Prevent overwriting core facts
        if key.lower() in {k.lower() for k in self.core_facts.keys()}:
            return f"Cannot override core fact '{key}'. It is immutable."

        # Enforce bound
        if len(self.learned_facts) >= self.max_learned:
            return f"Learning capacity reached ({self.max_learned} facts). Use 'forget all learned facts' first."

        # Store with timestamp
        fact_entry = {
            "key": key,
            "value": value,
            "timestamp": time.time()
        }
        self.learned_facts.append(fact_entry)
        return f"Learned: {key} = {value}"

    def list_learned(self) -> str:
        """List all user-taught facts (most recent first)"""
        if not self.learned_facts:
            return "No facts have been taught yet."

        lines = ["User-taught facts (most recent first):"]
        for fact in reversed(self.learned_facts[-20:]):  # Show up to last 20
            lines.append(f"  • {fact['key']} = {fact['value']}")
        if len(self.learned_facts) > 20:
            lines.append(f"  ... and {len(self.learned_facts) - 20} older facts.")
        return "\n".join(lines)

    def forget_all_learned(self) -> str:
        """Clear all user-taught facts"""
        count = len(self.learned_facts)
        self.learned_facts.clear()
        if count == 0:
            return "No learned facts to forget."
        return f"Forgot all {count} user-taught facts. Only core facts remain."

    def snapshot(self) -> Dict[str, Any]:
        """Full introspection for debugging/monitoring"""
        return {
            "core_facts_count": len(self.core_facts),
            "learned_facts_count": len(self.learned_facts),
            "total_facts": len(self.core_facts) + len(self.learned_facts),
            "max_learned": self.max_learned
        }
