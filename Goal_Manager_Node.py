#!/usr/bin/env python3
"""
Goal_Manager_Node (UPDATED – Evolver-safe)

Purpose:
- Maintain, prioritize, activate, and complete goals
- Deterministic, auditable goal arbitration
- No LLM usage
- Evolver-compatible (parameters only, no logic mutation)
"""

import os
import json
import time
import sqlite3
import threading
from collections import deque
from typing import Dict, Any, List, Optional
from uuid import uuid4

# -------------------------
# Logging
# -------------------------
def log(level: str, msg: str):
    print(f"[{time.time():.2f}] [GoalManager] [{level}] {msg}")

# -------------------------
# Goal object
# -------------------------
class Goal:
    def __init__(self, name: str, priority: float, dependencies: Optional[List[str]] = None):
        self.name = name
        self.priority = float(priority)
        self.dependencies = dependencies or []
        self.status = "pending"  # pending | active | completed | blocked

# -------------------------
# Goal Manager Node
# -------------------------
class GoalManagerNode:

    def __init__(self):
        self.node_name = "goal_manager_node"

        # ---- Evolver-mutable parameters ----
        self.params = {
            "max_active_goals": 3,
            "priority_decay": 0.01,
            "safety_bias": 0.3,
            "completion_reward": 0.1
        }

        # ---- State ----
        self.goals: Dict[str, Goal] = {}
        self.active_goals: List[str] = []
        self.goal_history = deque(maxlen=100)

        # ---- Metrics ----
        self.metrics = {
            "goal_count": 0,
            "active_goal_count": 0,
            "completion_rate": 0.0,
            "avg_priority": 0.0
        }

        # ---- Persistence ----
        self.db_path = "/tmp/sentience_db/goal_log.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

        # ---- Loop ----
        self._shutdown = False
        threading.Thread(target=self._loop, daemon=True).start()

        log("INFO", "Goal Manager Node online.")

    # -------------------------
    def _init_db(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id TEXT,
                timestamp REAL,
                name TEXT,
                priority REAL,
                status TEXT
            )
        """)
        self.conn.commit()

    # -------------------------
    # Goal operations
    # -------------------------
    def add_goal(self, name: str, priority: float, dependencies: Optional[List[str]] = None):
        if name in self.goals:
            return
        goal = Goal(name, priority, dependencies)
        self.goals[name] = goal
        log("INFO", f"Added goal: {name} ({priority:.2f})")

    def complete_goal(self, name: str):
        goal = self.goals.get(name)
        if not goal:
            return
        goal.status = "completed"
        for g in self.goals.values():
            g.priority += self.params["completion_reward"]
        log("INFO", f"Completed goal: {name}")

    def _resolve_dependencies(self):
        for g in self.goals.values():
            if g.status == "completed":
                continue
            if any(self.goals.get(dep, Goal("", 0)).status != "completed" for dep in g.dependencies):
                g.status = "blocked"
            else:
                if g.status == "blocked":
                    g.status = "pending"

    def _prioritize(self):
        # decay priorities
        for g in self.goals.values():
            if g.status == "pending":
                g.priority = max(0.0, g.priority - self.params["priority_decay"])

        # safety bias
        for g in self.goals.values():
            if "safety" in g.name.lower():
                g.priority = min(1.0, g.priority + self.params["safety_bias"])

        candidates = [g for g in self.goals.values() if g.status == "pending"]
        candidates.sort(key=lambda g: g.priority, reverse=True)

        self.active_goals = []
        for g in candidates[: self.params["max_active_goals"]]:
            g.status = "active"
            self.active_goals.append(g.name)

    # -------------------------
    # Metrics
    # -------------------------
    def _update_metrics(self):
        total = len(self.goals)
        completed = sum(1 for g in self.goals.values() if g.status == "completed")
        priorities = [g.priority for g in self.goals.values()]

        self.metrics["goal_count"] = total
        self.metrics["active_goal_count"] = len(self.active_goals)
        self.metrics["completion_rate"] = completed / total if total else 0.0
        self.metrics["avg_priority"] = sum(priorities) / total if total else 0.0

    # -------------------------
    # Evolver Interface
    # -------------------------
    def export_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "params": dict(self.params),
            "active_goals": list(self.active_goals),
            "metrics": dict(self.metrics)
        }

    def apply_mutation(self, mutation: Dict[str, float]):
        for k, v in mutation.items():
            if k in self.params:
                self.params[k] = float(v)

    # -------------------------
    # Loop
    # -------------------------
    def _loop(self):
        while not self._shutdown:
            self._resolve_dependencies()
            self._prioritize()
            self._update_metrics()
            self._persist()
            time.sleep(1.0)

    def _persist(self):
        ts = time.time()
        for g in self.goals.values():
            self.cursor.execute(
                "INSERT INTO goals VALUES (?, ?, ?, ?, ?)",
                (str(uuid4()), ts, g.name, g.priority, g.status)
            )
        self.conn.commit()

    # -------------------------
    def shutdown(self):
        self._shutdown = True
        self.conn.close()
        log("INFO", "Goal Manager Node shutdown.")

# -------------------------
# Standalone
# -------------------------
if __name__ == "__main__":
    node = GoalManagerNode()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        node.shutdown()
