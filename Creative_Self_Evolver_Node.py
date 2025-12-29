#!/usr/bin/env python3
import os
import json
import time
import sqlite3
import sys
import argparse
import threading
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List
from collections import deque

# Optional ROS
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def _log(level, node, msg):
    stream = sys.stderr if level in ("WARN", "ERROR") else sys.stdout
    print(f"[{datetime.now().isoformat()}] {node}: [{level}] {msg}", file=stream)

def _log_info(n, m): _log("INFO", n, m)
def _log_warn(n, m): _log("WARN", n, m)
def _log_error(n, m): _log("ERROR", n, m)
def _log_debug(n, m): _log("DEBUG", n, m)


# ---------------------------------------------------------------------
# Creative Self Evolver Node
# ---------------------------------------------------------------------
class CreativeSelfEvolverNode:
    """
    PURPOSE:
    - Observe system-wide metrics
    - Detect stagnation, overload, imbalance
    - Propose *bounded* structural or parameter evolution
    - NEVER write code
    - NEVER integrate nodes
    - NEVER execute suggestions
    """

    def __init__(self, ros_enabled: bool = False):
        self.node_name = "creative_self_evolver_node"
        self.ros_enabled = ros_enabled or os.getenv("ROS_ENABLED", "false").lower() == "true"

        # -----------------------------
        # Evolution constraints
        # -----------------------------
        self.min_cycles_between_proposals = 10
        self.max_open_proposals = 5
        self.confidence_threshold = 0.65

        # -----------------------------
        # Observed signals
        # -----------------------------
        self.node_snapshots: Deque[Dict[str, Any]] = deque(maxlen=50)
        self.error_reports: Deque[Dict[str, Any]] = deque(maxlen=50)
        self.performance_metrics: Deque[Dict[str, Any]] = deque(maxlen=50)

        # -----------------------------
        # Proposal tracking
        # -----------------------------
        self.proposals: Deque[Dict[str, Any]] = deque(maxlen=20)
        self.cycle_counter = 0

        # -----------------------------
        # Evolver metrics (STANDARD)
        # -----------------------------
        self.evolver_metrics = {
            "cycles": 0,
            "proposals_generated": 0,
            "proposals_rejected": 0,
            "avg_confidence": 0.0,
            "last_proposal_ts": None,
        }

        # -----------------------------
        # Persistence
        # -----------------------------
        db_root = "/tmp/sentience_db"
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "creative_self_evolver.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_proposals (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                proposal_json TEXT,
                confidence REAL,
                status TEXT
            )
        """)
        self.conn.commit()

        # -----------------------------
        # ROS / Dynamic
        # -----------------------------
        self.pub_proposal = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_proposal = rospy.Publisher(
                "/self_evolution_proposals", String, queue_size=10
            )
            rospy.Timer(rospy.Duration(5.0), self._cycle)
        else:
            self._shutdown_flag = threading.Event()
            self._thread = threading.Thread(target=self._dynamic_loop, daemon=True)
            self._thread.start()

        _log_info(self.node_name, "CreativeSelfEvolverNode online (proposal-only, sandboxed).")

    # -----------------------------------------------------------------
    # Inputs (from other nodes / evolver)
    # -----------------------------------------------------------------
    def observe_node_snapshot(self, snapshot: Dict[str, Any]):
        self.node_snapshots.append(snapshot)

    def observe_error(self, error: Dict[str, Any]):
        self.error_reports.append(error)

    def observe_performance(self, perf: Dict[str, Any]):
        self.performance_metrics.append(perf)

    # -----------------------------------------------------------------
    # Core cycle
    # -----------------------------------------------------------------
    def _cycle(self, event=None):
        self.cycle_counter += 1
        self.evolver_metrics["cycles"] += 1

        if self.cycle_counter < self.min_cycles_between_proposals:
            return

        if len(self.proposals) >= self.max_open_proposals:
            return

        proposal = self._analyze_and_propose()
        if proposal:
            self._register_proposal(proposal)

    def _dynamic_loop(self):
        while not self._shutdown_flag.is_set():
            self._cycle()
            time.sleep(5.0)

    # -----------------------------------------------------------------
    # Analysis (NO LLM, NO CODE)
    # -----------------------------------------------------------------
    def _analyze_and_propose(self) -> Optional[Dict[str, Any]]:
        if not self.node_snapshots:
            return None

        recent = list(self.node_snapshots)[-10:]

        avg_confidence = sum(
            n.get("confidence_mean", 0.5) for n in recent
        ) / max(len(recent), 1)

        error_rate = len(self.error_reports) / max(self.evolver_metrics["cycles"], 1)

        if avg_confidence > 0.85 and error_rate < 0.05:
            return None  # System healthy

        proposal = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "type": "parameter_adjustment",
            "target": "evolver",
            "recommendation": {},
            "rationale": [],
        }

        if avg_confidence < 0.5:
            proposal["recommendation"]["learning_rate"] = "decrease"
            proposal["rationale"].append("Low confidence across nodes")

        if error_rate > 0.2:
            proposal["recommendation"]["safety_thresholds"] = "increase"
            proposal["rationale"].append("Elevated error frequency")

        confidence = min(1.0, max(0.1, 1.0 - error_rate))

        proposal["confidence"] = confidence
        return proposal if confidence >= self.confidence_threshold else None

    # -----------------------------------------------------------------
    # Proposal handling
    # -----------------------------------------------------------------
    def _register_proposal(self, proposal: Dict[str, Any]):
        self.proposals.append(proposal)
        self.evolver_metrics["proposals_generated"] += 1
        self.evolver_metrics["last_proposal_ts"] = time.time()
        self._update_avg_confidence(proposal["confidence"])

        self.cursor.execute(
            "INSERT INTO evolution_proposals VALUES (?, ?, ?, ?, ?)",
            (
                proposal["id"],
                datetime.now().isoformat(),
                json.dumps(proposal),
                proposal["confidence"],
                "pending",
            )
        )
        self.conn.commit()

        if ROS_AVAILABLE and self.ros_enabled and self.pub_proposal:
            self.pub_proposal.publish(String(data=json.dumps(proposal)))

        _log_info(self.node_name, f"Evolution proposal issued ({proposal['id']})")

    def reject_proposal(self, proposal_id: str):
        self.evolver_metrics["proposals_rejected"] += 1
        self._update_proposal_status(proposal_id, "rejected")

    def accept_proposal(self, proposal_id: str):
        self._update_proposal_status(proposal_id, "accepted")

    def _update_proposal_status(self, pid: str, status: str):
        self.cursor.execute(
            "UPDATE evolution_proposals SET status=? WHERE id=?",
            (status, pid),
        )
        self.conn.commit()

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    def _update_avg_confidence(self, value):
        n = self.evolver_metrics["proposals_generated"]
        prev = self.evolver_metrics["avg_confidence"]
        self.evolver_metrics["avg_confidence"] = prev + (value - prev) / max(n, 1)

    def export_evolution_snapshot(self) -> Dict[str, Any]:
        return {
            "node": self.node_name,
            "metrics": self.evolver_metrics.copy(),
            "open_proposals": len(self.proposals),
            "timestamp": time.time(),
        }

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------
    def shutdown(self):
        if hasattr(self, "_shutdown_flag"):
            self._shutdown_flag.set()
        if self.conn:
            self.conn.close()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("CreativeSelfEvolverNode shutdown")

    def run(self):
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.spin()
        else:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        self.shutdown()


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creative Self Evolver Node")
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = CreativeSelfEvolverNode(ros_enabled=args.ros_enabled)
    node.run()
