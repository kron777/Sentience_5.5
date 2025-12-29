#!/usr/bin/env python3
"""
TheoryOfMindNode – Sentience 5.5 (updated)

Fixes / updates:
- Fixed sqlite cursor initialization bug
- Added missing imports (uuid)
- Added sensory_data snapshot handling
- Deterministic update pipeline (no uncontrolled growth)
- Clear separation: model → predict → publish
- Safe dynamic / ROS dual-mode
- Lightweight, non-LLM, prediction-ready
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import sqlite3
import argparse
import threading
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

    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    TheoryOfMindUpdate = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    TheoryOfMindUpdate = ROSMsgFallback


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
def log(level: str, node: str, msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# --------------------------------------------------------------------------- #
# Node                                                                          #
# --------------------------------------------------------------------------- #
class TheoryOfMindNode:
    """
    Maintains belief / desire / intention models of external agents
    and produces lightweight behavioral predictions.
    """

    def __init__(self, db_root: str = "/tmp/sentience_db", ros_enabled: bool = False):
        self.node_name = "theory_of_mind_node"
        self.ros_enabled = ros_enabled and ROS_AVAILABLE

        # Compassion bias (bounded)
        self.ethical_compassion_bias: float = 0.2

        # Sensory snapshot (shared contract)
        self.sensory_data: Dict[str, Any] = {}

        # Agent models
        self.agents: Dict[str, Dict[str, List[str]]] = {}
        self.history: Deque[Dict[str, Any]] = deque(maxlen=100)

        # DB
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "theory_of_mind.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

        # ROS
        self.pub_update = None
        if self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_update = rospy.Publisher(
                "/theory_of_mind/update", String, queue_size=10
            )
            rospy.Timer(rospy.Duration(4.0), self._periodic_publish)
        else:
            self._shutdown = threading.Event()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        log("INFO", self.node_name, "TheoryOfMindNode online")

    # ------------------------------------------------------------------ #
    # DB                                                                  #
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tom_log (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                agent_id TEXT,
                beliefs_json TEXT,
                desires_json TEXT,
                intentions_json TEXT,
                sensory_snapshot_json TEXT
            )
            """
        )
        self.conn.commit()

    def _persist(self, agent_id: str):
        model = self.agents.get(agent_id)
        if not model:
            return
        self.cursor.execute(
            """
            INSERT INTO tom_log
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                time.time(),
                agent_id,
                json.dumps(model["beliefs"]),
                json.dumps(model["desires"]),
                json.dumps(model["intentions"]),
                json.dumps(self.sensory_data),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Core Logic                                                          #
    # ------------------------------------------------------------------ #
    def upsert_agent(
        self,
        agent_id: str,
        beliefs: Optional[List[str]] = None,
        desires: Optional[List[str]] = None,
        intentions: Optional[List[str]] = None,
    ):
        model = self.agents.setdefault(
            agent_id, {"beliefs": [], "desires": [], "intentions": []}
        )

        if beliefs:
            model["beliefs"] = list(set(model["beliefs"] + beliefs))
        if desires:
            model["desires"] = list(set(model["desires"] + desires))
        if intentions:
            model["intentions"] = list(set(model["intentions"] + intentions))

        # Compassion bias injection
        if self.ethical_compassion_bias > 0.15:
            if "empathy" not in model["beliefs"]:
                model["beliefs"].append("empathy")

        self._persist(agent_id)
        log("INFO", self.node_name, f"Updated agent '{agent_id}'")

    def predict(self, agent_id: str) -> Dict[str, Any]:
        model = self.agents.get(agent_id)
        if not model:
            return {"agent_id": agent_id, "prediction": "unknown"}

        if model["intentions"]:
            action = model["intentions"][0]
        elif model["desires"]:
            action = f"seek {model['desires'][0]}"
        else:
            action = "observe"

        prediction = {
            "agent_id": agent_id,
            "prediction": action,
            "confidence": min(1.0, 0.6 + self.ethical_compassion_bias),
            "timestamp": time.time(),
        }

        self.history.append(prediction)
        return prediction

    # ------------------------------------------------------------------ #
    # Publishing                                                          #
    # ------------------------------------------------------------------ #
    def _periodic_publish(self, *_):
        if not self.agents:
            return
        summary = {aid: self.predict(aid) for aid in self.agents}
        self._publish(summary)

    def _publish(self, payload: Dict[str, Any]):
        msg = json.dumps(payload)
        if self.ros_enabled and self.pub_update:
            self.pub_update.publish(String(data=msg))
        else:
            log("INFO", self.node_name, f"ToM update: {msg}")

    # ------------------------------------------------------------------ #
    # Dynamic loop                                                        #
    # ------------------------------------------------------------------ #
    def _loop(self):
        while not self._shutdown.is_set():
            for aid in list(self.agents.keys()):
                self.predict(aid)
            self._periodic_publish()
            time.sleep(4.0)

    # ------------------------------------------------------------------ #
    # Shutdown                                                            #
    # ------------------------------------------------------------------ #
    def shutdown(self):
        log("INFO", self.node_name, "Shutting down")
        if hasattr(self, "_shutdown"):
            self._shutdown.set()
        self.conn.close()
        if self.ros_enabled:
            rospy.signal_shutdown("shutdown")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = TheoryOfMindNode(ros_enabled=args.ros_enabled)

    # Demo (non-ROS)
    if not args.ros_enabled:
        node.upsert_agent(
            "Alice",
            beliefs=["weather is uncertain"],
            desires=["stay dry"],
            intentions=["carry umbrella"],
        )
        time.sleep(6)

    try:
        if args.ros_enabled:
            rospy.spin()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
