#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import uuid
import sys
import argparse
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List
from collections import deque

# Optional ROS Integration
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
# Logging helpers
# ---------------------------------------------------------------------
def _log(level: str, node: str, msg: str):
    stream = sys.stderr if level in ("WARN", "ERROR") else sys.stdout
    print(f"[{datetime.now().isoformat()}] {node}: [{level}] {msg}", file=stream)

def _log_info(node, msg): _log("INFO", node, msg)
def _log_warn(node, msg): _log("WARN", node, msg)
def _log_error(node, msg): _log("ERROR", node, msg)
def _log_debug(node, msg): _log("DEBUG", node, msg)


# ---------------------------------------------------------------------
# Creative Expression Node
# ---------------------------------------------------------------------
class CreativeExpressionNode:
    """
    Role:
    - Translates internal state into external expression
    - Does NOT reason, plan, or decide
    - Learning is about expression quality & stability, not beliefs
    """

    def __init__(self, ros_enabled: bool = False):
        self.node_name = "creative_expression_node"
        self.ros_enabled = ros_enabled or os.getenv("ROS_ENABLED", "false").lower() == "true"

        # -----------------------------
        # Parameters (evolver-tunable)
        # -----------------------------
        self.expression_interval = 2.0
        self.salience_threshold = 0.65
        self.learning_rate = 0.05
        self.max_creativity = 0.9

        # -----------------------------
        # Internal State
        # -----------------------------
        self.recent_emotions: Deque[Dict[str, Any]] = deque(maxlen=8)
        self.recent_attention: Deque[Dict[str, Any]] = deque(maxlen=8)
        self.recent_directives: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_social: Deque[Dict[str, Any]] = deque(maxlen=5)

        self.last_expression: Dict[str, Any] = {
            "timestamp": time.time(),
            "type": "idle",
            "content": {"text": "…"},
            "themes": [],
            "creativity_score": 0.1,
        }

        self.cumulative_salience = 0.0

        # -----------------------------
        # Learning metrics
        # -----------------------------
        self.learning_metrics = {
            "expressions_generated": 0,
            "fallback_used": 0,
            "avg_creativity": 0.1,
            "last_adjustment": None,
        }

        # -----------------------------
        # Persistence
        # -----------------------------
        db_root = "/tmp/sentience_db"
        os.makedirs(db_root, exist_ok=True)
        self.db_path = os.path.join(db_root, "creative_expression_log.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS creative_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                expression_json TEXT,
                creativity REAL
            )
        """)
        self.conn.commit()

        # -----------------------------
        # ROS / Dynamic loop
        # -----------------------------
        self.pub_expression = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_expression = rospy.Publisher(
                "creative_expression", String, queue_size=10
            )
            rospy.Timer(
                rospy.Duration(self.expression_interval),
                self._expression_tick
            )
        else:
            self._shutdown_flag = threading.Event()
            self._thread = threading.Thread(
                target=self._dynamic_loop, daemon=True
            )
            self._thread.start()

        _log_info(self.node_name, "CreativeExpressionNode online (stable, bounded, evolver-ready).")

    # -----------------------------------------------------------------
    # Input hooks
    # -----------------------------------------------------------------
    def update_emotion(self, data: Dict[str, Any]):
        self.recent_emotions.append(data)
        self._bump_salience(data.get("intensity", 0.2))

    def update_attention(self, data: Dict[str, Any]):
        self.recent_attention.append(data)
        self._bump_salience(data.get("priority", 0.1))

    def update_directive(self, data: Dict[str, Any]):
        if data.get("target_node") == self.node_name:
            self.recent_directives.append(data)
            self._bump_salience(data.get("urgency", 0.5))

    def update_social(self, data: Dict[str, Any]):
        self.recent_social.append(data)
        self._bump_salience(data.get("confidence", 0.2))

    # -----------------------------------------------------------------
    # Core loop
    # -----------------------------------------------------------------
    def _expression_tick(self, event=None):
        if self.cumulative_salience < self.salience_threshold:
            self._generate_fallback()
        else:
            self._generate_structured_expression()
        self.cumulative_salience = 0.0

    def _dynamic_loop(self):
        while not self._shutdown_flag.is_set():
            self._expression_tick()
            time.sleep(self.expression_interval)

    # -----------------------------------------------------------------
    # Expression generation (NO belief creation)
    # -----------------------------------------------------------------
    def _generate_structured_expression(self):
        mood = self._dominant("mood", self.recent_emotions)
        focus = self._dominant("focus", self.recent_attention)

        text = f"A moment shaped by {mood or 'neutrality'} and attention toward {focus or 'nothing in particular'}."
        creativity = min(self.max_creativity, 0.4 + self.cumulative_salience * 0.5)

        self._publish_expression(
            expression_type="text_reflection",
            content={"text": text},
            themes=[mood, focus],
            creativity=creativity,
        )

    def _generate_fallback(self):
        self.learning_metrics["fallback_used"] += 1
        self._publish_expression(
            expression_type="idle",
            content={"text": "I am present."},
            themes=[],
            creativity=0.1,
        )

    # -----------------------------------------------------------------
    # Publish & learn
    # -----------------------------------------------------------------
    def _publish_expression(self, expression_type, content, themes, creativity):
        expr = {
            "timestamp": time.time(),
            "type": expression_type,
            "content": content,
            "themes": [t for t in themes if t],
            "creativity_score": creativity,
        }

        self.last_expression = expr
        self.learning_metrics["expressions_generated"] += 1
        self._update_avg_creativity(creativity)

        payload = json.dumps(expr)
        if ROS_AVAILABLE and self.ros_enabled and self.pub_expression:
            self.pub_expression.publish(String(data=payload))

        self.cursor.execute(
            "INSERT INTO creative_log VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), datetime.now().isoformat(), payload, creativity),
        )
        self.conn.commit()

        _log_info(self.node_name, f"Creative expression: {expression_type}")

    # -----------------------------------------------------------------
    # Learning helpers
    # -----------------------------------------------------------------
    def _update_avg_creativity(self, value):
        n = self.learning_metrics["expressions_generated"]
        prev = self.learning_metrics["avg_creativity"]
        self.learning_metrics["avg_creativity"] = prev + (value - prev) / max(n, 1)

    def _bump_salience(self, score):
        self.cumulative_salience = min(1.0, self.cumulative_salience + float(score))

    def _dominant(self, key, history):
        for item in reversed(history):
            if key in item:
                return item[key]
        return None

    # -----------------------------------------------------------------
    # Evolver Interface (STANDARD)
    # -----------------------------------------------------------------
    def get_internal_state(self) -> Dict[str, Any]:
        return {
            "last_expression": self.last_expression,
            "salience": self.cumulative_salience,
        }

    def get_learning_metrics(self) -> Dict[str, Any]:
        return self.learning_metrics.copy()

    def apply_adjustment(self, delta: Dict[str, Any]):
        if "learning_rate" in delta:
            self.learning_rate = max(0.001, min(0.2, float(delta["learning_rate"])))
        if "salience_threshold" in delta:
            self.salience_threshold = max(0.2, min(0.9, float(delta["salience_threshold"])))
        self.learning_metrics["last_adjustment"] = time.time()

    def export_evolution_snapshot(self) -> Dict[str, Any]:
        return {
            "node": self.node_name,
            "metrics": self.get_learning_metrics(),
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
            rospy.signal_shutdown("CreativeExpressionNode shutdown")

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
    parser = argparse.ArgumentParser(description="Sentience Creative Expression Node")
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = CreativeExpressionNode(ros_enabled=args.ros_enabled)
    node.run()
