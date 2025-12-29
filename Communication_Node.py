#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import argparse
import asyncio
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from collections import deque
from uuid import uuid4

# Optional ROS Integration
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

    CommunicationMessage = ROSMsgFallback
    FeedbackResponse = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    CommunicationMessage = ROSMsgFallback
    FeedbackResponse = ROSMsgFallback


# --- Utilities ---
try:
    from sentience.scripts.utils import parse_message_data, load_config
except ImportError:
    def parse_message_data(msg, fields_map, node_name="unknown"):
        data = {}
        if hasattr(msg, 'data'):
            try:
                parsed = json.loads(msg.data)
                for k, (d, t) in fields_map.items():
                    data[t] = parsed.get(k, d)
            except Exception:
                for k, (d, t) in fields_map.items():
                    data[t] = d
        elif isinstance(msg, dict):
            for k, (d, t) in fields_map.items():
                data[t] = msg.get(k, d)
        return data

    def load_config(node_name, config_path=None):
        return {
            'db_root_path': '/tmp/sentience_db',
            'communication_node': {
                'valid_channels': ['console', 'file', 'network'],
                'default_channel': 'console',
                'flush_interval': 3.0,
                'ethical_compassion_bias': 0.2
            }
        }.get(node_name, {})


def _log(node, lvl, msg):
    print(f"[{datetime.now().isoformat()}] {node} [{lvl}] {msg}", file=sys.stdout)


class CommunicationNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = "communication_node"
        self.ros_enabled = ros_enabled or os.getenv("ROS_ENABLED", "false").lower() == "true"

        config = load_config(self.node_name, config_file_path)
        self.valid_channels = config.get("valid_channels", ["console"])
        self.output_channel = config.get("default_channel", "console")
        self.ethical_compassion_bias = config.get("ethical_compassion_bias", 0.2)
        self.flush_interval = config.get("flush_interval", 3.0)

        # --- Evolver metrics ---
        self.evolver_metrics = {
            "cycles": 0,
            "messages_sent": 0,
            "feedback_received": 0,
            "fallbacks": 0,
            "last_update_ts": time.time()
        }

        # --- Queues ---
        self.pending_messages = deque(maxlen=20)
        self.feedback_queue = deque(maxlen=10)
        self.history = deque(maxlen=50)

        # --- DB ---
        db_root = load_config("global", config_file_path).get("db_root_path", "/tmp/sentience_db")
        self.db_path = os.path.join(db_root, "communication_log.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS communication_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                channel TEXT,
                content TEXT,
                feedback BOOLEAN
            )
        """)
        self.conn.commit()

        # --- Async loop ---
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._async_thread.start()

        # --- ROS ---
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_msg = rospy.Publisher("/communication_out", String, queue_size=10)
            self.sub_feedback = rospy.Subscriber("/communication_feedback", String, self._ros_feedback)

        self._shutdown_flag = threading.Event()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

        _log(self.node_name, "INFO", "Communication node online.")

    # ---------------- Core ----------------

    def send(self, message: Dict[str, Any]):
        self.pending_messages.append(message)

    async def _send_async(self, message: Dict[str, Any]):
        tone = message.get("tone", "neutral")
        content = message.get("content", "")

        if self.ethical_compassion_bias > 0.3 and tone == "neutral":
            tone = "compassionate"
            content = f"I understand. {content}"

        payload = json.dumps({"tone": tone, "content": content})

        try:
            if self.output_channel == "console":
                print(payload)
            elif self.output_channel == "file":
                with open("communication_output.log", "a") as f:
                    f.write(payload + "\n")
            elif self.output_channel == "network":
                await asyncio.sleep(0.05)

            self._log_message(payload)
            self.evolver_metrics["messages_sent"] += 1
        except Exception:
            self.evolver_metrics["fallbacks"] += 1

    def receive_feedback(self, feedback: Dict[str, Any]):
        self.feedback_queue.append(feedback)
        self.evolver_metrics["feedback_received"] += 1

    # ---------------- Internals ----------------

    def _run_loop(self):
        while not self._shutdown_flag.is_set():
            self.evolver_metrics["cycles"] += 1
            self.evolver_metrics["last_update_ts"] = time.time()

            if self.pending_messages:
                msg = self.pending_messages.popleft()
                asyncio.run_coroutine_threadsafe(
                    self._send_async(msg), self._async_loop
                )

            time.sleep(self.flush_interval)

    def _log_message(self, content: str):
        try:
            self.cursor.execute(
                "INSERT INTO communication_log VALUES (?, ?, ?, ?, ?)",
                (str(uuid4()), str(time.time()), self.output_channel, content, False)
            )
            self.conn.commit()
        except Exception:
            pass

    def _ros_feedback(self, msg):
        try:
            data = json.loads(msg.data)
            self.receive_feedback(data)
        except Exception:
            pass

    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    # ---------------- Evolver API ----------------

    def get_internal_state(self):
        return {
            "channel": self.output_channel,
            "queue_depth": len(self.pending_messages)
        }

    def get_learning_metrics(self):
        return self.evolver_metrics.copy()

    def apply_adjustment(self, delta: Dict[str, Any]):
        if "ethical_compassion_bias" in delta:
            self.ethical_compassion_bias = max(
                0.0, min(1.0, self.ethical_compassion_bias + delta["ethical_compassion_bias"])
            )

    def export_evolution_snapshot(self):
        return {
            "node": self.node_name,
            "metrics": self.get_learning_metrics(),
            "state": self.get_internal_state()
        }

    # ---------------- Shutdown ----------------

    def shutdown(self):
        self._shutdown_flag.set()
        self.conn.close()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("shutdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = CommunicationNode(ros_enabled=args.ros_enabled)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.shutdown()
