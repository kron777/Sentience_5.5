#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import argparse
import uuid
import random
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque
import threading

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

    ErrorLog = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    ErrorLog = ROSMsgFallback


# --- Utils ---
try:
    from sentience.scripts.utils import parse_message_data, load_config
except ImportError:
    def parse_message_data(msg, fields_map, node_name="unknown"):
        data = {}
        if hasattr(msg, 'data'):
            try:
                payload = json.loads(msg.data)
            except Exception:
                payload = {}
        elif isinstance(msg, dict):
            payload = msg
        else:
            payload = {}

        for k, (default, target) in fields_map.items():
            data[target] = payload.get(k, default)
        return data

    def load_config(node_name, config_path=None):
        return {
            "db_root_path": "/tmp/sentience_db",
            "error_logger_node": {
                "flush_interval": 5.0,
                "severity_learning_rate": 0.1
            }
        }.get(node_name, {})


def _log(node, level, msg):
    print(f"[{datetime.now().isoformat()}] {node} [{level}] {msg}", file=sys.stdout)


# ============================================================
# ERROR LOGGER NODE (OBSERVABILITY + LEARNING SUPPORT)
# ============================================================
class ErrorLoggerNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = "error_logger_node"
        self.ros_enabled = ros_enabled or os.getenv("ROS_ENABLED", "false").lower() == "true"

        config = load_config("error_logger_node", config_file_path)
        global_config = load_config("global", config_file_path)

        self.db_path = os.path.join(
            global_config.get("db_root_path", "/tmp/sentience_db"),
            "error_log.db"
        )

        self.flush_interval = config.get("flush_interval", 5.0)
        self.severity_learning_rate = config.get("severity_learning_rate", 0.1)

        # --- Internal state ---
        self.pending_logs: Deque[Dict[str, Any]] = deque(maxlen=100)
        self.error_history: Deque[Dict[str, Any]] = deque(maxlen=500)

        # Rolling statistics for evolver.py
        self.error_stats = {
            "total": 0,
            "by_type": {},
            "avg_severity": 0.0
        }

        # Sensory snapshot (normalized, optional)
        self.sensory_data: Dict[str, Any] = {}

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                source_node TEXT,
                error_type TEXT,
                description TEXT,
                severity REAL,
                sensory_snapshot_json TEXT
            )
        """)
        self.conn.commit()

        _log(self.node_name, "INFO", "Error Logger Node online (observability + learning).")

        # ROS wiring
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            rospy.Subscriber("/error_reports", String, self.error_report_callback)
            rospy.Timer(rospy.Duration(self.flush_interval), self.flush_pending_logs)
        else:
            self._shutdown_flag = threading.Event()
            threading.Thread(target=self._dynamic_loop, daemon=True).start()

    # ---------------------------------------------------------
    # CORE LOGGING
    # ---------------------------------------------------------
    def log_error(
        self,
        source_node: str,
        error_type: str,
        description: str,
        severity: float,
        sensory_snapshot: Optional[Dict[str, Any]] = None
    ):
        severity = float(max(0.0, min(1.0, severity)))

        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": str(time.time()),
            "source_node": source_node,
            "error_type": error_type,
            "description": description,
            "severity": severity,
            "sensory_snapshot": sensory_snapshot or {}
        }

        self.pending_logs.append(entry)
        self.error_history.append(entry)
        self._update_stats(entry)

        _log(self.node_name, "ERROR", f"{source_node} | {error_type} | sev={severity}")

    # ---------------------------------------------------------
    # LEARNING SURFACE (USED BY evolver.py)
    # ---------------------------------------------------------
    def _update_stats(self, entry: Dict[str, Any]):
        self.error_stats["total"] += 1

        et = entry["error_type"]
        self.error_stats["by_type"].setdefault(et, 0)
        self.error_stats["by_type"][et] += 1

        # Exponential moving average
        prev = self.error_stats["avg_severity"]
        self.error_stats["avg_severity"] = (
            prev + self.severity_learning_rate * (entry["severity"] - prev)
        )

    def export_metrics(self) -> Dict[str, Any]:
        """Used by evolver.py"""
        return {
            "total_errors": self.error_stats["total"],
            "errors_by_type": dict(self.error_stats["by_type"]),
            "avg_severity": self.error_stats["avg_severity"]
        }

    # ---------------------------------------------------------
    # ROS / EXTERNAL INPUT
    # ---------------------------------------------------------
    def error_report_callback(self, msg: Any):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return

        self.log_error(
            source_node=payload.get("source_node", "unknown"),
            error_type=payload.get("error_type", "unknown"),
            description=payload.get("description", ""),
            severity=payload.get("severity", 0.5),
            sensory_snapshot=payload.get("sensory_snapshot", {})
        )

    # ---------------------------------------------------------
    # PERSISTENCE
    # ---------------------------------------------------------
    def flush_pending_logs(self, event=None):
        if not self.pending_logs:
            return

        batch = list(self.pending_logs)
        self.pending_logs.clear()

        try:
            self.cursor.executemany("""
                INSERT INTO error_log
                (id, timestamp, source_node, error_type, description, severity, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    e["id"],
                    e["timestamp"],
                    e["source_node"],
                    e["error_type"],
                    e["description"],
                    e["severity"],
                    json.dumps(e["sensory_snapshot"])
                )
                for e in batch
            ])
            self.conn.commit()
            _log(self.node_name, "INFO", f"Flushed {len(batch)} errors.")
        except Exception as e:
            _log(self.node_name, "WARN", f"DB flush failed: {e}")
            for item in batch:
                self.pending_logs.append(item)

    # ---------------------------------------------------------
    def _dynamic_loop(self):
        while not self._shutdown_flag.is_set():
            self.flush_pending_logs()
            time.sleep(self.flush_interval)

    def shutdown(self):
        _log(self.node_name, "INFO", "Shutting down ErrorLoggerNode.")
        if hasattr(self, "_shutdown_flag"):
            self._shutdown_flag.set()
        self.flush_pending_logs()
        self.conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentience Error Logger Node")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ros-enabled", action="store_true")
    args = parser.parse_args()

    node = ErrorLoggerNode(args.config, args.ros_enabled)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.shutdown()
