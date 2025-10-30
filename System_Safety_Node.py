```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque

# --- Asyncio Imports (for potential future async operations) ---
import asyncio
import threading
from collections import deque

# Optional ROS Integration (for compatibility)
ROS_AVAILABLE = False
rospy = None
String = None
try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
    # Placeholder for custom messages - use String or dict fallbacks
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    SafetyStatus = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    SafetyStatus = ROSMsgFallback


# --- Import shared utility functions ---
# Assuming 'sentience/scripts/utils.py' exists and contains parse_message_data and load_config
try:
    from sentience.scripts.utils import parse_message_data, load_config
except ImportError:
    # Fallback implementations
    def parse_message_data(msg: Any, fields_map: Dict[str, tuple], node_name: str = "unknown_node") -> Dict[str, Any]:
        """
        Generic parser for messages (ROS String/JSON or plain dict). 
        """
        data: Dict[str, Any] = {}
        if hasattr(msg, 'data') and isinstance(getattr(msg, 'data', None), str):
            try:
                parsed_json = json.loads(msg.data)
                for key_in_msg, (default_val, target_key) in fields_map.items():
                    data[target_key] = parsed_json.get(key_in_msg, default_val)
            except json.JSONDecodeError:
                _log_error(node_name, f"Could not parse message data as JSON: {msg.data}")
                for key_in_msg, (default_val, target_key) in fields_map.items():
                    data[target_key] = default_val
        elif isinstance(msg, dict):
            for key_in_msg, (default_val, target_key) in fields_map.items():
                data[target_key] = msg.get(key_in_msg, default_val)
        else:
            # Fallback: treat as object with attributes
            for key_in_msg, (default_val, target_key) in fields_map.items():
                data[target_key] = getattr(msg, key_in_msg, default_val)
        return data

    def load_config(node_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Fallback config loader: returns hardcoded defaults.
        """
        _log_warn(node_name, f"Using hardcoded default configuration as '{config_path}' could not be loaded.")
        return {
            'db_root_path': '/tmp/sentience_db',
            'default_log_level': 'INFO',
            'ros_enabled': False,
            'system_safety_node': {
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate safety (e.g., gentle warnings)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            }
        }.get(node_name, {})  # Return node-specific or empty dict


def _log_info(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [INFO] {msg}", file=sys.stdout)

def _log_warn(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [WARN] {msg}", file=sys.stderr)

def _log_error(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [ERROR] {msg}", file=sys.stderr)

def _log_debug(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [DEBUG] {msg}", file=sys.stdout)


class SystemSafetyNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'system_safety_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "system_safety_log.db")
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing safety compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.safety_status = {"safe": True, "shutdown_triggered": False}
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for updates
        self.safety_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for safety logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_safety_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                safe BOOLEAN,
                shutdown_triggered BOOLEAN,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "System Safety Node online, safeguarding with compassionate and mindful risk assessment.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_safety_status = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_safety_status = rospy.Publisher('safety_status', SafetyStatus, queue_size=10)
            rospy.Subscriber('health_status', String, self.health_callback)
            rospy.Subscriber('prediction_output', String, self.prediction_callback)
            rospy.Timer(rospy.Duration(2.0), self.check_safety_periodic)  # Periodic check
        else:
            # Dynamic mode: Start polling thread for simulated data
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing safety compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on safety data
            if sensor_type == 'vision':
                self.pending_updates.append({'type': 'health', 'data': {'cpu_usage': random.uniform(50, 95)}})
            elif sensor_type == 'sound':
                self.pending_updates.append({'type': 'prediction', 'data': {'predicted_cpu_usage': random.uniform(60, 100)}})
            elif sensor_type == 'instructions':
                self.pending_updates.append({'type': 'health', 'data': {'memory_usage': random.uniform(40, 95)}})
            # Compassionate bias: If distress in sound, lower safety threshold compassionately
            if 'distress' in str(processed):
                self.ethical_compassion_bias = min(1.0, self.ethical_compassion_bias + 0.1)
            _log_debug(self.node_name, f"{sensor_type} input updated safety context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._simulate_health_data()
            self._simulate_prediction_data()
            self.check_safety()
            time.sleep(2.0)

    def _simulate_health_data(self):
        """Simulate health data in non-ROS mode."""
        health_data = {'cpu_usage': random.uniform(50, 95), 'memory_usage': random.uniform(40, 95)}
        self.pending_updates.append({'type': 'health', 'data': health_data})
        _log_debug(self.node_name, f"Simulated health data: {json.dumps(health_data)}")

    def _simulate_prediction_data(self):
        """Simulate prediction data in non-ROS mode."""
        prediction_data = {'predicted_cpu_usage': random.uniform(60, 100), 'predicted_memory_usage': random.uniform(50, 95)}
        self.pending_updates.append({'type': 'prediction', 'data': prediction_data})
        _log_debug(self.node_name, f"Simulated prediction data: {json.dumps(prediction_data)}")

    # --- Core Safety Logic ---
    def health_callback(self, msg: Any):
        """Handle incoming health status data."""
        fields_map = {'data': ('', 'health_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        health_data = json.loads(data.get('health_data', '{}'))
        self.check_safety(health_data)

    def prediction_callback(self, msg: Any):
        """Handle incoming prediction data for safety check."""
        fields_map = {'data': ('', 'prediction_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        prediction_data = json.loads(data.get('prediction_data', '{}'))
        self.check_safety(prediction_data)

    def check_safety(self, data: Dict[str, Any]) -> None:
        """Check system safety based on health and prediction data with compassionate bias."""
        try:
            # Base checks
            if data.get("cpu_usage", 0) > 95 or data.get("memory_usage", 0) > 95:
                self.safety_status["safe"] = False
                self.safety_status["shutdown_triggered"] = True
                _log_warn(self.node_name, "Safety breach detected, triggering shutdown")
            elif data.get("predicted_cpu_usage", 0) > 90 or data.get("predicted_memory_usage", 0) > 90:
                self.safety_status["safe"] = False
                _log_warn(self.node_name, "Predicted safety breach")

            # Compassionate bias: If high load, consider gentle cooldown instead of hard shutdown
            if self.safety_status["shutdown_triggered"] and self.ethical_compassion_bias > 0.1:
                self.safety_status["shutdown_triggered"] = False  # Opt for soft intervention
                self.safety_status["safe"] = False  # Still unsafe, but compassionate
                _log_warn(self.node_name, f"High load detected - Compassionate cooldown instead of shutdown. Bias: {self.ethical_compassion_bias}.")

            self.publish_safety_status()
            # Log with sensory snapshot
            sensory_snapshot = json.dumps(self.sensory_data)
            self._log_safety_status(sensory_snapshot)
        except Exception as e:
            _log_error(self.node_name, f"Error checking safety: {e}")

    def _log_safety_status(self, sensory_snapshot: str):
        """Log safety status to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO system_safety_log (id, timestamp, safe, shutdown_triggered, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), self.safety_status['safe'],
                self.safety_status['shutdown_triggered'], sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log safety status: {e}")

    def publish_safety_status(self) -> None:
        """Publish the current safety status (ROS or log)."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_safety_status:
                if hasattr(SafetyStatus, 'data'):
                    self.pub_safety_status.publish(String(data=json.dumps(self.safety_status)))
                else:
                    status_msg = SafetyStatus(data=json.dumps(self.safety_status))
                    self.pub_safety_status.publish(status_msg)
                if self.safety_status["shutdown_triggered"]:
                    rospy.signal_shutdown("Safety shutdown triggered")
            else:
                # Dynamic mode: Log
                _log_info(self.node_name, f"Safety status: {json.dumps(self.safety_status)}")
                if self.safety_status["shutdown_triggered"]:
                    sys.exit("Safety shutdown triggered")
        except Exception as e:
            _log_error(self.node_name, f"Error publishing safety status: {e}")

    def check_safety_periodic(self):
        """Periodic safety check."""
        self.publish_safety_status()

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down SystemSafetyNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Run the node with simulated or actual ROS."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                while True:
                    self._simulate_health_data()
                    self._simulate_prediction_data()
                    self.check_safety()
                    time.sleep(2.0)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience System Safety Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = SystemSafetyNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate data
            node.health_callback({'data': json.dumps({'cpu_usage': 96})})
            time.sleep(1)
            print("Safety simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
