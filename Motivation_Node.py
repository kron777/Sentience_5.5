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
    MotivationStatus = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    MotivationStatus = ROSMsgFallback


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
            'motivation_node': {
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate motivation (e.g., self-care goals)
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


class MotivationNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'motivation_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "motivation_log.db")
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing motivation compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.motivation_level = 0.5  # Default motivation level (0 to 1)
        self.goal = "default_goal"
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for updates
        self.motivation_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for motivation logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS motivation_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                motivation_level REAL,
                goal TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Motivation Node initialized, motivating with compassionate self-regulation.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_motivation_status = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_motivation_status = rospy.Publisher('motivation_status', MotivationStatus, queue_size=10)
            rospy.Subscriber('integration_output', String, self.integration_callback)
            rospy.Timer(rospy.Duration(1.0), self._periodic_update)  # 1 Hz update
        else:
            # Dynamic mode: Start polling thread for simulated inputs
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing motivation compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on motivation
            if sensor_type == 'vision':
                self.pending_updates.append({'type': 'integration', 'data': {'confidence': random.uniform(0.4, 0.9)}})
            elif sensor_type == 'sound':
                self.pending_updates.append({'type': 'integration', 'data': {'status': 'alert' if random.random() < 0.3 else 'normal'}})
            elif sensor_type == 'instructions':
                self.pending_updates.append({'type': 'integration', 'data': {'confidence': random.uniform(0.6, 1.0)}})
            # Compassionate bias: If distress in sound, adjust motivation toward self-care
            if 'distress' in str(processed):
                self.motivation_level = min(1.0, self.motivation_level + self.ethical_compassion_bias * 0.1)
                self.goal = 'self_care' if self.motivation_level < 0.3 else self.goal
            _log_debug(self.node_name, f"{sensor_type} input updated motivation context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            # Simulate periodic integration data
            self._simulate_integration_data()
            self._periodic_update()
            time.sleep(1.0)

    def _simulate_integration_data(self):
        """Simulate integration data in non-ROS mode."""
        integrated_data = {
            "components": {
                "decision": {"confidence": random.uniform(0.3, 0.9)},
                "system_status": random.choice(["alert", "normal", "optimal"])
            }
        }
        self.integration_callback({'data': json.dumps(integrated_data)})
        _log_debug(self.node_name, "Simulated integration data.")

    # --- Core Motivation Logic ---
    def integration_callback(self, msg: Any):
        """Handle incoming integrated data."""
        fields_map = {'data': ('', 'integrated_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        integrated_data = json.loads(data.get('integrated_data', '{}'))
        self.update_motivation(integrated_data)

    def update_motivation(self, integrated_data: Dict[str, Any]) -> None:
        """Update motivation level based on integrated data with compassionate bias."""
        try:
            confidence = integrated_data.get("components", {}).get("decision", {}).get("confidence", 0)
            status = integrated_data.get("components", {}).get("system_status", "unknown")
            
            # Compassionate bias: If low confidence, bias toward reflective/recovery goals
            if status == "alert" or confidence < 0.3:
                self.motivation_level = max(0.0, self.motivation_level - 0.1)
                self.goal = "recovery"
            elif confidence > 0.7:
                self.motivation_level = min(1.0, self.motivation_level + 0.1)
                self.goal = "optimization"
            else:
                self.motivation_level = 0.5
                self.goal = "maintenance"

            # Compassionate adjustment: If status indicates stress, prioritize self-care
            if status == "alert" and self.ethical_compassion_bias > 0.1:
                self.goal = "self_care" if "error" in status.lower() else self.goal
                self.motivation_level = min(1.0, self.motivation_level + self.ethical_compassion_bias * 0.05)

            _log_info(self.node_name, f"Motivation updated - Level: {self.motivation_level:.2f}, Goal: {self.goal}")

            # Log to DB with sensory snapshot
            sensory_snapshot = json.dumps(self.sensory_data)
            self._log_motivation_update(sensory_snapshot)
        except Exception as e:
            _log_error(self.node_name, f"Error updating motivation: {e}")

    def _log_motivation_update(self, sensory_snapshot: str):
        """Log motivation update to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO motivation_log (id, timestamp, motivation_level, goal, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), self.motivation_level, self.goal, sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log motivation update: {e}")

    def _periodic_update(self):
        """Periodic motivation status publishing."""
        if time.time() - self.last_check > 1.0:  # 1 Hz
            self.last_check = time.time()
            self.publish_motivation_status()

    def publish_motivation_status(self):
        """Publish motivation status (ROS or log)."""
        message = {
            'motivation_level': self.motivation_level,
            'goal': self.goal,
            'timestamp': time.time()
        }
        if ROS_AVAILABLE and self.ros_enabled and self.pub_motivation_status:
            if hasattr(MotivationStatus, 'data'):
                self.pub_motivation_status.publish(String(data=json.dumps(message)))
            else:
                status_msg = MotivationStatus(data=json.dumps(message))
                self.pub_motivation_status.publish(status_msg)
        else:
            # Dynamic mode: Log
            _log_info(self.node_name, f"Published motivation status: {json.dumps(message)}")
        _log_debug(self.node_name, f"Motivation status: Level {self.motivation_level:.2f}, Goal '{self.goal}'")

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down MotivationNode.")
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
                    self._periodic_update()
                    time.sleep(1.0)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Motivation Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = MotivationNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate integration data
            integrated_data = {"components": {"decision": {"confidence": 0.8}, "system_status": "normal"}}
            node.integration_callback({'data': json.dumps(integrated_data)})
            time.sleep(2)
            print("Motivation simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
