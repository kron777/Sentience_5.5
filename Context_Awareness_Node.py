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
    ContextStatus = ROSMsgFallback
    ControlOutput = ROSMsgFallback
    HealthStatus = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    ContextStatus = ROSMsgFallback
    ControlOutput = ROSMsgFallback
    HealthStatus = ROSMsgFallback


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
            'context_awareness_node': {
                'default_environment': 'neutral',
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate context assessment
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


class ContextAwarenessNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'context_awareness_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "context_awareness_log.db")
        self.default_environment = self.params.get('default_environment', 'neutral')
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing context assessment)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.context: Dict[str, Any] = {
            "environment": self.default_environment,
            "time_of_day": "unknown",
            "priority": "normal",
            "compassionate_adjustment": 0.0  # Bias toward compassionate context
        }
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for updates
        self.context_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for context logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_awareness_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                environment TEXT,
                time_of_day TEXT,
                priority TEXT,
                compassionate_adjustment REAL,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Context Awareness Node online, attuned to mindful and compassionate environmental perception.")

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_context_status = None
        self.sub_control = None
        self.sub_health = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_context_status = rospy.Publisher('context_status', ContextStatus, queue_size=10)
            self.sub_control = rospy.Subscriber('control_output', ControlOutput, self.control_callback)
            self.sub_health = rospy.Subscriber('health_status', HealthStatus, self.health_callback)

            # Sensory subscribers
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(0.5), self.process_pending_updates)  # Periodic processing
        else:
            # Dynamic mode: Polling loop
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        # Initial publish
        self.publish_context()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing context (e.g., compassionate adjustment from emotional cues)."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on context
            if sensor_type == 'vision':
                self.pending_updates.append({'source': 'vision', 'data': {'environment': 'crowded' if 'people' in str(processed) else 'calm'}})
            elif sensor_type == 'sound':
                self.pending_updates.append({'source': 'sound', 'data': {'priority': 'high' if 'alarm' in str(processed) else 'normal'}})
            elif sensor_type == 'instructions':
                self.pending_updates.append({'source': 'instructions', 'data': {'priority': 'high' if 'urgent' in str(processed) else 'normal'}})
            # Compassionate bias: Adjust for emotional context
            if 'distress' in str(processed):
                self.context['compassionate_adjustment'] = min(1.0, self.context['compassionate_adjustment'] + self.ethical_compassion_bias)
            _log_debug(self.node_name, f"{sensor_type} input updated context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.process_pending_updates()
            time.sleep(0.5)

    def _get_current_time(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    # --- Core Context Awareness Logic ---
    def update_context_from_control(self, control_data: Dict[str, Any]) -> None:
        """Update context based on control actions with compassionate bias."""
        try:
            action = control_data.get("action", "idle")
            if action == "execute_task":
                self.context["environment"] = "active"
                self.context["priority"] = "high"
            elif action == "wait":
                self.context["environment"] = "idle"
                self.context["priority"] = "low"
            self.context["time_of_day"] = self._get_current_time()
            # Compassionate bias: If high priority, adjust for compassionate context
            if self.context["priority"] == "high" and self.ethical_compassion_bias > 0.2:
                self.context['compassionate_adjustment'] = min(1.0, self.context.get('compassionate_adjustment', 0.0) + self.ethical_compassion_bias)
            _log_info(self.node_name, f"Updated context from control: {json.dumps(self.context)}")
            self.publish_context()
        except Exception as e:
            _log_error(self.node_name, f"Error updating context from control: {e}")

    def update_context_from_health(self, health_data: Dict[str, Any]) -> None:
        """Update context based on health status with compassionate bias."""
        try:
            if health_data.get("cpu_usage", 0) > 80 or health_data.get("memory_usage", 0) > 80:
                self.context["environment"] = "stressed"
                self.context["priority"] = "high"
                # Compassionate bias: In stress, prioritize self-care compassion
                self.context['compassionate_adjustment'] = min(1.0, self.context.get('compassionate_adjustment', 0.0) + self.ethical_compassion_bias * 0.5)
            self.publish_context()
        except Exception as e:
            _log_error(self.node_name, f"Error updating context from health: {e}")

    def publish_context(self) -> None:
        """Publish the current context status (ROS or log)."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_context_status:
                if hasattr(ContextStatus, 'data'):
                    self.pub_context_status.publish(String(data=json.dumps(self.context)))
                else:
                    context_msg = ContextStatus(data=json.dumps(self.context))
                    self.pub_context_status.publish(context_msg)
                _log_info(self.node_name, f"Published context status: {json.dumps(self.context)}")
            else:
                # Dynamic: Log
                _log_info(self.node_name, f"Dynamic context status: {json.dumps(self.context)}")
            # Log to DB with sensory snapshot
            sensory_snapshot = json.dumps(self.sensory_data)
            self._log_context_update(sensory_snapshot)
        except Exception as e:
            _log_error(self.node_name, f"Error publishing context: {e}")

    def _log_context_update(self, sensory_snapshot: str):
        """Log context update to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO context_awareness_log (id, timestamp, environment, time_of_day, priority, compassionate_adjustment, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(time.time()), self.context.get('environment'), self.context.get('time_of_day'),
                self.context.get('priority'), self.context.get('compassionate_adjustment', 0.0), sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log context update: {e}")

    # --- Callbacks / Input Methods ---
    def control_callback(self, msg: Any):
        """ROS callback for control output."""
        fields_map = {'data': ('', 'control_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        control_data = json.loads(data.get('control_data', '{}'))
        self.update_context_from_control(control_data)

    def health_callback(self, msg: Any):
        """ROS callback for health status."""
        fields_map = {'data': ('', 'health_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        health_data = json.loads(data.get('health_data', '{}'))
        self.update_context_from_health(health_data)

    def process_pending_updates(self, event: Any = None):
        """Process pending updates in dynamic or timer mode."""
        if self.pending_updates:
            update_data = self.pending_updates.popleft()
            if update_data.get('source') == 'control':
                self.update_context_from_control(update_data.get('data', {}))
            elif update_data.get('source') == 'health':
                self.update_context_from_health(update_data.get('data', {}))
            self.context_history.append(self.context.copy())

    # Dynamic input methods
    def update_context_direct(self, source: str, data: Dict[str, Any]):
        """Dynamic method to update context from source."""
        self.pending_updates.append({'source': source, 'data': data})
        _log_debug(self.node_name, f"Queued context update from {source}.")

    def get_context(self) -> Dict[str, Any]:
        return self.context.copy()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down ContextAwarenessNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Run the node."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Context Awareness Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = ContextAwarenessNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            node.update_context_direct('control', {"action": "execute_task"})
            node.update_context_direct('health', {"cpu_usage": 90})
            print(node.get_context())
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
