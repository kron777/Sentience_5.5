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
    ControlOutput = ROSMsgFallback
    IntegrationOutput = ROSMsgFallback
    MotivationStatus = ROSMsgFallback
    AdaptationOutput = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    ControlOutput = ROSMsgFallback
    IntegrationOutput = ROSMsgFallback
    MotivationStatus = ROSMsgFallback
    AdaptationOutput = ROSMsgFallback


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
            'control_node': {
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate priority adjustments
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


class ControlNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'control_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "control_log.db")
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing priority compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.current_action = {"action": "idle", "priority": "low"}
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for updates
        self.control_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for control logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS control_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                action TEXT,
                priority TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Control Node online, steering with compassionate and mindful guidance.")

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_control_output = None
        self.sub_integration = None
        self.sub_motivation = None
        self.sub_adaptation = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_control_output = rospy.Publisher("control_output", ControlOutput, queue_size=10)
            self.sub_integration = rospy.Subscriber("integration_output", IntegrationOutput, self.integration_callback)
            self.sub_motivation = rospy.Subscriber("motivation_status", MotivationStatus, self.motivation_callback)
            self.sub_adaptation = rospy.Subscriber("adaptation_output", AdaptationOutput, self.adaptation_callback)

            # Sensory subscribers
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(0.5), self.process_pending_updates)
        else:
            # Dynamic mode: Polling loop
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing control compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on action/priority
            if sensor_type == 'vision':
                self.pending_updates.append({'source': 'vision', 'data': {'action': 'observe' if 'object' in str(processed) else 'idle'}})
            elif sensor_type == 'sound':
                self.pending_updates.append({'source': 'sound', 'data': {'priority': 'high' if 'urgent' in str(processed) else 'normal'}})
            elif sensor_type == 'instructions':
                self.pending_updates.append({'source': 'instructions', 'data': {'action': processed.get('action', 'idle')}})
            # Compassionate bias: If distress detected, lower priority for self-care
            if 'distress' in str(processed):
                self.current_action['priority'] = 'low'
            _log_debug(self.node_name, f"{sensor_type} input updated control at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.process_pending_updates()
            time.sleep(0.5)

    # --- Core Control Logic ---
    def update_action(self, integrated_data: Dict[str, Any]) -> None:
        """Update action based on integrated data with compassionate bias."""
        try:
            action = integrated_data.get("final_action", {})
            self.current_action["action"] = action.get("action", "idle")
            # Compassionate bias: If high priority, adjust for compassionate context
            if self.ethical_compassion_bias > 0.2 and self.current_action["priority"] == "high":
                self.current_action["priority"] = "medium" if random.random() < self.ethical_compassion_bias else self.current_action["priority"]
            _log_info(self.node_name, f"Updated action to {self.current_action['action']}")
            self.execute_action()
        except Exception as e:
            _log_error(self.node_name, f"Error updating action: {e}")

    def adjust_priority(self, motivation_data: Dict[str, Any]) -> None:
        """Adjust priority based on motivation with compassionate bias."""
        try:
            motivation_level = motivation_data.get("motivation_level", 0.5)
            if motivation_level < 0.3:
                self.current_action["priority"] = "low"
            elif motivation_level > 0.7:
                self.current_action["priority"] = "high"
            else:
                self.current_action["priority"] = "medium"
            # Compassionate bias: Lower priority if high motivation but potential harm
            if self.current_action["priority"] == "high" and self.ethical_compassion_bias > 0.2:
                self.current_action["priority"] = "medium"
            _log_info(self.node_name, f"Adjusted priority to {self.current_action['priority']}")
            self.execute_action()
        except Exception as e:
            _log_error(self.node_name, f"Error adjusting priority: {e}")

    def apply_adaptation(self, adaptation_data: Dict[str, Any]) -> None:
        """Apply adaptation with compassionate bias."""
        try:
            strategy = adaptation_data.get("strategy", "default")
            if strategy == "optimized":
                self.current_action["priority"] = "high" if self.current_action["priority"] != "low" else "medium"
            elif strategy == "conservative":
                self.current_action["priority"] = "low" if self.current_action["priority"] != "high" else "medium"
            # Compassionate bias: Favor conservative in uncertain contexts
            if strategy == "optimized" and self.ethical_compassion_bias > 0.3:
                self.current_action["priority"] = "medium"
            _log_info(self.node_name, f"Applied adaptation, priority: {self.current_action['priority']}")
            self.execute_action()
        except Exception as e:
            _log_error(self.node_name, f"Error applying adaptation: {e}")

    def execute_action(self) -> None:
        """Execute action and log."""
        try:
            output = json.dumps(self.current_action)
            if ROS_AVAILABLE and self.ros_enabled and self.pub_control_output:
                if hasattr(ControlOutput, 'data'):
                    self.pub_control_output.publish(String(data=output))
                else:
                    action_msg = ControlOutput(data=output)
                    self.pub_control_output.publish(action_msg)
            else:
                # Dynamic: Log
                _log_info(self.node_name, f"Dynamic control output: {output}")
            # Log to DB with sensory snapshot
            sensory_snapshot = json.dumps(self.sensory_data)
            self._log_control_action(sensory_snapshot)
            _log_info(self.node_name, f"Executed action: {output}")
        except Exception as e:
            _log_error(self.node_name, f"Error executing action: {e}")

    def _log_control_action(self, sensory_snapshot: str):
        """Log control action to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO control_log (id, timestamp, action, priority, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(time.time()), self.current_action.get('action'), self.current_action.get('priority'), sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log control action: {e}")

    # --- Callbacks / Input Methods ---
    def integration_callback(self, msg: Any):
        """ROS callback for integration output."""
        fields_map = {'data': ('', 'integrated_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        integrated_data = json.loads(data.get('integrated_data', '{}'))
        self.update_action(integrated_data)

    def motivation_callback(self, msg: Any):
        """ROS callback for motivation status."""
        fields_map = {'data': ('', 'motivation_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        motivation_data = json.loads(data.get('motivation_data', '{}'))
        self.adjust_priority(motivation_data)

    def adaptation_callback(self, msg: Any):
        """ROS callback for adaptation output."""
        fields_map = {'data': ('', 'adaptation_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        adaptation_data = json.loads(data.get('adaptation_data', '{}'))
        self.apply_adaptation(adaptation_data)

    def process_pending_updates(self, event: Any = None):
        """Process pending updates in dynamic or timer mode."""
        if self.pending_updates:
            update_data = self.pending_updates.popleft()
            if update_data.get('source') == 'integration':
                self.update_action(update_data.get('data', {}))
            elif update_data.get('source') == 'motivation':
                self.adjust_priority(update_data.get('data', {}))
            elif update_data.get('source') == 'adaptation':
                self.apply_adaptation(update_data.get('data', {}))
            self.control_history.append(self.current_action.copy())

    # Dynamic input methods
    def update_from_source(self, source: str, data: Dict[str, Any]):
        """Dynamic method to update from source."""
        self.pending_updates.append({'source': source, 'data': data})
        _log_debug(self.node_name, f"Queued update from {source}.")

    def get_current_action(self) -> Dict[str, Any]:
        return self.current_action.copy()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down ControlNode.")
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
                    time.sleep(1)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Control Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = ControlNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            node.update_from_source('integration', {"final_action": {"action": "navigate"}})
            node.update_from_source('motivation', {"motivation_level": 0.8})
            node.update_from_source('adaptation', {"strategy": "optimized"})
            print(node.get_current_action())
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
