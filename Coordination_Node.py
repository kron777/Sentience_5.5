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
    CoordinationCommand = ROSMsgFallback
    ControlOutput = ROSMsgFallback
    MotivationStatus = ROSMsgFallback
    HealthStatus = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    CoordinationCommand = ROSMsgFallback
    ControlOutput = ROSMsgFallback
    MotivationStatus = ROSMsgFallback
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
            'coordination_node': {
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate coordination
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


class CoordinationNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'coordination_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "coordination_log.db")
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing coordination compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for updates
        self.coordination_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for coordination logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS coordination_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                command_type TEXT,
                commands_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Coordination Node online, harmonizing with compassionate and mindful synchronization.")

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_coordination_commands = None
        self.sub_control = None
        self.sub_motivation = None
        self.sub_health = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_coordination_commands = rospy.Publisher("coordination_commands", CoordinationCommand, queue_size=10)
            self.sub_control = rospy.Subscriber("control_output", ControlOutput, self.control_callback)
            self.sub_motivation = rospy.Subscriber("motivation_status", MotivationStatus, self.motivation_callback)
            self.sub_health = rospy.Subscriber("health_status", HealthStatus, self.health_callback)

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
        """Dynamic placeholder for sensory inputs influencing coordination compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on coordination
            if sensor_type == 'vision':
                self.pending_updates.append({'source': 'vision', 'data': {'status': 'crowded' if 'people' in str(processed) else 'clear'}})
            elif sensor_type == 'sound':
                self.pending_updates.append({'source': 'sound', 'data': {'status': 'noisy' if 'loud' in str(processed) else 'quiet'}})
            elif sensor_type == 'instructions':
                self.pending_updates.append({'source': 'instructions', 'data': {'command': processed.get('instruction', 'idle')}})
            # Compassionate bias: If distress detected, prioritize coordination for support
            if 'distress' in str(processed):
                self.agent_status['sensory'] = {'compassion_trigger': True}
            _log_debug(self.node_name, f"{sensor_type} input updated coordination at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.process_pending_updates()
            time.sleep(0.5)

    # --- Core Coordination Logic ---
    def update_agent_status(self, source: str, data: Dict[str, Any]) -> None:
        """Update the status of agents based on received data with compassionate bias."""
        try:
            self.agent_status[source] = data
            # Compassionate bias: If health indicates stress, adjust coordination to prioritize well-being
            if source == 'health' and data.get('error_rate', 0) > 0.1:
                self.agent_status[source]['compassionate_adjust'] = True
                if random.random() < self.ethical_compassion_bias:
                    self.agent_status[source]['priority'] = 'low'  # Slow down for safety
            _log_info(self.node_name, f"Updated {source} status: {json.dumps(data)}")
            self.coordinate_agents()
        except Exception as e:
            _log_error(self.node_name, f"Error updating agent status: {e}")

    def coordinate_agents(self) -> None:
        """Coordinate actions among agents based on their status with compassionate bias."""
        try:
            if not self.agent_status:
                _log_warn(self.node_name, "No agent status available")
                return

            command = {"timestamp": time.time(), "commands": {}}
            if self.agent_status.get("health", {}).get("error_rate", 0) > 0.1:
                command["commands"]["control"] = "pause_and_diagnose"
                # Compassionate bias: Add supportive note
                command["commands"]["compassion"] = "prioritize well-being"
            elif self.agent_status.get("motivation", {}).get("motivation_level", 0.5) > 0.7:
                command["commands"]["control"] = "increase_effort"
                if self.ethical_compassion_bias > 0.2:
                    command["commands"]["compassion"] = "balance effort with self-care"

            if command["commands"]:
                self.publish_coordination_command(command)
                _log_info(self.node_name, f"Published coordination command: {json.dumps(command)}")
        except Exception as e:
            _log_error(self.node_name, f"Error coordinating agents: {e}")

    # --- Callbacks / Input Methods ---
    def control_callback(self, msg: Any):
        """ROS callback for control output."""
        fields_map = {'data': ('', 'control_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        control_data = json.loads(data.get('control_data', '{}'))
        self.update_agent_status("control", control_data)

    def motivation_callback(self, msg: Any):
        """ROS callback for motivation status."""
        fields_map = {'data': ('', 'motivation_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        motivation_data = json.loads(data.get('motivation_data', '{}'))
        self.update_agent_status("motivation", motivation_data)

    def health_callback(self, msg: Any):
        """ROS callback for health status."""
        fields_map = {'data': ('', 'health_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        health_data = json.loads(data.get('health_data', '{}'))
        self.update_agent_status("health", health_data)

    def process_pending_updates(self, event: Any = None):
        """Process pending updates in dynamic or timer mode."""
        if self.pending_updates:
            update_data = self.pending_updates.popleft()
            if update_data.get('source') == 'control':
                self.update_agent_status('control', update_data.get('data', {}))
            elif update_data.get('source') == 'motivation':
                self.update_agent_status('motivation', update_data.get('data', {}))
            elif update_data.get('source') == 'health':
                self.update_agent_status('health', update_data.get('data', {}))
            self.coordination_history.append(self.agent_status.copy())

    # Dynamic input methods
    def update_status_direct(self, source: str, data: Dict[str, Any]):
        """Dynamic method to update agent status from source."""
        self.pending_updates.append({'source': source, 'data': data})
        _log_debug(self.node_name, f"Queued status update from {source}.")

    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        return self.agent_status.copy()

    def publish_coordination_command(self, command: Dict[str, Any]):
        """Publish coordination command (ROS or log)."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_coordination_commands:
                if hasattr(CoordinationCommand, 'data'):
                    self.pub_coordination_commands.publish(String(data=json.dumps(command)))
                else:
                    command_msg = CoordinationCommand(data=json.dumps(command))
                    self.pub_coordination_commands.publish(command_msg)
                _log_info(self.node_name, f"Published coordination command: {json.dumps(command)}")
            else:
                # Dynamic: Log
                _log_info(self.node_name, f"Dynamic coordination command: {json.dumps(command)}")
            # Log to DB with sensory snapshot
            sensory_snapshot = json.dumps(self.sensory_data)
            self._log_coordination_event(command, sensory_snapshot)
        except Exception as e:
            _log_error(self.node_name, f"Failed to publish coordination command: {e}")

    def _log_coordination_event(self, command: Dict[str, Any], sensory_snapshot: str):
        """Log coordination event to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO coordination_log (id, timestamp, command_type, commands_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(time.time()), 'coordination', json.dumps(command['commands']), sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log coordination event: {e}")

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down CoordinationNode.")
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
    parser = argparse.ArgumentParser(description='Sentience Coordination Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = CoordinationNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            node.update_status_direct('control', {"action": "move_forward"})
            node.update_status_direct('motivation', {"motivation_level": 0.8})
            node.update_status_direct('health', {"error_rate": 0.05})
            print(node.get_agent_status())
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
