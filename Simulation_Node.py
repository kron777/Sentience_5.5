```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique message IDs
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
    Range = ROSMsgFallback  # For proximity, but use String for generality
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    Range = ROSMsgFallback


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
                data[target_key] = getattr(msg, msg_in_msg, default_val)
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
            'simulation_node': {
                'update_rate_hz': 5.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate simulations (e.g., safe proximity ranges)
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


class SimulationNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'simulation_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "simulation_log.db")
        self.update_rate_hz = self.params.get('update_rate_hz', 5.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing simulations compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.pending_simulations: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for simulations
        self.simulation_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for simulation logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                proximity_range REAL,
                state TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Simulation Node online, simulating with compassionate and ethical scenario generation.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_proximity = None
        self.pub_state = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_proximity = rospy.Publisher('/sentience/raw_sensors/proximity', RawSensorData, queue_size=10)
            self.pub_state = rospy.Publisher('/sentience/world_model_state', String, queue_size=10)
            rospy.Timer(rospy.Duration(1.0 / self.update_rate_hz), self.simulate_loop)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_simulation_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing simulations compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on simulations
            if sensor_type == 'vision':
                self.pending_simulations.append({'type': 'proximity', 'data': {'range': random.uniform(0.5, 2.5)}})
            elif sensor_type == 'sound':
                self.pending_simulations.append({'type': 'state', 'data': {'state': 'processing' if random.random() < 0.7 else 'idle'}})
            elif sensor_type == 'instructions':
                self.pending_simulations.append({'type': 'proximity', 'data': {'range': random.uniform(0.2, 1.0)}})
            # Compassionate bias: If distress in sound, simulate safer proximity (farther range)
            if 'distress' in str(processed):
                self.pending_simulations[-1]['data']['range'] = min(4.0, self.pending_simulations[-1]['data']['range'] + self.ethical_compassion_bias * 1.0)
            _log_debug(self.node_name, f"{sensor_type} input updated simulation context at {timestamp}")
        return placeholder_callback

    def _dynamic_simulation_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.simulate_loop()
            time.sleep(1.0 / self.update_rate_hz)

    # --- Core Simulation Logic ---
    def simulate_proximity(self) -> Dict[str, Any]:
        """Simulate proximity data."""
        range_value = random.uniform(0.2, 4.0)
        # Compassionate bias: Occasionally bias toward safer (higher) ranges
        if random.random() < self.ethical_compassion_bias:
            range_value = min(4.0, range_value + 0.5)
        return {
            'header': {'stamp': self._get_current_time()},
            'radiation_type': 'INFRARED',
            'field_of_view': 0.5,
            'min_range': 0.2,
            'max_range': 4.0,
            'range': range_value
        }

    def simulate_state(self) -> str:
        """Simulate state string."""
        states = ['idle', 'exploring', 'charging', 'processing']
        # Compassionate bias: Favor 'charging' or 'processing' for self-care
        if random.random() < self.ethical_compassion_bias:
            state = random.choice(['charging', 'processing'])
        else:
            state = random.choice(states)
        return state

    def simulate_loop(self):
        """Primary simulation loop."""
        proximity_data = self.simulate_proximity()
        state_data = self.simulate_state()

        # Log to DB with sensory snapshot
        sensory_snapshot = json.dumps(self.sensory_data)
        self._log_simulation_data(proximity_data, state_data, sensory_snapshot)

        if ROS_AVAILABLE and self.ros_enabled:
            if self.pub_proximity:
                if hasattr(RawSensorData, 'data'):
                    self.pub_proximity.publish(String(data=json.dumps(proximity_data)))
                else:
                    prox_msg = RawSensorData()
                    prox_msg.header.stamp = rospy.Time.now()
                    prox_msg.radiation_type = 'INFRARED'
                    prox_msg.field_of_view = 0.5
                    prox_msg.min_range = 0.2
                    prox_msg.max_range = 4.0
                    prox_msg.range = proximity_data['range']
                    self.pub_proximity.publish(prox_msg)
            if self.pub_state:
                self.pub_state.publish(String(data=state_data))
        else:
            # Dynamic mode: Log
            _log_info(self.node_name, f"Simulated proximity: {proximity_data['range']:.2f}m, state: {state_data}")

        _log_debug(self.node_name, f"Published proximity: {proximity_data['range']:.2f}m, state: {state_data}")

    def _log_simulation_data(self, proximity_data: Dict[str, Any], state_data: str, sensory_snapshot: str):
        """Log simulation data to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO simulation_log (id, timestamp, proximity_range, state, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), proximity_data['range'], state_data, sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log simulation data: {e}")

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down SimulationNode.")
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
                    time.sleep(1.0 / self.update_rate_hz)
                    self.simulate_loop()
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Simulation Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = SimulationNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Run a few simulations
            for _ in range(5):
                node.simulate_loop()
                time.sleep(1.0 / node.update_rate_hz)
            print("Simulation run complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
