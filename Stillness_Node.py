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
from threading import Event, Thread

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
    StillnessStatus = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    StillnessStatus = ROSMsgFallback


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
            'stillness_node': {
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate pauses (e.g., longer reflection for stress)
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


class StillnessNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'stillness_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "stillness_log.db")
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound triggering pauses compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.is_still = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initially not paused
        self.pending_triggers: Deque[Dict[str, Any]] = deque(maxlen=5)  # Queue for triggers
        self.pause_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for stillness logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS stillness_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                duration REAL,
                trigger_type TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Stillness Node online, pausing for compassionate reflection and consolidation.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_stillness_status = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_stillness_status = rospy.Publisher('/sentience/stillness_status', StillnessStatus, queue_size=10)
            rospy.Subscriber('/trigger_stillness', String, self.trigger_stillness_callback)
            rospy.Timer(rospy.Duration(10.0), self._periodic_check_and_publish)  # Periodic status
        else:
            # Dynamic mode: Start polling thread for simulated triggers
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs triggering pauses compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory trigger for pause (e.g., high salience -> reflect)
            if sensor_type == 'vision':
                if random.random() < 0.4:
                    self.pending_triggers.append({'type': 'pause', 'data': {'duration': 2.0, 'reason': 'visual overload'}})
            elif sensor_type == 'sound':
                if 'distress' in str(processed) and self.ethical_compassion_bias > 0.1:
                    self.pending_triggers.append({'type': 'pause', 'data': {'duration': 5.0, 'reason': 'emotional distress'}})
            elif sensor_type == 'instructions':
                self.pending_triggers.append({'type': 'pause', 'data': {'duration': 1.0, 'reason': 'instruction reflection'}})
            _log_debug(self.node_name, f"{sensor_type} input updated stillness context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._simulate_trigger()
            self._process_pending_triggers()
            self._periodic_check_and_publish()
            time.sleep(1.0)

    def _simulate_trigger(self):
        """Simulate a trigger in non-ROS mode."""
        if random.random() < 0.2:
            duration = random.uniform(1.0, 5.0)
            self.pending_triggers.append({'type': 'pause', 'data': {'duration': duration, 'reason': 'simulated reflection'}})
            _log_debug(self.node_name, f"Simulated trigger for pause of {duration}s.")

    # --- Core Stillness Logic ---
    def enter_stillness(self, duration: float):
        """Enter stillness state for 'duration' seconds with compassionate adjustment."""
        if self.is_still:
            _log_warn(self.node_name, "Already in stillness.")
            return
        # Compassionate bias: Extend duration slightly for deeper reflection if stressed
        duration = duration + (self.ethical_compassion_bias * 0.5)
        self.is_still = True
        self._pause_event.clear()

        def _stillness_timer():
            _log_info(self.node_name, f"Entering stillness for {duration}s...")
            time.sleep(duration)
            self.exit_stillness()

        Thread(target=_stillness_timer, daemon=True).start()
        _log_info(self.node_name, "Initiated stillness mode.")

    def exit_stillness(self):
        """Exit stillness state and resume processing."""
        self.is_still = False
        self._pause_event.set()
        _log_info(self.node_name, "Exiting stillness, resuming cognition with refreshed awareness.")
        self._log_stillness_event('exit', self._get_current_time())

    def wait_if_still(self):
        """Call this in cognitive loops to pause if stillness active."""
        self._pause_event.wait()

    def trigger_stillness_callback(self, msg: Any):
        """Handle incoming stillness triggers."""
        fields_map = {'data': ('', 'trigger_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        trigger_data = json.loads(data.get('trigger_data', '{}'))
        duration = trigger_data.get('duration', 2.0)
        self.enter_stillness(duration)

    def _process_pending_triggers(self):
        """Process pending triggers in dynamic or timer mode."""
        while self.pending_triggers:
            update_data = self.pending_triggers.popleft()
            if update_data.get('type') == 'pause':
                duration = update_data['data']['duration']
                self.enter_stillness(duration)
            self.pause_history.append(update_data)

    def _periodic_check_and_publish(self):
        """Periodic check and publishing of status."""
        self.publish_stillness_status()

    def publish_stillness_status(self):
        """Publish stillness status (ROS or log)."""
        status = {
            'is_still': self.is_still,
            'timestamp': self._get_current_time()
        }
        if ROS_AVAILABLE and self.ros_enabled and self.pub_stillness_status:
            if hasattr(StillnessStatus, 'data'):
                self.pub_stillness_status.publish(String(data=json.dumps(status)))
            else:
                status_msg = StillnessStatus(data=json.dumps(status))
                self.pub_stillness_status.publish(status_msg)
        else:
            # Dynamic mode: Log
            _log_debug(self.node_name, f"Stillness status: {'paused' if self.is_still else 'active'}")

    def _log_stillness_event(self, event_type: str, timestamp: str):
        """Log stillness event to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO stillness_log (id, timestamp, duration, trigger_type, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), timestamp, 0.0 if event_type == 'exit' else time.time() - self.last_pause_time, event_type, json.dumps(self.sensory_data)
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log stillness event: {e}")

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down StillnessNode.")
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
                    self._simulate_trigger()
                    self._process_pending_triggers()
                    self._periodic_check_and_publish()
                    time.sleep(1.0)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Stillness Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = StillnessNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate a pause
            node.enter_stillness(3.0)
            time.sleep(4)
            print("Stillness simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
