```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique drive event IDs
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
    MotivationState = ROSMsgFallback
    MemoryFeedback = ROSMsgFallback
    BatteryState = ROSMsgFallback
    SurpriseDetector = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    MotivationState = ROSMsgFallback
    MemoryFeedback = ROSMsgFallback
    BatteryState = ROSMsgFallback
    SurpriseDetector = ROSMsgFallback


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
            'drive_system_node': {
                'decay_rate': 0.02,
                'drive_threshold': 0.5,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate drives (e.g., social/safety)
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


class DriveSystemNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'drive_system_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "drive_log.db")
        self.decay_rate = self.params.get('decay_rate', 0.02)
        self.drive_threshold = self.params.get('drive_threshold', 0.5)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing drives compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.drives = {
            'curiosity': 0.4,
            'energy': 0.5,
            'social_contact': 0.3,
            'safety': 0.6
        }
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for updates
        self.drive_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for drive logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS drive_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                dominant_drive TEXT,
                overall_drive_level REAL,
                active_goals_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Drive System Node online, motivating with compassionate and mindful drive modulation.")

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_motivation = None
        self.sub_memory = None
        self.sub_battery = None
        self.sub_surprise = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_motivation = rospy.Publisher('/motivation_state', MotivationState, queue_size=10)
            self.sub_memory = rospy.Subscriber('/memory_feedback', MemoryFeedback, self.memory_feedback_callback)
            self.sub_battery = rospy.Subscriber('/battery_state', BatteryState, self.battery_callback)
            self.sub_surprise = rospy.Subscriber('/surprise_detector', SurpriseDetector, self.surprise_callback)

            # Sensory subscribers
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(4), self.update_drives)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing drives compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on drives
            if sensor_type == 'vision':
                self.pending_updates.append({'type': 'memory', 'data': {'novelty_score': random.uniform(0.2, 0.8)}})
            elif sensor_type == 'sound':
                self.pending_updates.append({'type': 'surprise', 'data': {'surprise_intensity': random.uniform(0.1, 0.5)}})
            elif sensor_type == 'instructions':
                self.pending_updates.append({'type': 'battery', 'data': {'battery_level': random.uniform(0.4, 0.9)}})
            # Compassionate bias: If distress in sound, boost safety drive compassionately
            if 'distress' in str(processed):
                self.drives['safety'] = min(1.0, self.drives['safety'] + self.ethical_compassion_bias)
            _log_debug(self.node_name, f"{sensor_type} input updated drive context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.update_drives()
            time.sleep(4)

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    # --- Core Drive System Logic ---
    def memory_feedback_callback(self, msg: Any):
        """Increase curiosity if repeated inputs detected (low novelty)."""
        fields_map = {'data': ('', 'memory_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        memory_data = json.loads(data.get('memory_data', '{}'))
        novelty_score = memory_data.get("novelty_score", 0.5)
        self.drives['curiosity'] += (0.5 - novelty_score) * 0.1
        self.drives['curiosity'] = min(1.0, max(0.0, self.drives['curiosity']))
        _log_debug(self.node_name, f"Updated curiosity drive to {self.drives['curiosity']:.2f} based on novelty {novelty_score}.")

    def battery_callback(self, msg: Any):
        """Drive for energy management."""
        fields_map = {'data': ('', 'battery_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        battery_data = json.loads(data.get('battery_data', '{}'))
        level = battery_data.get("battery_level", 1.0)
        self.drives['energy'] = 1.0 - level  # Low battery = high energy drive
        self.drives['energy'] = min(1.0, max(0.0, self.drives['energy']))
        # Compassionate bias: If low battery, boost safety compassionately
        if level < 0.2 and self.ethical_compassion_bias > 0.1:
            self.drives['safety'] = min(1.0, self.drives['safety'] + self.ethical_compassion_bias)
        _log_debug(self.node_name, f"Updated energy drive to {self.drives['energy']:.2f} based on battery level {level}.")

    def surprise_callback(self, msg: Any):
        """Boost safety drive if surprises are frequent."""
        fields_map = {'data': ('', 'surprise_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        surprise_data = json.loads(data.get('surprise_data', '{}'))
        intensity = surprise_data.get("surprise_intensity", 0.0)
        self.drives['safety'] += intensity * 0.1
        self.drives['safety'] = min(1.0, max(0.0, self.drives['safety']))
        # Compassionate bias: High surprise -> boost social for support
        if intensity > 0.5 and self.ethical_compassion_bias > 0.1:
            self.drives['social_contact'] = min(1.0, self.drives['social_contact'] + self.ethical_compassion_bias * 0.5)
        _log_debug(self.node_name, f"Updated safety drive to {self.drives['safety']:.2f} based on surprise intensity {intensity}.")

    def update_drives(self, event: Any = None):
        """Periodic drive update with compassionate bias."""
        # Decay all drives slowly
        for key in self.drives:
            self.drives[key] = max(0.0, self.drives[key] - self.decay_rate)

        # Determine dominant drive
        dominant_drive = max(self.drives, key=self.drives.get)
        urgency = self.drives[dominant_drive]

        # Active goals based on drives, with compassionate bias
        active_goals = []
        if urgency > self.drive_threshold:
            if dominant_drive == 'curiosity':
                active_goals = ['explore', 'seek_novelty']
            elif dominant_drive == 'energy':
                active_goals = ['locate_power', 'reduce_activity']
            elif dominant_drive == 'social_contact':
                active_goals = ['initiate_dialogue']
                # Compassionate bias: If compassion high, add empathetic goal
                if random.random() < self.ethical_compassion_bias:
                    active_goals.append('offer_support')
            elif dominant_drive == 'safety':
                active_goals = ['scan_environment', 'avoid_threats']

        state_msg = {
            "timestamp": time.time(),
            "dominant_drive": dominant_drive,
            "overall_drive_level": round(urgency, 2),
            "active_goals_json": json.dumps(active_goals),
            "compassionate_adjustment": self.ethical_compassion_bias if 'social_contact' in active_goals else 0.0
        }

        # Log to DB with sensory snapshot
        sensory_snapshot = json.dumps(self.sensory_data)
        self._log_drive_update(state_msg, sensory_snapshot)

        if ROS_AVAILABLE and self.ros_enabled and self.pub_motivation:
            if hasattr(MotivationState, 'data'):
                self.pub_motivation.publish(String(data=json.dumps(state_msg)))
            else:
                motivation_msg = MotivationState(data=json.dumps(state_msg))
                self.pub_motivation.publish(motivation_msg)
        else:
            # Dynamic: Log
            _log_info(self.node_name, f"Dynamic motivation state: {state_msg}")

        _log_info(self.node_name, f"Dominant Drive: {dominant_drive} (Urgency: {urgency:.2f})")

    def _log_drive_update(self, state_msg: Dict[str, Any], sensory_snapshot: str):
        """Log drive update to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO drive_log (id, timestamp, dominant_drive, overall_drive_level, active_goals_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), state_msg.get('dominant_drive'),
                state_msg.get('overall_drive_level'), state_msg.get('active_goals_json'), sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log drive update: {e}")

    def process_pending_updates(self, event: Any = None):
        """Process pending updates in dynamic or timer mode."""
        if self.pending_updates:
            update_data = self.pending_updates.popleft()
            if update_data.get('type') == 'memory':
                self.memory_feedback_callback(update_data.get('data', {}))
            elif update_data.get('type') == 'battery':
                self.battery_callback(update_data.get('data', {}))
            elif update_data.get('type') == 'surprise':
                self.surprise_callback(update_data.get('data', {}))
            self.drive_history.append(self.drives.copy())

    # Dynamic input methods
    def update_drive_direct(self, drive_type: str, delta: float):
        """Dynamic method to update a drive."""
        if drive_type in self.drives:
            self.drives[drive_type] = min(1.0, max(0.0, self.drives[drive_type] + delta))
            _log_debug(self.node_name, f"Updated {drive_type} drive to {self.drives[drive_type]:.2f}.")

    def get_drives(self) -> Dict[str, float]:
        return self.drives.copy()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down DriveSystemNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
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
    parser = argparse.ArgumentParser(description='Sentience Drive System Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = DriveSystemNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            node.update_drive_direct('curiosity', 0.2)
            time.sleep(2)
            print(node.get_drives())
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
