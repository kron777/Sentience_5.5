```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

# --- Asyncio Imports (for potential future async operations) ---
import asyncio
import threading

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
    ErrorLog = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    ErrorLog = ROSMsgFallback


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
            'error_logger_node': {
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate error handling (e.g., emphasize learning)
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


class ErrorLoggerNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'error_logger_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "error_log.db")
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., log errors with compassionate notes based on sensory context)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.pending_logs: Deque[Dict[str, Any]] = deque(maxlen=20)  # Queue for pending logs
        self.error_history: Deque[Dict[str, Any]] = deque(maxlen=100)  # History for patterns

        # Initialize SQLite database for error logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                error_type TEXT,
                description TEXT,
                severity REAL,
                compassionate_note TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Error Logger Node online, logging with compassionate and mindful error reflection.")

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_error_summary = None
        self.sub_error_reports = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_error_summary = rospy.Publisher('/error_summary', ErrorLog, queue_size=10)
            self.sub_error_reports = rospy.Subscriber('/error_reports', String, self.error_report_callback)

            # Sensory subscribers
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(5.0), self.flush_pending_logs)
        else:
            # Dynamic mode: Polling loop
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing error logs compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on error logging (e.g., log errors with compassionate context)
            if sensor_type == 'vision':
                self.pending_logs.append({'error_type': 'perception_error', 'description': processed.get('description', 'visual anomaly'), 'severity': random.uniform(0.1, 0.5)})
            elif sensor_type == 'sound':
                self.pending_logs.append({'error_type': 'auditory_error', 'description': processed.get('transcription', 'audio anomaly'), 'severity': random.uniform(0.2, 0.6)})
            elif sensor_type == 'instructions':
                self.pending_logs.append({'error_type': 'command_error', 'description': processed.get('instruction', 'invalid command'), 'severity': random.uniform(0.3, 0.7)})
            # Compassionate bias: If distress in sound, add compassionate note to log
            if 'distress' in str(processed):
                self.pending_logs[-1]['compassionate_note'] = "Emphasizing compassionate error handling for potential distress."
            _log_debug(self.node_name, f"{sensor_type} input updated error context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.flush_pending_logs()
            time.sleep(5.0)

    # --- Core Error Logging Logic ---
    def log(self, error_type: str, description: str, severity: float = 0.5, context: Dict[str, Any] = None):
        """Log an error with compassionate note."""
        # Compassionate bias: Add note if severity high
        compassionate_note = ""
        if severity > 0.7 and self.ethical_compassion_bias > 0.1:
            compassionate_note = f"Compassionate reflection: Prioritize learning from this to avoid harm. Bias: {self.ethical_compassion_bias}."

        error_entry = {
            'id': str(uuid.uuid4()),
            'timestamp': str(time.time()),
            'error_type': error_type,
            'description': description,
            'severity': severity,
            'compassionate_note': compassionate_note,
            'context': context or {}
        }
        self.pending_logs.append(error_entry)
        self.error_history.append(error_entry)
        _log_error(self.node_name, f"Logged {error_type}: {description} (Severity: {severity}).")

    def error_report_callback(self, msg: Any):
        """ROS callback for error reports."""
        fields_map = {'data': ('', 'error_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        error_data = json.loads(data.get('error_data', '{}'))
        self.log(
            error_data.get('error_type', 'unknown'),
            error_data.get('description', 'no description'),
            error_data.get('severity', 0.5),
            error_data.get('context', {})
        )

    def receive_error_direct(self, error_type: str, description: str, severity: float = 0.5, context: Dict[str, Any] = None):
        """Dynamic method for error logging."""
        self.log(error_type, description, severity, context)
        _log_debug(self.node_name, f"Dynamic error logged: {error_type}.")

    def flush_pending_logs(self, event: Any = None):
        """Flush pending logs to DB with sensory snapshot."""
        if not self.pending_logs:
            return
        entries = list(self.pending_logs)
        self.pending_logs.clear()
        try:
            self.cursor.executemany('''
                INSERT INTO error_log (id, timestamp, error_type, description, severity, compassionate_note, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', [(e['id'], e['timestamp'], e['error_type'], e['description'], e['severity'], e['compassionate_note'], json.dumps(self.sensory_data)) for e in entries])
            self.conn.commit()
            _log_info(self.node_name, f"Flushed {len(entries)} error logs to DB.")
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to flush error logs: {e}")
            for entry in entries:
                self.pending_logs.append(entry)

    def get_error_history(self) -> Deque[Dict[str, Any]]:
        """Get recent error history."""
        return self.error_history.copy()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down ErrorLoggerNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        self.flush_pending_logs()
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
    parser = argparse.ArgumentParser(description='Sentience Error Logger Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = ErrorLoggerNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            node.log('test_error', 'Dynamic test error', 0.5)
            time.sleep(2)
            print("Error logged dynamically.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
