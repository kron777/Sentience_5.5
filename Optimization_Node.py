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
    OptimizationSuggestions = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    OptimizationSuggestions = ROSMsgFallback


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
            'optimization_node': {
                'alert_threshold': 0.3,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate suggestions (e.g., growth-oriented)
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


class OptimizationNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'optimization_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "optimization_log.db")
        self.alert_threshold = self.params.get('alert_threshold', 0.3)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing suggestions compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.optimization_suggestions: Deque[Dict[str, Any]] = deque(maxlen=20)  # Bounded for efficiency
        self.pending_data: Deque[Dict[str, Any]] = deque(maxlen=50)  # Queue for incoming data
        self.performance_history: Deque[Dict[str, Any]] = deque(maxlen=100)  # Recent performance for trends

        # Initialize SQLite database for optimization logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                type TEXT,
                action TEXT,
                priority TEXT,
                reason TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Optimization Node online, suggesting with compassionate and mindful self-improvement focus.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_optimization_suggestions = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_optimization_suggestions = rospy.Publisher('optimization_suggestions', OptimizationSuggestions, queue_size=10)
            rospy.Subscriber('monitoring_output', String, self.monitoring_callback)
            rospy.Subscriber('memory_status', String, self.memory_callback)
            rospy.Subscriber('self_awareness_status', String, self.self_awareness_callback)
            rospy.Timer(rospy.Duration(5.0), self._check_and_publish)
        else:
            # Dynamic mode: Start polling thread for simulated data
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing suggestions compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on monitoring data
            if sensor_type == 'vision':
                self.pending_data.append({'type': 'monitoring', 'data': {'performance': random.uniform(0.4, 0.9)}})
            elif sensor_type == 'sound':
                self.pending_data.append({'type': 'memory', 'data': {'usage': random.uniform(0.6, 0.95)}})
            elif sensor_type == 'instructions':
                self.pending_data.append({'type': 'self_awareness', 'data': {'coherence': random.uniform(0.3, 0.8)}})
            # Compassionate bias: If distress in sound, bias toward supportive suggestions
            if 'distress' in str(processed):
                self.ethical_compassion_bias = min(1.0, self.ethical_compassion_bias + 0.1)
            _log_debug(self.node_name, f"{sensor_type} input updated optimization context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            # Simulate periodic data
            self._simulate_data('monitoring', {'status': 'normal' if random.random() > 0.2 else 'alert'})
            self._simulate_data('memory', {'total_entries': random.randint(80, 120)})
            self._simulate_data('self_awareness', {'coherence': random.uniform(0.4, 0.9)})
            self._check_and_publish()
            time.sleep(5.0)

    def _simulate_data(self, data_type: str, data: Dict[str, Any]):
        """Simulate incoming data in non-ROS mode."""
        entry = {
            'type': data_type,
            'data': data,
            'timestamp': time.time()
        }
        self.pending_data.append(entry)
        _log_debug(self.node_name, f"Simulated {data_type} data: {json.dumps(data)}")

    # --- Core Optimization Logic ---
    def monitoring_callback(self, msg: Any):
        """Handle incoming monitoring data."""
        fields_map = {'data': ('', 'monitoring_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        monitoring_data = json.loads(data.get('monitoring_data', '{}'))
        self.analyze_performance(monitoring_data)

    def memory_callback(self, msg: Any):
        """Handle incoming memory status data."""
        fields_map = {'data': ('', 'memory_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        memory_data = json.loads(data.get('memory_data', '{}'))
        self.analyze_memory_usage(memory_data)

    def self_awareness_callback(self, msg: Any):
        """Handle incoming self-awareness data."""
        fields_map = {'data': ('', 'awareness_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        awareness_data = json.loads(data.get('awareness_data', '{}'))
        self.analyze_coherence(awareness_data)

    def analyze_performance(self, monitoring_data: Dict[str, Any]) -> None:
        """Analyze performance metrics and suggest optimizations with compassionate bias."""
        try:
            if monitoring_data.get("status") == "alert":
                # Compassionate bias: Add encouraging language in suggestion
                suggestion = {
                    "timestamp": time.time(),
                    "type": "performance",
                    "action": "reallocate_resources",
                    "priority": "high",
                    "reason": monitoring_data.get("message", "Performance issue detected"),
                    "compassionate_note": f"With compassion, view this as a growth opportunity. Bias: {self.ethical_compassion_bias}."
                }
                self.optimization_suggestions.append(suggestion)
                _log_info(self.node_name, f"Suggested optimization: {json.dumps(suggestion)}")
        except Exception as e:
            _log_error(self.node_name, f"Error analyzing performance: {e}")

    def analyze_memory_usage(self, memory_data: Dict[str, Any]) -> None:
        """Analyze memory usage and suggest optimizations with compassionate bias."""
        try:
            total_entries = memory_data.get("total_entries", 0)
            if total_entries >= 90:  # Near max capacity (assuming max_entries = 100)
                suggestion = {
                    "timestamp": time.time(),
                    "type": "memory",
                    "action": "clear_old_entries",
                    "priority": "medium",
                    "reason": f"Memory at {total_entries} entries, nearing limit",
                    "compassionate_note": "Gently prune to maintain clarity and focus."
                }
                self.optimization_suggestions.append(suggestion)
                _log_info(self.node_name, f"Suggested optimization: {json.dumps(suggestion)}")
        except Exception as e:
            _log_error(self.node_name, f"Error analyzing memory usage: {e}")

    def analyze_coherence(self, awareness_data: Dict[str, Any]) -> None:
        """Analyze coherence and suggest optimizations with compassionate bias."""
        try:
            coherence = awareness_data.get("coherence", 0.8)
            if coherence < 0.5:
                suggestion = {
                    "timestamp": time.time(),
                    "type": "coherence",
                    "action": "recalibrate_system",
                    "priority": "high",
                    "reason": f"Low coherence detected: {coherence}",
                    "compassionate_note": f"With gentle self-compassion, realign for harmony. Bias: {self.ethical_compassion_bias}."
                }
                self.optimization_suggestions.append(suggestion)
                _log_info(self.node_name, f"Suggested optimization: {json.dumps(suggestion)}")
        except Exception as e:
            _log_error(self.node_name, f"Error analyzing coherence: {e}")

    def _check_and_publish(self):
        """Check for new suggestions and publish."""
        if self.optimization_suggestions:
            latest_suggestion = self.optimization_suggestions[-1]
            self.publish_suggestions(latest_suggestion)
            # Log to DB with sensory snapshot
            sensory_snapshot = json.dumps(self.sensory_data)
            self._log_optimization_suggestion(latest_suggestion, sensory_snapshot)
        else:
            # Publish normal status
            status = {"status": "normal", "alert": False}
            self.publish_status(status)

    def publish_suggestions(self, suggestion: Dict[str, Any]):
        """Publish optimization suggestion (ROS or log)."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_optimization_suggestions:
                if hasattr(OptimizationSuggestions, 'data'):
                    self.pub_optimization_suggestions.publish(String(data=json.dumps(suggestion)))
                else:
                    suggestion_msg = OptimizationSuggestions(data=json.dumps(suggestion))
                    self.pub_optimization_suggestions.publish(suggestion_msg)
            else:
                # Dynamic mode: Log
                _log_info(self.node_name, f"Published optimization suggestion: {json.dumps(suggestion)}")
        except Exception as e:
            _log_error(self.node_name, f"Error publishing suggestions: {e}")

    def publish_status(self, status: Dict[str, Any]):
        """Publish normal status (ROS or log)."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_optimization_suggestions:
                if hasattr(OptimizationSuggestions, 'data'):
                    self.pub_optimization_suggestions.publish(String(data=json.dumps(status)))
                else:
                    status_msg = OptimizationSuggestions(data=json.dumps(status))
                    self.pub_optimization_suggestions.publish(status_msg)
            else:
                # Dynamic mode: Log
                _log_debug(self.node_name, f"Status: {json.dumps(status)}")
        except Exception as e:
            _log_error(self.node_name, f"Error publishing status: {e}")

    def _log_optimization_suggestion(self, suggestion: Dict[str, Any], sensory_snapshot: str):
        """Log optimization suggestion to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO optimization_log (id, timestamp, type, action, priority, reason, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(time.time()), suggestion['type'], suggestion['action'],
                suggestion['priority'], suggestion['reason'], sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log optimization suggestion: {e}")

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down OptimizationNode.")
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
                    self._check_and_publish()
                    time.sleep(self.check_interval)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Optimization Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = OptimizationNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate data
            node.monitoring_callback({'data': json.dumps({'status': 'alert', 'message': 'Low performance'})})
            node.memory_callback({'data': json.dumps({'total_entries': 95})})
            node.self_awareness_callback({'data': json.dumps({'coherence': 0.4})})
            time.sleep(2)
            print("Optimization simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
