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
    SelfAwarenessStatus = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    SelfAwarenessStatus = ROSMsgFallback


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
            'self_awareness_node': {
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate self-awareness (e.g., self-care on low coherence)
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


class SelfAwarenessNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'self_awareness_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "self_awareness_log.db")
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing self-awareness compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.internal_state = {"coherence": 0.8, "uptime": 0.0, "anomalies": []}
        self.pending_data: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for data
        self.awareness_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns
        self.start_time = time.time()

        # Initialize SQLite database for self-awareness logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS self_awareness_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                coherence REAL,
                uptime REAL,
                anomalies_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Self-Awareness Node initialized, reflecting with compassionate inner observation.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_self_awareness_status = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_self_awareness_status = rospy.Publisher('self_awareness_status', SelfAwarenessStatus, queue_size=10)
            rospy.Subscriber('integration_output', String, self.integration_callback)
            rospy.Subscriber('monitoring_output', String, self.monitoring_callback)
            rospy.Timer(rospy.Duration(2.0), self.update_status)  # Periodic update
        else:
            # Dynamic mode: Start polling thread for simulated data
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing self-awareness compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on self-awareness
            if sensor_type == 'vision':
                self.pending_data.append({'type': 'integration', 'data': {'confidence': random.uniform(0.5, 0.9)}})
            elif sensor_type == 'sound':
                self.pending_data.append({'type': 'monitoring', 'data': {'status': 'alert' if random.random() < 0.3 else 'normal'}})
            elif sensor_type == 'instructions':
                self.pending_data.append({'type': 'integration', 'data': {'coherence': random.uniform(0.4, 0.8)}})
            # Compassionate bias: If distress in sound, boost coherence check compassionately
            if 'distress' in str(processed):
                self.internal_state['coherence'] = max(0.0, self.internal_state['coherence'] - self.ethical_compassion_bias * 0.1)
            _log_debug(self.node_name, f"{sensor_type} input updated self-awareness context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            # Simulate periodic data
            self._simulate_integration_data()
            self._simulate_monitoring_data()
            self.update_status()
            time.sleep(2.0)

    def _simulate_integration_data(self):
        """Simulate integration data in non-ROS mode."""
        integrated_data = {'components': {'decision': {'confidence': random.uniform(0.3, 0.9)}}}
        self.integration_callback({'data': json.dumps(integrated_data)})
        _log_debug(self.node_name, f"Simulated integration data: {json.dumps(integrated_data)}")

    def _simulate_monitoring_data(self):
        """Simulate monitoring data in non-ROS mode."""
        monitoring_data = {'status': 'normal' if random.random() > 0.2 else 'alert'}
        self.monitoring_callback({'data': json.dumps(monitoring_data)})
        _log_debug(self.node_name, f"Simulated monitoring data: {json.dumps(monitoring_data)}")

    # --- Core Self-Awareness Logic ---
    def integration_callback(self, msg: Any):
        """Handle incoming integrated data for self-assessment."""
        fields_map = {'data': ('', 'integrated_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        integrated_data = json.loads(data.get('integrated_data', '{}'))
        self.assess_coherence(integrated_data)

    def monitoring_callback(self, msg: Any):
        """Handle incoming monitoring data for self-assessment."""
        fields_map = {'data': ('', 'monitoring_data')}
        data = parse_message_data(msg, self.node_name)
        monitoring_data = json.loads(data.get('monitoring_data', '{}'))
        self.detect_anomalies(monitoring_data)

    def assess_coherence(self, integrated_data: Dict[str, Any]) -> None:
        """Assess the system's coherence based on integrated outputs with compassionate bias."""
        try:
            confidence_sum = sum(d.get("confidence", 0) for d in integrated_data.get("components", {}).values() if isinstance(d, dict))
            component_count = len([d for d in integrated_data.get("components", {}).values() if isinstance(d, dict)])
            coherence = confidence_sum / component_count if component_count > 0 else 0.5
            # Compassionate bias: If low coherence, gently adjust toward self-reflection
            if coherence < 0.5 and self.ethical_compassion_bias > 0.1:
                coherence = max(0.0, coherence + self.ethical_compassion_bias * 0.05)  # Slight boost for self-care
            self.internal_state["coherence"] = max(0.0, min(1.0, coherence))
            _log_info(self.node_name, f"Assessed coherence: {self.internal_state['coherence']:.2f}")
            self.update_status()
        except Exception as e:
            _log_error(self.node_name, f"Error assessing coherence: {e}")

    def detect_anomalies(self, monitoring_data: Dict[str, Any]) -> None:
        """Detect anomalies based on monitoring data with compassionate note."""
        try:
            if monitoring_data.get("status") == "alert":
                anomaly = {
                    "timestamp": time.time(),
                    "message": monitoring_data.get("message", "Unknown anomaly"),
                    "severity": "high",
                    "compassionate_note": f"View this anomaly as a growth opportunity. Bias: {self.ethical_compassion_bias}." if self.ethical_compassion_bias > 0.1 else ""
                }
                self.internal_state["anomalies"].append(anomaly)
                _log_warn(self.node_name, f"Detected anomaly: {json.dumps(anomaly)}")
            self.update_status()
        except Exception as e:
            _log_error(self.node_name, f"Error detecting anomalies: {e}")

    def update_status(self):
        """Update and publish the internal state."""
        try:
            self.internal_state["uptime"] = time.time() - self.start_time
            status = self.internal_state.copy()
            # Log to DB with sensory snapshot
            sensory_snapshot = json.dumps(self.sensory_data)
            self._log_self_awareness_state(status, sensory_snapshot)

            if ROS_AVAILABLE and self.ros_enabled and self.pub_self_awareness_status:
                if hasattr(SelfAwarenessStatus, 'data'):
                    self.pub_self_awareness_status.publish(String(data=json.dumps(status)))
                else:
                    status_msg = SelfAwarenessStatus(data=json.dumps(status))
                    self.pub_self_awareness_status.publish(status_msg)
            else:
                # Dynamic mode: Log
                _log_info(self.node_name, f"Published self-awareness status: coherence {status['coherence']:.2f}, anomalies {len(status['anomalies'])}")
        except Exception as e:
            _log_error(self.node_name, f"Error updating status: {e}")

    def _log_self_awareness_state(self, status: Dict[str, Any], sensory_snapshot: str):
        """Log self-awareness state to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO self_awareness_log (id, timestamp, coherence, uptime, anomalies_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), status['coherence'],
                status['uptime'], json.dumps(status['anomalies']), sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log self-awareness state: {e}")

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down SelfAwarenessNode.")
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
                    self._simulate_integration_data()
                    self._simulate_monitoring_data()
                    self.update_status()
                    time.sleep(2.0)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Self-Awareness Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = SelfAwarenessNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate data
            node.integration_callback({'data': json.dumps({'components': {'decision': {'confidence': 0.7}}})})
            node.monitoring_callback({'data': json.dumps({'status': 'alert', 'message': 'Low coherence detected'})})
            time.sleep(2)
            print("Self-awareness simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
