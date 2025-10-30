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
    MonitoringOutput = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    MonitoringOutput = ROSMsgFallback


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
            'monitoring_node': {
                'alert_threshold': 0.3,
                'check_interval': 5.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate alerts (e.g., encouraging language)
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


class MonitoringNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'monitoring_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "monitoring_log.db")
        self.alert_threshold = self.params.get('alert_threshold', 0.3)
        self.check_interval = self.params.get('check_interval', 5.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing monitoring compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.metrics: Deque[Dict[str, Any]] = deque(maxlen=100)  # Bounded history for efficiency
        self.last_check = 0.0
        self.pending_metrics: Deque[Dict[str, Any]] = deque(maxlen=20)  # Queue for incoming metrics

        # Initialize SQLite database for monitoring logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                node_name TEXT,
                metric_json TEXT,
                status TEXT,
                alert_triggered BOOLEAN,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Monitoring Node online, observing with compassionate and mindful performance reflection.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_monitoring_output = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_monitoring_output = rospy.Publisher('monitoring_output', MonitoringOutput, queue_size=10)
            # Subscribers for metrics
            rospy.Subscriber('decision_making_output', String, self.decision_callback)
            rospy.Subscriber('learning_output', String, self.learning_callback)
            rospy.Subscriber('communication_output', String, self.communication_callback)
            rospy.Timer(rospy.Duration(self.check_interval), self.check_performance)
        else:
            # Dynamic mode: Start polling thread for simulated inputs
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing monitoring compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on metrics (e.g., high emotional sound boosts alert sensitivity)
            if sensor_type == 'vision':
                self.pending_metrics.append({'node': 'vision_monitor', 'metric': {'confidence': random.uniform(0.4, 0.9)}, 'timestamp': timestamp})
            elif sensor_type == 'sound':
                self.pending_metrics.append({'node': 'audio_monitor', 'metric': {'emotion_intensity': random.uniform(0.2, 0.6)}, 'timestamp': timestamp})
            elif sensor_type == 'instructions':
                self.pending_metrics.append({'node': 'command_monitor', 'metric': {'success_rate': random.uniform(0.5, 0.95)}, 'timestamp': timestamp})
            # Compassionate bias: If distress in sound, lower threshold for alerts compassionately
            if 'distress' in str(processed):
                self.alert_threshold = max(0.0, self.alert_threshold - self.ethical_compassion_bias * 0.1)
            _log_debug(self.node_name, f"{sensor_type} input updated monitoring context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            # Simulate periodic metrics
            self._simulate_metric('decision_making', {'confidence': random.uniform(0.3, 0.9)})
            self._simulate_metric('learning', {'accuracy': random.uniform(0.6, 0.95)})
            self._simulate_metric('communication', {'response_time': random.uniform(0.1, 1.0)})
            self.check_performance()
            time.sleep(self.check_interval)

    def _simulate_metric(self, node_name: str, metric: Dict[str, Any]):
        """Simulate a metric entry in non-ROS mode."""
        entry = {
            "node": node_name,
            "timestamp": time.time(),
            "metric": metric,
            "status": "simulated"
        }
        self.pending_metrics.append(entry)
        _log_debug(self.node_name, f"Simulated metric from {node_name}: {json.dumps(metric)}")

    # --- Core Monitoring Logic ---
    def decision_callback(self, msg: Any):
        """Handle incoming decision metrics."""
        fields_map = {'data': ('', 'decision_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        decision_data = json.loads(data.get('decision_data', '{}'))
        self.collect_metric("decision_making", decision_data)

    def learning_callback(self, msg: Any):
        """Handle incoming learning metrics."""
        fields_map = {'data': ('', 'learning_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        learning_data = json.loads(data.get('learning_data', '{}'))
        self.collect_metric("learning", learning_data)

    def communication_callback(self, msg: Any):
        """Handle incoming communication metrics."""
        fields_map = {'data': ('', 'communication_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        communication_data = json.loads(data.get('communication_data', '{}'))
        self.collect_metric("communication", communication_data)

    def collect_metric(self, node_name: str, metric: Dict[str, Any]) -> None:
        """Collect a metric with compassionate note for low performance."""
        try:
            timestamp = time.time()
            # Compassionate bias: If low confidence, add encouraging note
            compassionate_note = ""
            if metric.get("confidence", 1.0) < 0.5 and self.ethical_compassion_bias > 0.1:
                compassionate_note = f"Compassionate note: Encourage growth from this learning opportunity. Bias: {self.ethical_compassion_bias}."
            
            entry = {
                "node": node_name,
                "timestamp": timestamp,
                "metric": metric,
                "status": "ok",
                "compassionate_note": compassionate_note
            }
            self.pending_metrics.append(entry)
            _log_info(self.node_name, f"Collected metric from {node_name}: {json.dumps(metric)}")
        except Exception as e:
            _log_error(self.node_name, f"Error collecting metric: {e}")

    def check_performance(self):
        """Check overall performance and trigger alerts compassionately."""
        if not self.pending_metrics or (time.time() - self.last_check < self.check_interval):
            return

        self.last_check = time.time()
        metrics = list(self.pending_metrics)
        self.pending_metrics.clear()
        self.metrics.extend(metrics)

        total_metrics = len(self.metrics)
        if total_metrics == 0:
            return

        # Calculate average confidence as proxy for performance
        avg_performance = sum(m["metric"].get("confidence", 0) for m in self.metrics[-10:]) / min(10, total_metrics)  # Recent average
        _log_info(self.node_name, f"Average performance: {avg_performance:.2f}")

        # Log metrics to DB with sensory snapshot
        sensory_snapshot = json.dumps(self.sensory_data)
        self._log_metrics_batch(metrics, sensory_snapshot)

        if avg_performance < self.alert_threshold:
            alert = {
                "status": "alert",
                "message": f"Performance below threshold ({avg_performance:.2f} < {self.alert_threshold})",
                "recommendation": "retrain_or_restart",
                "compassionate_note": f"With compassion, view this as a growth opportunity. Bias: {self.ethical_compassion_bias}."
            }
            self.publish_alert(alert)
            _log_warn(self.node_name, f"{alert['message']} - {alert.get('compassionate_note', '')}")
        else:
            status = {"status": "normal", "alert": False}
            self.publish_status(status)
            _log_debug(self.node_name, "Performance normal.")

    def _log_metrics_batch(self, metrics: List[Dict[str, Any]], sensory_snapshot: str):
        """Log a batch of metrics to DB."""
        try:
            self.cursor.executemany('''
                INSERT INTO monitoring_log (id, timestamp, node_name, metric_json, status, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', [(str(uuid.uuid4()), m['timestamp'], m['node'], json.dumps(m['metric']), m['status'], sensory_snapshot) for m in metrics])
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log metrics batch: {e}")

    def publish_alert(self, alert: Dict[str, Any]):
        """Publish an alert (ROS or log)."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_monitoring_output:
                if hasattr(MonitoringOutput, 'data'):
                    self.pub_monitoring_output.publish(String(data=json.dumps(alert)))
                else:
                    alert_msg = MonitoringOutput(data=json.dumps(alert))
                    self.pub_monitoring_output.publish(alert_msg)
            else:
                # Dynamic mode: Log
                _log_warn(self.node_name, f"Alert: {json.dumps(alert)}")
        except Exception as e:
            _log_error(self.node_name, f"Failed to publish alert: {e}")

    def publish_status(self, status: Dict[str, Any]):
        """Publish normal status (ROS or log)."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_monitoring_output:
                if hasattr(MonitoringOutput, 'data'):
                    self.pub_monitoring_output.publish(String(data=json.dumps(status)))
                else:
                    status_msg = MonitoringOutput(data=json.dumps(status))
                    self.pub_monitoring_output.publish(status_msg)
            else:
                # Dynamic mode: Log
                _log_debug(self.node_name, f"Status: {json.dumps(status)}")
        except Exception as e:
            _log_error(self.node_name, f"Failed to publish status: {e}")

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down MonitoringNode.")
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
                    time.sleep(1)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Monitoring Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = MonitoringNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate metrics
            node.collect_metric("decision_making", {"confidence": 0.8})
            node.collect_metric("learning", {"accuracy": 0.9})
            node.collect_metric("communication", {"response_time": 0.2})
            time.sleep(2)
            print("Monitoring simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
