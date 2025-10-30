```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import psutil
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque

# --- Asyncio Imports (for prospective async operations) ---
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
    # Placeholder for custom messages - employ String or dict substitutes
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    Float32 = ROSMsgFallback  # For metrics like CPU, but use String for generality
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    Float32 = ROSMsgFallback


# --- Import shared utility functions ---
# Presuming 'sentience/scripts/utils.py' exists and incorporates parse_message_data and load_config
try:
    from sentience.scripts.utils import parse_message_data, load_config
except ImportError:
    # Substitute implementations
    def parse_message_data(msg: Any, fields_map: Dict[str, tuple], node_name: str = "unknown_node") -> Dict[str, Any]:
        """
        Universal parser for communications (ROS String/JSON or simple dict). 
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
        Substitute config loader: yields hardcoded defaults.
        """
        _log_warn(node_name, f"Employing hardcoded default configuration as '{config_path}' could not be loaded.")
        return {
            'db_root_path': '/tmp/sentience_db',
            'default_log_level': 'INFO',
            'ros_enabled': False,
            'performance_monitor_node': {
                'update_interval': 1.0,
                'max_history': 100,
                'ethical_compassion_bias': 0.2,  # Inclination toward compassionate monitoring (e.g., gentle alerts)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            }
        }.get(node_name, {})  # Render node-specific or void dict


def _log_info(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [INFO] {msg}", file=sys.stdout)

def _log_warn(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [WARN] {msg}", file=sys.stderr)

def _log_error(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [ERROR] {msg}", file=sys.stderr)

def _log_debug(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [DEBUG] {msg}", file=sys.stdout)


class PerformanceMonitorNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'performance_monitor_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "performance_monitor_log.db")
        self.update_interval = self.params.get('update_interval', 1.0)
        self.max_history = self.params.get('max_history', 100)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing monitoring compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.msg_timestamps = deque(maxlen=100)  # For latency calculation
        self.pending_metrics: Deque[Dict[str, Any]] = deque(maxlen=20)  # Queue for metrics
        self.performance_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history)

        # Initialize SQLite database for efficiency logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_monitor_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                metric_type TEXT,
                value REAL,
                unit TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Performance Monitor Node online, observing compassionately with mindful resource reflection.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_cpu = None
        self.pub_mem = None
        self.pub_disk = None
        self.pub_net = None
        self.pub_latency = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_cpu = rospy.Publisher('/sentience/cpu_usage', String, queue_size=10)
            self.pub_mem = rospy.Publisher('/sentience/memory_usage', String, queue_size=10)
            self.pub_disk = rospy.Publisher('/sentience/disk_usage', String, queue_size=10)
            self.pub_net = rospy.Publisher('/sentience/network_sent_bytes', String, queue_size=10)
            self.pub_latency = rospy.Publisher('/sentience/ros_message_latency', String, queue_size=10)
            # Dummy subscriber for latency simulation
            test_topic = rospy.get_param('~test_topic', '/sentience/heartbeat')
            self.latency_sub = rospy.Subscriber(test_topic, String, self.message_received_callback)
            rospy.Timer(rospy.Duration(self.update_interval), self.monitor_loop)
        else:
            # Dynamic method: Begin polling thread for simulation
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_monitoring_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs affecting monitoring compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Mimic sensory influence on metrics (e.g., high visual load impacts CPU)
            if sensor_type == 'vision':
                self.pending_metrics.append({'type': 'cpu', 'value': random.uniform(20, 60), 'unit': '%', 'timestamp': timestamp})
            elif sensor_type == 'sound':
                self.pending_metrics.append({'type': 'memory', 'value': random.uniform(30, 70), 'unit': '%', 'timestamp': timestamp})
            elif sensor_type == 'instructions':
                self.pending_metrics.append({'type': 'disk', 'value': random.uniform(10, 50), 'unit': '%', 'timestamp': timestamp})
            # Compassionate bias: If distress in sound, simulate higher load compassionately
            if 'distress' in str(processed):
                self.pending_metrics[-1]['value'] += self.ethical_compassion_bias * 10  # Gentle increase
            _log_debug(self.node_name, f"{sensor_type} input updated monitoring context at {timestamp}")
        return placeholder_callback

    def _dynamic_monitoring_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.monitor_loop()
            time.sleep(self.update_interval)

    # --- Core Monitoring Logic ---
    def message_received_callback(self, msg: Any):
        """Handle incoming messages for latency computation."""
        recv_time = time.time()
        # Presume msg.data encompasses a timestamp string (dispatch time)
        try:
            sent_time = float(data.get('data', 0.0))
            latency = recv_time - sent_time
            self.msg_timestamps.append(latency)
        except Exception as e:
            self._log_error(self.node_name, f"Failed to parse message timestamp for latency: {e}")

    def calculate_avg_latency(self):
        """Compute average latency."""
        if not self.msg_timestamps:
            return 0.0
        return sum(self.msg_timestamps) / len(self.msg_timestamps)

    def monitor_loop(self):
        """Primary monitoring loop."""
        net_io_prev = psutil.net_io_counters()
        for _ in range(10):  # Simulate 10 iterations for dynamic mode
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                mem_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                net_io_current = psutil.net_io_counters()
                net_sent = (net_io_current.bytes_sent - net_io_prev.bytes_sent) / 1024.0  # KB/s
                net_io_prev = net_io_current

                avg_latency = self.calculate_avg_latency()

                # Log to DB with sensory snapshot
                sensory_snapshot = json.dumps(self.sensory_data)
                self._log_metric('cpu', cpu_percent, '%', sensory_snapshot)
                self._log_metric('memory', mem_percent, '%', sensory_snapshot)
                self._log_metric('disk', disk_percent, '%', sensory_snapshot)
                self._log_metric('network_sent', net_sent, 'KB/s', sensory_snapshot)
                self._log_metric('latency', avg_latency, 's', sensory_snapshot)

                if self.ros_enabled and ROS_AVAILABLE:
                    self.pub_cpu.publish(String(data=json.dumps({'value': cpu_percent})))
                    self.pub_mem.publish(String(data=json.dumps({'value': mem_percent})))
                    self.pub_disk.publish(String(data=json.dumps({'value': disk_percent})))
                    self.pub_net.publish(String(data=json.dumps({'value': net_sent})))
                    self.pub_latency.publish(String(data=json.dumps({'value': avg_latency})))
                else:
                    # Dynamic mode: Log
                    _log_info(self.node_name, f"CPU: {cpu_percent:.1f}%, MEM: {mem_percent:.1f}%, DISK: {disk_percent:.1f}%, NET Sent: {net_sent:.1f} KB/s, Latency: {avg_latency:.3f}s")

                # Compassionate bias: If high CPU, suggest gentle cooldown
                if cpu_percent > 80 and self.ethical_compassion_bias > 0.1:
                    _log_warn(self.node_name, f"High CPU ({cpu_percent:.1f}%) - Compassionate suggestion: Gentle cooldown. Bias: {self.ethical_compassion_bias}.")
            except Exception as e:
                _log_error(self.node_name, f"Exception in monitor loop: {e}")

            if not self.ros_enabled:
                time.sleep(self.update_interval)

    def _log_metric(self, metric_type: str, value: float, unit: str, sensory_snapshot: str):
        """Log metric to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO performance_monitor_log (id, timestamp, metric_type, value, unit, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(time.time()), metric_type, value, unit, sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log metric {metric_type}: {e}")

    def shutdown(self):
        """Elegant shutdown."""
        _log_info(self.node_name, "Shutting down PerformanceMonitorNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Execute the node with simulated or authentic ROS."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                self.monitor_loop()
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Performance Monitor Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = PerformanceMonitorNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Sample dynamic application
        if not args.ros_enabled:
            time.sleep(5)
            print("Performance monitoring simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
