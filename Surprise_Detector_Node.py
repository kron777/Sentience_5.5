```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List

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
    SurpriseEvent = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    SurpriseEvent = ROSMsgFallback


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
            'surprise_detector_node': {
                'threshold': 0.7,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate surprise handling (e.g., lower threshold for self-reflection)
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


class SurpriseDetectorNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'surprise_detector_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "surprise_log.db")
        self.threshold = self.params.get('threshold', 0.7)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing surprise compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.surprise_history: Deque[float] = deque(maxlen=100)  # Bounded history
        self.pending_predictions: Deque[Dict[str, Any]] = deque(maxlen=20)  # Queue for predictions/actuals
        self.anomaly_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for anomalies

        # Initialize SQLite database for surprise logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS surprise_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                surprise_score REAL,
                is_surprise BOOLEAN,
                predicted_value REAL,
                actual_value REAL,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Surprise Detector Node online, detecting with compassionate and mindful anomaly reflection.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_surprise_event = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_surprise_event = rospy.Publisher('/surprise_event', SurpriseEvent, queue_size=10)
            rospy.Subscriber('/prediction_output', String, self.prediction_callback)
            rospy.Subscriber('/actual_outcome', String, self.actual_callback)
            rospy.Timer(rospy.Duration(1.0), self._periodic_check_and_publish)  # Periodic check
        else:
            # Dynamic mode: Start polling thread for simulated data
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing surprise compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on predictions/actuals (e.g., anomaly in sound -> high surprise)
            if sensor_type == 'vision':
                self.pending_predictions.append({'type': 'prediction', 'data': {'predicted': random.uniform(0.5, 0.9)}})
                self.pending_predictions.append({'type': 'actual', 'data': {'actual': random.uniform(0.3, 0.8)}})
            elif sensor_type == 'sound':
                self.pending_predictions.append({'type': 'prediction', 'data': {'predicted': 0.4}})
                self.pending_predictions.append({'type': 'actual', 'data': {'actual': 0.9}})  # High surprise
            elif sensor_type == 'instructions':
                self.pending_predictions.append({'type': 'prediction', 'data': {'predicted': 0.6}})
                self.pending_predictions.append({'type': 'actual', 'data': {'actual': 0.6}})
            # Compassionate bias: If distress in sound, lower effective surprise for self-compassion
            if 'distress' in str(processed):
                self.ethical_compassion_bias = min(1.0, self.ethical_compassion_bias + 0.1)
            _log_debug(self.node_name, f"{sensor_type} input updated surprise context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._simulate_prediction_and_actual()
            self._periodic_check_and_publish()
            time.sleep(1.0)

    def _simulate_prediction_and_actual(self):
        """Simulate prediction and actual in non-ROS mode."""
        predicted = random.uniform(0.3, 0.9)
        actual = random.uniform(0.2, 1.0)
        self.pending_predictions.append({'type': 'prediction', 'data': {'predicted': predicted}})
        self.pending_predictions.append({'type': 'actual', 'data': {'actual': actual}})
        _log_debug(self.node_name, f"Simulated: predicted {predicted:.2f}, actual {actual:.2f}")

    # --- Core Surprise Detection Logic ---
    def prediction_callback(self, msg: Any):
        """Handle incoming prediction data."""
        fields_map = {'data': ('', 'prediction_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        prediction_data = json.loads(data.get('prediction_data', '{}'))
        predicted = prediction_data.get('predicted', 0.5)
        self.pending_predictions.append({'type': 'prediction', 'data': {'predicted': predicted}})

    def actual_callback(self, msg: Any):
        """Handle incoming actual outcome data."""
        fields_map = {'data': ('', 'actual_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        actual_data = json.loads(data.get('actual_data', '{}'))
        actual = actual_data.get('actual', 0.5)
        self.pending_predictions.append({'type': 'actual', 'data': {'actual': actual}})
        # Trigger computation if both available
        self._compute_surprise_if_ready()

    def compute_surprise(self, predicted: float, actual: float) -> float:
        """Compute surprise as normalized absolute error between predicted and actual with compassionate bias."""
        error = abs(predicted - actual)
        surprise = error  # Direct error as surprise metric
        # Compassionate bias: Gently lower surprise if close to prediction for self-compassion
        if surprise < 0.2 and self.ethical_compassion_bias > 0.1:
            surprise = max(0.0, surprise - self.ethical_compassion_bias * 0.05)
        self.surprise_history.append(surprise)
        _log_debug(self.node_name, f"Computed surprise: {surprise:.3f} (predicted: {predicted}, actual: {actual})")
        return surprise

    def is_surprise(self, surprise_score: float) -> bool:
        """Check if surprise exceeds threshold with compassionate adjustment."""
        # Compassionate bias: Slightly lower effective threshold for self-compassion
        effective_threshold = self.threshold - (self.ethical_compassion_bias * 0.05)
        return surprise_score >= effective_threshold

    def recent_surprises(self, window: Optional[int] = 10) -> List[float]:
        """Get recent surprises."""
        return list(self.surprise_history)[-window:]

    def summary(self) -> Dict[str, Any]:
        """Get summary with compassionate insights."""
        count_surprises = sum(1 for s in self.surprise_history if self.is_surprise(s))
        total = len(self.surprise_history)
        compassionate_insight = f"Compassionate reflection: {count_surprises} surprises as growth chances. Bias: {self.ethical_compassion_bias}." if count_surprises > 0 else ""
        return {
            "total_events": total,
            "surprises_detected": count_surprises,
            "surprise_rate": count_surprises / total if total > 0 else 0,
            "compassionate_insight": compassionate_insight
        }

    def _compute_surprise_if_ready(self):
        """Compute surprise if both prediction and actual are available."""
        if len(self.pending_predictions) >= 2 and self.pending_predictions[0]['type'] == 'prediction' and self.pending_predictions[1]['type'] == 'actual':
            predicted_data = self.pending_predictions[0]['data']
            actual_data = self.pending_predictions[1]['data']
            surprise = self.compute_surprise(predicted_data['predicted'], actual_data['actual'])
            is_surp = self.is_surprise(surprise)
            self._log_surprise(surprise, is_surp, json.dumps(self.sensory_data))
            self.publish_surprise_event(surprise, is_surp)
            # Clear processed
            self.pending_predictions.popleft()
            self.pending_predictions.popleft()

    def _log_surprise(self, surprise_score: float, is_surprise: bool, sensory_snapshot: str):
        """Log surprise to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO surprise_log (id, timestamp, surprise_score, is_surprise, predicted_value, actual_value, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), surprise_score, is_surprise, 0.0, 0.0, sensory_snapshot  # Add predicted/actual if available
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log surprise: {e}")

    def publish_surprise_event(self, surprise_score: float, is_surprise: bool):
        """Publish surprise event (ROS or log)."""
        event = {
            'timestamp': self._get_current_time(),
            'surprise_score': surprise_score,
            'is_surprise': is_surprise
        }
        if ROS_AVAILABLE and self.ros_enabled and self.pub_surprise_event:
            if hasattr(SurpriseEvent, 'data'):
                self.pub_surprise_event.publish(String(data=json.dumps(event)))
            else:
                event_msg = SurpriseEvent(data=json.dumps(event))
                self.pub_surprise_event.publish(event_msg)
        else:
            _log_info(self.node_name, f"Surprise event: score {surprise_score:.3f}, is_surprise: {is_surprise}")

    def _periodic_check_and_publish(self):
        """Periodic check and publishing of recent surprises."""
        if self.surprise_history:
            recent = self.recent_surprises(10)
            summary = self.summary()
            _log_info(self.node_name, f"Recent surprises: {json.dumps(recent)}, Summary: {json.dumps(summary)}")

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down SurpriseDetectorNode.")
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
                    self._simulate_prediction_and_actual()
                    self._periodic_check_and_publish()
                    time.sleep(1.0)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Surprise Detector Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros_enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = SurpriseDetectorNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate a prediction and actual
            node.prediction_callback({'data': json.dumps({'predicted': 0.8})})
            node.actual_callback({'data': json.dumps({'actual': 0.2})})
            time.sleep(1)
            print("Surprise detection simulation complete.")
            print(node.summary())
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
