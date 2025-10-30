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

# --- Asyncio Imports (for possible upcoming asynchronous operations) ---
import asyncio
import threading
from collections import deque

# Simulated ROS Integration (for compatibility)
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
    MetaAwarenessState = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    MetaAwarenessState = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback


# --- Import shared utility functions ---
# Assuming 'sentience/scripts/utils.py' exists and has parse_message_data and load_config
try:
    from sentience.scripts.utils import parse_message_data, load_config
except ImportError:
    # Fallback implementations
    def parse_message_data(msg: Any, fields_map: Dict[str, tuple], node_name: str = "unknown_node") -> Dict[str, Any]:
        """
        Common parser for messages (simulated ROS String/JSON or plain dict). 
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
            'meta_awareness_node': {
                'confidence_threshold': 0.4,
                'conflict_threshold': 0.6,
                'update_interval': 3.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate meta-awareness (e.g., self-reflection on empathy)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            }
        }.get(node_name, {})  # Return node-specific or vacant dict


def _log_info(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [INFO] {msg}", file=sys.stdout)

def _log_warn(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [WARN] {msg}", file=sys.stderr)

def _log_error(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [ERROR] {msg}", file=sys.stderr)

def _log_debug(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [DEBUG] {msg}", file=sys.stdout)


class MetaAwarenessNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'meta_awareness_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "meta_awareness_log.db")
        self.confidence_threshold = self.params.get('confidence_threshold', 0.4)
        self.conflict_threshold = self.params.get('conflict_threshold', 0.6)
        self.update_interval = self.params.get('update_interval', 3.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing meta-awareness compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.last_confidence = 1.0
        self.dissonance_level = 0.0
        self.last_emotion = {'mood': 'neutral', 'intensity': 0.0}
        self.pending_updates: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for updates
        self.meta_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for meta-awareness logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS meta_awareness_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                confidence_level REAL,
                dissonance REAL,
                emotional_intensity REAL,
                mood TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Meta-Awareness Node online, observing inner confidence with compassionate self-reflection.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_meta_state = None
        self.pub_directive = None
        self.sub_prediction = None
        self.sub_narrative = None
        self.sub_emotion = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_meta_state = rospy.Publisher('/meta_awareness_state', MetaAwarenessState, queue_size=10)
            self.pub_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)
            self.sub_prediction = rospy.Subscriber('/prediction_state', String, self.prediction_callback)
            self.sub_narrative = rospy.Subscriber('/internal_narrative', String, self.narrative_callback)
            self.sub_emotion = rospy.Subscriber('/emotion_state', String, self.emotion_callback)
            rospy.Timer(rospy.Duration(self.update_interval), self.evaluate_meta_state)
        else:
            # Simulated mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._simulated_run_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing meta-awareness compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on meta-awareness (e.g., high salience sound boosts dissonance)
            if sensor_type == 'vision':
                self.pending_updates.append({'type': 'prediction', 'data': {'confidence': random.uniform(0.3, 0.8)}})
            elif sensor_type == 'sound':
                self.pending_updates.append({'type': 'narrative', 'data': {'main_theme': 'conflict' if random.random() < 0.3 else 'reflection'}})
            elif sensor_type == 'instructions':
                self.pending_updates.append({'type': 'emotion', 'data': {'mood': 'anxious' if random.random() < 0.2 else 'calm', 'intensity': random.uniform(0.1, 0.6)}})
            # Compassionate bias: If distress in sound, increase dissonance compassionately
            if 'distress' in str(processed):
                self.dissonance_level = min(1.0, self.dissonance_level + self.ethical_compassion_bias)
            _log_debug(self.node_name, f"{sensor_type} input updated meta-awareness context at {timestamp}")
        return placeholder_callback

    def _simulated_run_loop(self):
        """Simulated polling loop in non-ROS mode."""
        while not self._shutdown_flag.is_set():
            self.evaluate_meta_state(None)
            time.sleep(self.update_interval)

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    # --- Core Meta-Awareness Logic ---
    def prediction_callback(self, msg: Any):
        """Handle incoming prediction state data."""
        fields_map = {'data': ('', 'prediction_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        prediction_data = json.loads(data.get('prediction_data', '{}'))
        confidence = prediction_data.get("confidence", 1.0)
        accurate = prediction_data.get("is_accurate", True)

        if not accurate:
            delta = self.last_confidence - confidence
            self.dissonance_level += abs(delta)
        self.last_confidence = confidence
        _log_debug(self.node_name, f"Updated confidence to {confidence:.2f} (Accuracy: {accurate}).")

    def narrative_callback(self, msg: Any):
        """Handle incoming internal narrative data."""
        fields_map = {'data': ('', 'narrative_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        narrative_data = json.loads(data.get('narrative_data', '{}'))
        if "conflict" in narrative_data.get("main_theme", "").lower():
            self.dissonance_level += 0.2
        _log_debug(self.node_name, f"Updated dissonance due to narrative theme: {narrative_data.get('main_theme', 'N/A')}.")

    def emotion_callback(self, msg: Any):
        """Handle incoming emotion state data."""
        fields_map = {'data': ('', 'emotion_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        emotion_data = json.loads(data.get('emotion_data', '{}'))
        self.last_emotion = {'mood': emotion_data.get('mood', 'neutral'), 'intensity': emotion_data.get('mood_intensity', 0.0)}
        # Compassionate bias: High emotional intensity boosts dissonance for self-reflection
        if self.last_emotion['intensity'] > 0.7:
            self.dissonance_level = min(1.0, self.dissonance_level + self.ethical_compassion_bias)
        _log_debug(self.node_name, f"Updated emotion to {self.last_emotion['mood']} (Intensity: {self.last_emotion['intensity']:.2f}).")

    def evaluate_meta_state(self, event: Any = None):
        """Evaluate and publish the meta-awareness state."""
        meta_status = {
            'confidence_level': self.last_confidence,
            'dissonance': self.dissonance_level,
            'emotional_intensity': self.last_emotion['intensity'],
            'mood': self.last_emotion['mood'],
            'timestamp': time.time()
        }

        # Log to DB with sensory snapshot
        sensory_snapshot = json.dumps(self.sensory_data)
        self._log_meta_state(meta_status, sensory_snapshot)

        if ROS_AVAILABLE and self.ros_enabled and self.pub_meta_state:
            if hasattr(MetaAwarenessState, 'data'):
                self.pub_meta_state.publish(String(data=json.dumps(meta_status)))
            else:
                meta_msg = MetaAwarenessState(data=json.dumps(meta_status))
                self.pub_meta_state.publish(meta_msg)
        else:
            # Dynamic mode: Log
            _log_info(self.node_name, f"Meta-awareness snapshot: {json.dumps(meta_status)}")

        # Trigger reflection if thresholds exceeded
        if self.last_confidence < self.confidence_threshold or self.dissonance_level > self.conflict_threshold:
            self._trigger_self_reflection(meta_status)
            _log_warn(self.node_name, f"Triggering self-reflection due to low confidence ({self.last_confidence:.2f}) or high dissonance ({self.dissonance_level:.2f}).")

        # Decay dissonance
        self.dissonance_level *= 0.8
        self.dissonance_level = max(0.0, self.dissonance_level)

    def _trigger_self_reflection(self, meta_status: Dict[str, Any]):
        """Trigger self-reflection directive with compassionate note."""
        directive = {
            'directive_type': 'TriggerReflection',
            'target_node': '/reflection_node',
            'command_payload': json.dumps({'reason': 'low_confidence_or_conflict', 'meta_context': meta_status}),
            'timestamp': time.time()
        }
        # Compassionate bias: Add note if high emotional intensity
        if meta_status['emotional_intensity'] > 0.6:
            directive['compassionate_note'] = "Prioritizing empathetic self-reflection due to elevated emotions."

        if ROS_AVAILABLE and self.ros_enabled and self.pub_directive:
            if hasattr(CognitiveDirective, 'data'):
                self.pub_directive.publish(String(data=json.dumps(directive)))
            else:
                directive_msg = CognitiveDirective(data=json.dumps(directive))
                self.pub_directive.publish(directive_msg)
        else:
            # Dynamic mode: Log
            _log_info(self.node_name, f"Simulated reflection directive: {json.dumps(directive)}")

    def _log_meta_state(self, meta_status: Dict[str, Any], sensory_snapshot: str):
        """Log meta-awareness state to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO meta_awareness_log (id, timestamp, confidence_level, dissonance, emotional_intensity, mood, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), meta_status['confidence_level'],
                meta_status['dissonance'], meta_status['emotional_intensity'], meta_status['mood'], sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log meta-awareness state: {e}")

    # --- Polling Mode for Simulated Inputs ---
    def _simulate_inputs(self):
        """Simulate inputs in non-ROS mode to mimic ROS callbacks."""
        # Simulate prediction
        self.prediction_callback({'data': json.dumps({'confidence': random.uniform(0.3, 0.9), 'is_accurate': random.choice([True, False])})})
        # Simulate narrative
        self.narrative_callback({'data': json.dumps({'main_theme': random.choice(['reflection', 'conflict', 'harmony'])})})
        # Simulate emotion
        self.emotion_callback({'data': json.dumps({'mood': random.choice(['neutral', 'anxious', 'content']), 'mood_intensity': random.uniform(0.2, 0.8)})})

    # --- Public Methods ---
    def get_meta_state(self) -> Dict[str, Any]:
        """Get the latest meta-awareness state."""
        return {
            'confidence_level': self.last_confidence,
            'dissonance': self.dissonance_level,
            'emotional_intensity': self.last_emotion['intensity'],
            'mood': self.last_emotion['mood'],
            'timestamp': time.time()
        }

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down MetaAwarenessNode.")
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
                    self._simulate_inputs()
                    self.evaluate_meta_state(None)
                    time.sleep(self.update_interval)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Meta-Awareness Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = MetaAwarenessNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage if not ROS
        if not args.ros_enabled:
            # Simulate a loop for 10 seconds
            import time
            start_time = time.time()
            while time.time() - start_time < 10:
                node._simulate_inputs()
                node.evaluate_meta_state(None)
                time.sleep(node.update_interval)
            print("Meta-awareness simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
