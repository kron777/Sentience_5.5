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
    DecisionOutput = ROSMsgFallback
    CognitionInput = ROSMsgFallback
    EmotionInput = ROSMsgFallback
    HardwareStatus = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    DecisionOutput = ROSMsgFallback
    CognitionInput = ROSMsgFallback
    EmotionInput = ROSMsgFallback
    HardwareStatus = ROSMsgFallback


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
            'decision_making_node': {
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate decisions
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


class DecisionMakingNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'decision_making_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "decision_log.db")
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing decisions compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.cognition_input: Optional[Dict[str, Any]] = None
        self.emotion_input: Optional[Dict[str, Any]] = None
        self.hardware_status: Optional[Dict[str, Any]] = None
        self.pending_inputs: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for inputs
        self.decision_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for decision logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS decision_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                action TEXT,
                priority TEXT,
                confidence_score REAL,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Decision Making Node online, deciding with compassionate and mindful discernment.")

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_decision_output = None
        self.sub_cognition = None
        self.sub_emotion = None
        self.sub_hardware = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_decision_output = rospy.Publisher('decision_output', DecisionOutput, queue_size=10)
            self.sub_cognition = rospy.Subscriber('cognition_input', CognitionInput, self.cognition_callback)
            self.sub_emotion = rospy.Subscriber('emotion_input', EmotionInput, self.emotion_callback)
            self.sub_hardware = rospy.Subscriber('hardware_status', HardwareStatus, self.hardware_callback)

            # Sensory subscribers
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(0.5), self.process_pending_inputs)
        else:
            # Dynamic mode: Polling loop
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing decisions compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on decisions
            if sensor_type == 'vision':
                self.pending_inputs.append({'type': 'cognition', 'data': {'confidence': random.uniform(0.4, 0.8), 'task': 'analyze_scene'}})
            elif sensor_type == 'sound':
                self.pending_inputs.append({'type': 'emotion', 'data': {'intensity': random.uniform(0.2, 0.6), 'type': 'calm' if random.random() < 0.7 else 'distress'}})
            elif sensor_type == 'instructions':
                self.pending_inputs.append({'type': 'hardware', 'data': {'available': True, 'status': 'ready'}})
            # Compassionate bias: If distress in sound, bias toward empathetic decisions
            if 'distress' in str(processed):
                self.ethical_compassion_bias = min(1.0, self.ethical_compassion_bias + 0.1)
            _log_debug(self.node_name, f"{sensor_type} input updated decision context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.process_pending_inputs()
            time.sleep(0.5)

    # --- Core Decision Making Logic ---
    def receive_cognition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Receive and process cognition data with compassionate bias."""
        self.cognition_input = data
        _log_info(self.node_name, f"Received cognition data: {json.dumps(data)}")
        return self.process_decision()

    def receive_emotion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Receive and process emotion data with compassionate bias."""
        self.emotion_input = data
        _log_info(self.node_name, f"Received emotion data: {json.dumps(data)}")
        return self.process_decision()

    def receive_hardware(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Receive and process hardware status with compassionate bias."""
        self.hardware_status = data
        _log_info(self.node_name, f"Received hardware status: {json.dumps(data)}")
        return self.process_decision()

    def process_decision(self) -> Dict[str, Any]:
        """Process decision based on available inputs with compassionate bias."""
        if not (self.cognition_input and self.emotion_input and self.hardware_status):
            _log_warn(self.node_name, "Missing input data for decision making")
            return {"action": "pending", "priority": "none"}

        try:
            # Extract key values with defaults
            emotion_intensity = self.emotion_input.get("intensity", 0)
            hardware_available = self.hardware_status.get("available", False)
            task_confidence = self.cognition_input.get("confidence", 0)

            # Decision logic with compassionate bias
            if emotion_intensity > 0.7:
                action = "respond_emotionally"
                priority = "high"
                details = self.emotion_input
            elif hardware_available and task_confidence > 0.5:
                action = "execute_task"
                priority = "medium"
                details = self.cognition_input
            else:
                action = "wait"
                priority = "low"

            # Compassionate bias: If high emotion distress, escalate priority compassionately
            if emotion_intensity > 0.8 and 'distress' in str(self.emotion_input):
                priority = "high"
                details['compassionate_note'] = "Prioritizing empathetic response due to distress."

            # Log decision with sensory snapshot
            sensory_snapshot = json.dumps(self.sensory_data)
            self._log_decision(action, priority, details, sensory_snapshot)

            decision = {"action": action, "priority": priority, "details": details}
            _log_info(self.node_name, f"Decision: {json.dumps(decision)}")
            return decision
        except Exception as e:
            _log_error(self.node_name, f"Error in decision processing: {e}")
            return {"action": "error", "priority": "none"}

    # --- Callbacks / Input Methods ---
    def cognition_callback(self, msg: Any):
        """ROS callback for cognition input."""
        fields_map = {'data': ('', 'cognition_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        cognition_data = json.loads(data.get('cognition_data', '{}'))
        decision = self.receive_cognition(cognition_data)
        self.publish_decision(decision)

    def emotion_callback(self, msg: Any):
        """ROS callback for emotion input."""
        fields_map = {'data': ('', 'emotion_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        emotion_data = json.loads(data.get('emotion_data', '{}'))
        decision = self.receive_emotion(emotion_data)
        self.publish_decision(decision)

    def hardware_callback(self, msg: Any):
        """ROS callback for hardware status."""
        fields_map = {'data': ('', 'hardware_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        hardware_data = json.loads(data.get('hardware_data', '{}'))
        decision = self.receive_hardware(hardware_data)
        self.publish_decision(decision)

    def process_pending_inputs(self, event: Any = None):
        """Process pending inputs in dynamic or timer mode."""
        if self.pending_inputs:
            input_data = self.pending_inputs.popleft()
            if input_data.get('type') == 'cognition':
                self.receive_cognition(input_data.get('data', {}))
            elif input_data.get('type') == 'emotion':
                self.receive_emotion(input_data.get('data', {}))
            elif input_data.get('type') == 'hardware':
                self.receive_hardware(input_data.get('data', {}))
            self.decision_history.append(self.current_decision.copy())

    # Dynamic input methods
    def receive_input_direct(self, input_type: str, data: Dict[str, Any]):
        """Dynamic method to receive inputs."""
        self.pending_inputs.append({'type': input_type, 'data': data})
        _log_debug(self.node_name, f"Queued {input_type} input.")

    def get_current_decision(self) -> Dict[str, Any]:
        return self.current_decision.copy()

    def publish_decision(self, decision: Dict[str, Any]):
        """Publish decision (ROS or log)."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_decision_output:
                if hasattr(DecisionOutput, 'data'):
                    self.pub_decision_output.publish(String(data=json.dumps(decision)))
                else:
                    decision_msg = DecisionOutput(data=json.dumps(decision))
                    self.pub_decision_output.publish(decision_msg)
                _log_debug(self.node_name, f"Published decision: {decision}")
            # Log to DB
            sensory_snapshot = json.dumps(self.sensory_data)
            self._log_decision(decision.get('action'), decision.get('priority'), decision.get('details'), sensory_snapshot)
        except Exception as e:
            _log_error(self.node_name, f"Failed to publish decision: {e}")

    def _log_decision(self, action: str, priority: str, details: Dict[str, Any], sensory_snapshot: str):
        """Log decision to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO decision_log (id, timestamp, action, priority, confidence_score, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(time.time()), action, priority, details.get('confidence', 0.5), sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log decision: {e}")

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down DecisionMakingNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Run the node."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Decision Making Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = DecisionMakingNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            node.receive_input_direct('cognition', {"task": "analyze", "confidence": 0.9})
            node.receive_input_direct('emotion', {"intensity": 0.8, "type": "joy"})
            node.receive_input_direct('hardware', {"available": True, "status": "online"})
            print("Decision making simulated.")
            print(node.get_current_decision())
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
