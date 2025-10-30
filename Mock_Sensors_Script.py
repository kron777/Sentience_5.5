```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique message IDs
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
    SensoryQualia = ROSMsgFallback
    InteractionRequest = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    SensoryQualia = ROSMsgFallback
    InteractionRequest = ROSMsgFallback


# --- Import shared utility functions ---
# Assuming 'sentience/scripts/utils.py' exists and contains load_config
try:
    from sentience.scripts.utils import load_config
except ImportError:
    # Fallback implementation
    def load_config(node_name, config_path):
        _log_warn(node_name, f"Mocking load_config for {node_name}. Using hardcoded defaults.")
        return {
            'db_root_path': '/tmp/sentience_db',
            'default_log_level': 'INFO',
            'mock_sensors_node': {
                'publish_interval_sensory_qualia': 1.0,
                'publish_interval_interaction_request': 3.0,
                'simulated_sensory_events': [
                    {"type": "visual", "description": "a person walking by", "salience": 0.6},
                    {"type": "auditory", "description": "a knock on the door", "salience": 0.8},
                    {"type": "tactile", "description": "robot's arm brushes against a surface", "salience": 0.3},
                    {"type": "visual", "description": "a bright red object", "salience": 0.7},
                    {"type": "auditory", "description": "a human voice speaking", "salience": 0.9}
                ],
                'simulated_user_inputs': [
                    {"type": "speech_text", "text": "Hello robot, how are you?", "urgency": 0.5},
                    {"type": "speech_text", "text": "Can you fetch me the book?", "urgency": 0.8, "command_payload": {"action": "fetch", "object": "book"}},
                    {"type": "gesture", "text": "points to a direction", "urgency": 0.4, "gesture_data": {"direction": "forward"}},
                    {"type": "speech_text", "text": "That's wrong, try again!", "urgency": 0.9, "command_payload": {"feedback": "negative"}},
                    {"type": "speech_text", "text": "Thank you, that was helpful.", "urgency": 0.3, "command_payload": {"feedback": "positive"}}
                ],
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate simulations (e.g., positive user interactions)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            }
        }


def _log_info(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [INFO] {msg}", file=sys.stdout)

def _log_warn(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [WARN] {msg}", file=sys.stderr)

def _log_error(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [ERROR] {msg}", file=sys.stderr)

def _log_debug(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [DEBUG] {msg}", file=sys.stdout)


class MockSensorsNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'mock_sensors_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters from 'mock_sensors_node' section of config
        self.mock_sensor_params = self.params.get('mock_sensors_node', {})
        self.sensory_qualia_interval = self.mock_sensor_params.get('publish_interval_sensory_qualia', 1.0)
        self.interaction_request_interval = self.mock_sensor_params.get('publish_interval_interaction_request', 3.0)
        self.simulated_sensory_events = self.mock_sensor_params.get('simulated_sensory_events', [])
        self.simulated_user_inputs = self.mock_sensor_params.get('simulated_user_inputs', [])
        self.ethical_compassion_bias = self.mock_sensor_params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (for dynamic simulation)
        self.sensory_sources = self.mock_sensor_params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.execution_history = deque(maxlen=50)  # Track simulated events for logging
        self.pending_simulations: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for dynamic simulations

        # Initialize SQLite database for mock sensor logs
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "mock_sensors_log.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS mock_sensors_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                event_type TEXT,
                data_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Mock Sensors Node online, simulating inputs with compassionate bias toward positive interactions.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_sensory_qualia = None
        self.pub_interaction_request = None
        self.pub_error_report = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_sensory_qualia = rospy.Publisher('/sensory_qualia', SensoryQualia, queue_size=10)
            self.pub_interaction_request = rospy.Publisher('/interaction_request', InteractionRequest, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)

            # Timers for periodic publishing
            rospy.Timer(rospy.Duration(self.sensory_qualia_interval), self.publish_sensory_qualia)
            rospy.Timer(rospy.Duration(self.interaction_request_interval), self.publish_interaction_request)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing simulations compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on mock data
            if sensor_type == 'vision':
                self.pending_simulations.append({'type': 'sensory_qualia', 'data': {'type': 'visual', 'description': processed.get('description', 'scene'), 'salience': random.uniform(0.4, 0.8)}})
            elif sensor_type == 'sound':
                self.pending_simulations.append({'type': 'interaction_request', 'data': {'text': processed.get('transcription', 'sound input'), 'urgency': random.uniform(0.3, 0.6)}})
            elif sensor_type == 'instructions':
                self.pending_simulations.append({'type': 'sensory_qualia', 'data': {'type': 'tactile', 'description': processed.get('instruction', 'user command'), 'salience': random.uniform(0.5, 0.9)}})
            # Compassionate bias: If distress in sound, bias toward positive simulated interactions
            if 'distress' in str(processed):
                self.ethical_compassion_bias = min(1.0, self.ethical_compassion_bias + 0.1)
                # Adjust next simulation to be more supportive
                if self.pending_simulations:
                    self.pending_simulations[-1]['data']['compassionate_note'] = "Prioritizing empathetic response."
            _log_debug(self.node_name, f"{sensor_type} input updated simulation context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.publish_sensory_qualia(None)
            time.sleep(self.sensory_qualia_interval)
            self.publish_interaction_request(None)
            time.sleep(self.interaction_request_interval)

    # --- Core Simulation Logic ---
    def publish_sensory_qualia(self, event: Any = None):
        """Publish a simulated SensoryQualia message with compassionate bias toward positive events."""
        if not self.simulated_sensory_events:
            _log_debug(self.node_name, "No simulated sensory events configured. Skipping SensoryQualia.")
            return

        # Pick a random sensory event from the predefined list
        simulated_event = random.choice(self.simulated_sensory_events)
        
        # Compassionate bias: Occasionally boost positive events
        if random.random() < self.ethical_compassion_bias:
            simulated_event['description'] += " (with positive emotional nuance)"
            simulated_event['salience'] = min(1.0, simulated_event.get('salience', 0.5) + 0.1)
        
        timestamp = str(self._get_current_time())
        qualia_id = str(uuid.uuid4())  # Unique ID for each qualia event
        qualia_type = simulated_event.get('type', 'generic')
        modality = 'visual' if 'visual' in qualia_type.lower() else ('auditory' if 'auditory' in qualia_type.lower() else 'tactile')
        description_summary = simulated_event.get('description', 'simulated event')
        salience_score = simulated_event.get('salience', 0.5)
        raw_data_hash = str(random.getrandbits(128))  # Simulate a hash for raw data

        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_sensory_qualia:
                if hasattr(SensoryQualia, 'data'):  # String fallback
                    qualia_data = {
                        'timestamp': timestamp,
                        'qualia_id': qualia_id,
                        'qualia_type': qualia_type,
                        'modality': modality,
                        'description_summary': description_summary,
                        'salience_score': salience_score,
                        'raw_data_hash': raw_data_hash
                    }
                    self.pub_sensory_qualia.publish(String(data=json.dumps(qualia_data)))
                else:
                    qualia_msg = SensoryQualia()
                    qualia_msg.timestamp = timestamp
                    qualia_msg.qualia_id = qualia_id
                    qualia_msg.qualia_type = qualia_type
                    qualia_msg.modality = modality
                    qualia_msg.description_summary = description_summary
                    qualia_msg.salience_score = salience_score
                    qualia_msg.raw_data_hash = raw_data_hash
                    self.pub_sensory_qualia.publish(qualia_msg)
            else:
                # Dynamic mode: Log or store
                event_entry = {
                    'id': qualia_id,
                    'timestamp': timestamp,
                    'type': 'sensory_qualia',
                    'data': qualia_data,
                    'sensory_snapshot': self.sensory_data
                }
                self._log_mock_event(event_entry)
            _log_debug(self.node_name, f"Published Sensory Qualia: {description_summary} ({modality}).")
        except Exception as e:
            self._report_error("PUBLISH_SENSORY_QUALIA_ERROR", f"Failed to publish sensory qualia: {e}", 0.7)

    def publish_interaction_request(self, event: Any = None):
        """Publish a simulated InteractionRequest message with compassionate bias toward supportive interactions."""
        if not self.simulated_user_inputs:
            _log_debug(self.node_name, "No simulated user inputs configured. Skipping InteractionRequest.")
            return

        # Pick a random user input from the predefined list
        simulated_input = random.choice(self.simulated_user_inputs)
        
        # Compassionate bias: Occasionally bias toward positive/empathetic interactions
        if random.random() < self.ethical_compassion_bias:
            simulated_input['text'] += " (with appreciative tone)"
            simulated_input['urgency'] = min(1.0, simulated_input.get('urgency', 0.5) - 0.1)  # Lower urgency for positive

        timestamp = str(self._get_current_time())
        request_id = str(uuid.uuid4())  # Unique ID for each request
        request_type = simulated_input.get('type', 'speech_text')  # e.g., 'speech_text', 'gesture', 'command'
        user_id = simulated_input.get('user_id', 'simulated_user_1')
        command_payload = json.dumps(simulated_input.get('command_payload', {}))  # Ensure it's a JSON string
        urgency_score = simulated_input.get('urgency', 0.5)
        speech_text = simulated_input.get('text', '')
        gesture_data_json = json.dumps(simulated_input.get('gesture_data', {}))  # Ensure it's a JSON string

        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_interaction_request:
                if hasattr(InteractionRequest, 'data'):  # String fallback
                    request_data = {
                        'timestamp': timestamp,
                        'request_id': request_id,
                        'request_type': request_type,
                        'user_id': user_id,
                        'command_payload': command_payload,
                        'urgency_score': urgency_score,
                        'speech_text': speech_text,
                        'gesture_data_json': gesture_data_json
                    }
                    self.pub_interaction_request.publish(String(data=json.dumps(request_data)))
                else:
                    request_msg = InteractionRequest()
                    request_msg.timestamp = timestamp
                    request_msg.request_id = request_id
                    request_msg.request_type = request_type
                    request_msg.user_id = user_id
                    request_msg.command_payload = command_payload
                    request_msg.urgency_score = urgency_score
                    request_msg.speech_text = speech_text
                    request_msg.gesture_data_json = gesture_data_json
                    self.pub_interaction_request.publish(request_msg)
            else:
                # Dynamic mode: Log or store
                request_entry = {
                    'id': request_id,
                    'timestamp': timestamp,
                    'type': 'interaction_request',
                    'data': request_data,
                    'sensory_snapshot': self.sensory_data
                }
                self._log_mock_event(request_entry)
            _log_debug(self.node_name, f"Published Interaction Request: '{speech_text}' (Type: {request_type}).")
        except Exception as e:
            self._report_error("PUBLISH_INTERACTION_REQUEST_ERROR", f"Failed to publish interaction request: {e}", 0.7)

    def _log_mock_event(self, event_entry: Dict[str, Any]):
        """Log simulated event to DB for persistence."""
        try:
            self.cursor.execute('''
                INSERT INTO mock_sensors_log (id, timestamp, event_type, data_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                event_entry['id'], event_entry['timestamp'], event_entry['type'],
                json.dumps(event_entry['data']), json.dumps(self.sensory_data)
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log mock event: {e}")

    def _report_error(self, error_type: str, description: str, severity: float = 0.5, context: Optional[Dict] = None):
        """Report an error with compassionate note."""
        timestamp = str(self._get_current_time())
        compassionate_note = f"Compassionate reflection: Emphasize learning from this to improve future simulations. Bias: {self.ethical_compassion_bias}." if severity > 0.5 else ""
        error_msg_data = {
            'timestamp': timestamp,
            'source_node': self.node_name,
            'error_type': error_type,
            'description': description,
            'severity': severity,
            'compassionate_note': compassionate_note,
            'context': context or {}
        }
        if ROS_AVAILABLE and self.ros_enabled and self.pub_error_report:
            try:
                self.pub_error_report.publish(String(data=json.dumps(error_msg_data)))
                rospy.logerr(f"{self.node_name}: REPORTED ERROR: {error_type} - {description}")
            except Exception as e:
                _log_error(self.node_name, f"Failed to publish error report: {e}")
        else:
            _log_error(self.node_name, f"REPORTED ERROR: {error_type} - {description} (Severity: {severity})")
        # Log to DB
        self._log_mock_event({'type': 'error', 'data': error_msg_data})

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down MockSensorsNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Run the node with simulated or actual ROS publishing."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                while True:
                    time.sleep(1)  # Idle in dynamic mode; simulations run in thread
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Mock Sensors Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = MockSensorsNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate a few events
            time.sleep(5)
            print("Mock sensors simulation complete. Generated events logged to DB.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
