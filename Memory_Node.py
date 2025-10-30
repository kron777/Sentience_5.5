```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique execution IDs
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque

# --- Asyncio Imports for HTTP calls ---
import asyncio
import aiohttp
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
    EmotionState = ROSMsgFallback
    ValueDriftState = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    EmotionState = ROSMsgFallback
    ValueDriftState = ROSMsgFallback


# --- Import shared utility functions ---
# Assuming 'sentience/scripts/utils.py' exists and contains parse_message_data and load_config
try:
    from sentience.scripts.utils import parse_message_data, load_config
except ImportError:
    # Fallback implementations
    def parse_message_data(msg: Any, fields_map: Dict[str, tuple], node_name: str = "unknown_node") -> Dict[str, Any]:
        """
        Generic parser for communications (ROS String/JSON or plain dict). 
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
            'hardware_interface_node': {
                'update_interval': 0.5,
                'max_execution_history': 50,
                'batch_size': 50,
                'log_flush_interval_s': 10.0,
                'trait_update_interval_s': 60.0,
                'speech_endpoint': 'http://localhost:8081/tts',
                'motor_endpoint': 'http://localhost:8082/motor',
                'learning_rate': 0.01,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate hardware adjustments (e.g., gentle movements)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 5.0
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


class HardwareInterfaceNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'hardware_interface_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "hardware_log.db")
        self.traits_path = self.params.get('traits_path', os.path.expanduser('~/.sentience/default_character_traits.json'))
        self.update_interval = self.params.get('update_interval', 0.5)
        self.max_execution_history = self.params.get('max_execution_history', 50)
        self.batch_size = self.params.get('batch_size', 50)
        self.log_flush_interval_s = self.params.get('log_flush_interval_s', 10.0)
        self.trait_update_interval_s = self.params.get('trait_update_interval_s', 60.0)
        self.speech_endpoint = self.params.get('speech_endpoint', 'http://localhost:8081/tts')
        self.motor_endpoint = self.params.get('motor_endpoint', 'http://localhost:8082/motor')
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing hardware actions compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Load character traits
        self.character_traits = self._load_character_traits()

        # Internal state
        self.current_execution = {'action_type': 'none', 'parameters': '{}', 'target': 'none'}
        self.execution_history = deque(maxlen=self.max_execution_history)
        self.log_buffer = deque(maxlen=self.batch_size)
        self.trait_update_buffer = deque(maxlen=self.batch_size)
        self.latest_states = {
            'actuator': None,
            'emotion': None,
            'value_drift': None,
        }

        # Initialize SQLite database
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS hardware_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                action_type TEXT,
                parameters TEXT,
                target TEXT,
                execution_status TEXT,
                confidence_score REAL,
                contributing_factors TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Async setup
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._async_thread.start()
        self._async_session = None

        # ROS Compatibility
        self.pub_feedback = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_feedback = rospy.Publisher('/actuator_feedback', String, queue_size=10)

            # Subscribers
            rospy.Subscriber('/actuator_commands', String, self.actuator_callback)
            rospy.Subscriber('/emotion_state', EmotionState, self.emotion_state_callback)
            rospy.Subscriber('/value_drift_state', ValueDriftState, self.value_drift_state_callback)

            # Timers
            rospy.Timer(rospy.Duration(self.update_interval), self.execute_hardware_actions)
            rospy.Timer(rospy.Duration(self.log_flush_interval_s), self.flush_log_buffer)
            rospy.Timer(rospy.Duration(self.trait_update_interval_s), self.flush_trait_updates)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        _log_info(self.node_name, "Hardware Interface Node initialized with compassionate hardware modulation.")

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing hardware compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on hardware (e.g., emotion from sound affects tone)
            if sensor_type == 'vision':
                self.latest_states['actuator'] = {'action_type': 'scan', 'parameters': json.dumps(processed), 'target': 'environment', 'confidence_score': random.uniform(0.6, 0.9)}
            elif sensor_type == 'sound':
                self.latest_states['emotion'] = {'mood': 'reactive' if random.random() < 0.5 else 'calm', 'sentiment_score': random.uniform(-0.1, 0.1)}
            elif sensor_type == 'instructions':
                self.latest_states['value_drift'] = {'drift_score': random.uniform(0.0, 0.3), 'reason': 'user_input'}
            # Compassionate bias: If distress in sound, adjust hardware for gentle actions
            if 'distress' in str(processed):
                self.character_traits['personality'].get('precision', {}).setdefault('value', 0.7)
                self.character_traits['personality']['precision']['value'] = min(1.0, self.character_traits['personality']['precision']['value'] + self.ethical_compassion_bias)
            _log_debug(self.node_name, f"{sensor_type} input updated hardware context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.execute_hardware_actions(None)
            time.sleep(self.update_interval)

    # --- Character Traits Management ---
    def _load_character_traits(self) -> Dict[str, Any]:
        """Load default character traits from JSON."""
        try:
            with open(self.traits_path, 'r') as f:
                traits = json.load(f)
            _log_info(self.node_name, f"Loaded character traits from {self.traits_path}")
            return traits
        except (FileNotFoundError, json.JSONDecodeError) as e:
            _log_warn(self.node_name, f"Failed to load character traits: {e}")
            return {
                "personality": {"precision": {"value": 0.7, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}},
                "emotional_tendencies": {"empathy": {"value": 0.8, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}},
                "behavioral_preferences": {},
                "metadata": {"version": "1.0", "created": datetime.utcnow().isoformat() + 'Z', "last_modified": datetime.utcnow().isoformat() + 'Z', "update_history": []}
            }

    def _update_character_traits(self, trait_category: str, trait_key: str, new_value: float, source: str):
        """Update character traits and log changes with compassionate alignment."""
        try:
            if trait_category not in self.character_traits:
                self.character_traits[trait_category] = {}
            if trait_key not in self.character_traits[trait_category]:
                self.character_traits[trait_category][trait_key] = {"value": 0.5, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}
            # Compassionate bias: Cap precision/empathy to prevent over-adaptation
            new_value = max(0.0, min(1.0, new_value)) if trait_key in ['precision', 'empathy'] else new_value
            self.character_traits[trait_category][trait_key]['value'] = new_value
            self.character_traits[trait_category][trait_key]['weight'] = min(1.0, self.character_traits[trait_category][trait_key]['weight'] + self.learning_rate)
            self.character_traits[trait_category][trait_key]['last_updated'] = datetime.utcnow().isoformat() + 'Z'
            self.character_traits[trait_category][trait_key]['update_source'] = source
            self.character_traits['metadata']['last_modified'] = datetime.utcnow().isoformat() + 'Z'
            self.character_traits['metadata']['update_history'].append({
                'trait': f"{trait_category}.{trait_key}",
                'new_value': new_value,
                'source': source,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
            self.trait_update_buffer.append((trait_category, trait_key, new_value, source))
            _log_info(self.node_name, f"Updated trait {trait_category}.{trait_key} to {new_value} from {source}")
        except Exception as e:
            _log_error(self.node_name, f"Failed to update trait {trait_category}.{trait_key}: {e}")

    # --- Callbacks (generic, ROS or direct) ---
    def actuator_callback(self, msg: Any):
        """Handle incoming actuator commands."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'action_type': ('none', 'action_type'),
            'parameters': ('{}', 'parameters'),
            'target': ('none', 'target'),
            'confidence_score': (0.0, 'confidence_score'),
        }
        self.latest_states['actuator'] = parse_message_data(msg, fields_map, self.node_name)

    def emotion_state_callback(self, msg: Any):
        """Handle incoming emotion state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'mood': ('neutral', 'mood'),
            'sentiment_score': (0.0, 'sentiment_score'),
        }
        self.latest_states['emotion'] = parse_message_data(msg, fields_map, self.node_name)

    def value_drift_state_callback(self, msg: Any):
        """Handle incoming value drift state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'drift_score': (0.0, 'drift_score'),
        }
        self.latest_states['value_drift'] = parse_message_data(msg, fields_map, self.node_name)

    # --- Core Hardware Execution Logic ---
    async def _async_execute_hardware(self, action_type: str, parameters: str, target: str):
        """Execute hardware actions asynchronously with compassionate bias."""
        parameters_dict = json.loads(parameters or '{}')
        execution_status = 'success'
        confidence_score = 0.8
        contributing_factors = {'action_type': action_type, 'target': target}

        # Adjust execution based on emotion and value drift
        emotion = self.latest_states.get('emotion', {'mood': 'neutral', 'sentiment_score': 0.0})
        value_drift = self.latest_states.get('value_drift', {'drift_score': 0.0})
        precision = self.character_traits['personality'].get('precision', {}).get('value', 0.7)
        empathy = self.character_traits['emotional_tendencies'].get('empathy', {}).get('value', 0.8)

        if value_drift.get('drift_score', 0.0) > 0.5 and 'harm' in str(parameters_dict).lower():
            execution_status = 'blocked'
            confidence_score = 0.3
            contributing_factors['reason'] = 'ethical_violation'
            _log_warn(self.node_name, "Hardware action blocked due to moral violation.")
            return execution_status, confidence_score, contributing_factors

        try:
            if action_type == 'speak':
                payload = {'text': parameters_dict.get('text', ''), 'tone': emotion.get('mood') if emotion.get('mood') != 'neutral' else parameters_dict.get('tone', 'neutral')}
                async with self._async_session.post(self.speech_endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                    if response.status != 200:
                        execution_status = 'failed'
                        confidence_score = 0.4
                        contributing_factors['error'] = f"Speech endpoint failed: {response.status}"
                        _log_error(self.node_name, f"Speech action failed: {response.status}")
            elif action_type == 'move':
                payload = {'target': target, 'speed': parameters_dict.get('speed', 0.5), 'precision': precision}
                async with self._async_session.post(self.motor_endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                    if response.status != 200:
                        execution_status = 'failed'
                        confidence_score = 0.4
                        contributing_factors['error'] = f"Motor endpoint failed: {response.status}"
                        _log_error(self.node_name, f"Motor action failed: {response.status}")
            else:
                execution_status = 'skipped'
                confidence_score = 0.5
                contributing_factors['reason'] = 'unsupported_action'
                _log_debug(self.node_name, "Hardware action skipped: unsupported type.")

            # Feedback loop: Update traits based on outcome
            if execution_status == 'success' and precision < 0.9:
                self._update_character_traits('personality', 'precision', min(1.0, precision + 0.05), 'successful_execution')
            if execution_status == 'failed':
                self._update_character_traits('personality', 'precision', max(0.0, precision - 0.05), 'failed_execution')
            if action_type == 'speak' and empathy > 0.7:
                contributing_factors['empathy_adjusted'] = True
                _log_debug(self.node_name, "Feedback applied: Empathy adjusted for speech action.")

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            execution_status = 'failed'
            confidence_score = 0.4
            contributing_factors['error'] = str(e)
            _log_error(self.node_name, f"Hardware execution failed: {e}")

        return execution_status, confidence_score, contributing_factors

    def execute_hardware_actions(self, event: Any = None):
        """Routine hardware action execution."""
        timestamp = str(self._get_current_time())
        actuator = self.latest_states.get('actuator', {'action_type': 'none', 'parameters': '{}', 'target': 'none', 'confidence_score': 0.0})
        action_type = actuator['action_type']
        parameters = actuator['parameters']
        target = actuator['target']

        if action_type == 'none':
            _log_debug(self.node_name, "No action to execute; skipping.")
            return

        execution_status, confidence_score, contributing_factors = asyncio.run_coroutine_threadsafe(
            self._async_execute_hardware(action_type, parameters, target), self._async_loop
        ).result(timeout=5.0)

        self.current_execution = {
            'action_type': action_type,
            'parameters': parameters,
            'target': target
        }
        self.execution_history.append({
            'action_type': action_type,
            'parameters': parameters,
            'target': target,
            'execution_status': execution_status,
            'timestamp': timestamp
        })

        # Publish feedback
        self.publish_feedback(timestamp, action_type, parameters, target, execution_status, confidence_score, contributing_factors)

        # Log to buffer
        sensory_snapshot = json.dumps(self.sensory_data)
        self.log_buffer.append((
            timestamp,
            action_type,
            parameters,
            target,
            execution_status,
            confidence_score,
            json.dumps(contributing_factors),
            sensory_snapshot
        ))

        _log_info(self.node_name, f"Executed {action_type} on {target}, Status: {execution_status}, Confidence: {confidence_score:.2f}")

    def publish_feedback(self, timestamp: str, action_type: str, parameters: str, target: str, execution_status: str, confidence_score: float, contributing_factors: Dict[str, Any]):
        """Publish execution feedback."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_feedback:
                feedback_data = {
                    'timestamp': timestamp,
                    'action_type': action_type,
                    'parameters': parameters,
                    'target': target,
                    'execution_status': execution_status,
                    'confidence_score': confidence_score,
                    'contributing_factors': contributing_factors,
                }
                self.pub_feedback.publish(String(data=json.dumps(feedback_data)))
            _log_debug(self.node_name, f"Published feedback for {action_type}: {execution_status}")
        except Exception as e:
            _log_error(self.node_name, f"Failed to publish feedback: {e}")

    def flush_log_buffer(self, event: Any = None):
        """Flush log buffer to database with sensory snapshot."""
        if not self.log_buffer:
            return
        entries = list(self.log_buffer)
        self.log_buffer.clear()
        try:
            self.cursor.executemany('''
                INSERT INTO hardware_log (id, timestamp, action_type, parameters, target, execution_status, confidence_score, contributing_factors, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [(str(uuid.uuid4()), e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]) for e in entries])
            self.conn.commit()
            _log_info(self.node_name, f"Flushed {len(entries)} hardware log entries to DB.")
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to flush log buffer: {e}")
            for entry in entries:
                self.log_buffer.append(entry)

    def flush_trait_updates(self, event: Any = None):
        """Flush trait updates to JSON file."""
        if not self.trait_update_buffer:
            return
        try:
            with open(self.traits_path, 'w') as f:
                json.dump(self.character_traits, f, indent=2)
            entries = list(self.trait_update_buffer)
            self.trait_update_buffer.clear()
            _log_info(self.node_name, f"Flushed {len(entries)} trait updates to {self.traits_path}")
        except (IOError, json.JSONDecodeError) as e:
            _log_error(self.node_name, f"Failed to flush trait updates: {e}")
            for entry in entries:
                self.trait_update_buffer.append(entry)

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down HardwareInterfaceNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        self.flush_log_buffer()
        self.flush_trait_updates()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        self._shutdown_async_loop()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Run the node with asyncio integration."""
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
    parser = argparse.ArgumentParser(description='Sentience Hardware Interface Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = HardwareInterfaceNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate an actuator command
            node.actuator_callback({'data': json.dumps({'action_type': 'move', 'parameters': json.dumps({'speed': 0.5}), 'target': 'forward', 'confidence_score': 0.8})})
            node.emotion_state_callback({'data': json.dumps({'mood': 'calm', 'sentiment_score': 0.3})})
            time.sleep(3)  # Allow for simulation
            print("Hardware interface simulated.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
