```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import sys
import argparse
from collections import deque
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any, Optional

# --- Asyncio Imports for LLM calls ---
import asyncio
import aiohttp
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
    CognitiveDirective = ROSMsgFallback
    SensoryQualiaState = ROSMsgFallback
    EmotionState = ROSMsgFallback
    MotivationState = ROSMsgFallback
    ValueDriftState = ROSMsgFallback
    WorldModelState = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    CognitiveDirective = ROSMsgFallback
    SensoryQualiaState = ROSMsgFallback
    EmotionState = ROSMsgFallback
    MotivationState = ROSMsgFallback
    ValueDriftState = ROSMsgFallback
    WorldModelState = ROSMsgFallback


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
            'feedback_loop_node': {
                'update_interval': 0.5,
                'max_feedback_history': 50,
                'batch_size': 50,
                'log_flush_interval_s': 10.0,
                'trait_update_interval_s': 60.0,
                'llm_endpoint': 'http://localhost:8080/phi2',
                'learning_rate': 0.01,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate feedback
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
        }.get(node_name, {})  # Return node-specific or empty dict


def _log_info(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [INFO] {msg}", file=sys.stdout)

def _log_warn(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [WARN] {msg}", file=sys.stderr)

def _log_error(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [ERROR] {msg}", file=sys.stderr)

def _log_debug(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [DEBUG] {msg}", file=sys.stderr)


class FeedbackLoopNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'feedback_loop_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "feedback_log.db")
        self.traits_path = self.params.get('traits_path', os.path.expanduser('~/.sentience/default_character_traits.json'))
        self.update_interval = self.params.get('update_interval', 0.5)
        self.max_feedback_history = self.params.get('max_feedback_history', 50)
        self.batch_size = self.params.get('batch_size', 50)
        self.log_flush_interval_s = self.params.get('log_flush_interval_s', 10.0)
        self.trait_update_interval_s = self.params.get('trait_update_interval_s', 60.0)
        self.llm_endpoint = self.params.get('llm_endpoint', 'http://localhost:8080/phi2')
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing feedback compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Load character traits
        self.character_traits = self._load_character_traits()

        # Internal state
        self.current_feedback = {'type': 'none', 'target_node': 'none', 'adjustment': '{}'}
        self.feedback_history = deque(maxlen=self.max_feedback_history)
        self.log_buffer = deque(maxlen=self.batch_size)
        self.trait_update_buffer = deque(maxlen=self.batch_size)
        self.latest_states = {
            'actuator': None,
            'sensory_qualia': None,
            'emotion': None,
            'motivation': None,
            'value_drift': None,
            'world_model': None,
        }

        # Initialize SQLite database
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                feedback_type TEXT,
                target_node TEXT,
                adjustment TEXT,
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
        self.pub_directive = None
        self.pub_feedback = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)
            self.pub_feedback = rospy.Publisher('/action_feedback', String, queue_size=10)

            # Subscribers
            rospy.Subscriber('/actuator_commands', String, self.actuator_callback)
            rospy.Subscriber('/sensory_qualia_state', SensoryQualiaState, self.sensory_qualia_callback)
            rospy.Subscriber('/emotion_state', EmotionState, self.emotion_state_callback)
            rospy.Subscriber('/motivation_state', MotivationState, self.motivation_state_callback)
            rospy.Subscriber('/value_drift_state', ValueDriftState, self.value_drift_state_callback)
            rospy.Subscriber('/world_model_state', WorldModelState, self.world_model_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.update_interval), self.process_feedback)
            rospy.Timer(rospy.Duration(self.log_flush_interval_s), self.flush_log_buffer)
            rospy.Timer(rospy.Duration(self.trait_update_interval_s), self.flush_trait_updates)
        else:
            # Dynamic mode
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        _log_info(self.node_name, "Feedback Loop Node initialized with compassionate feedback modulation.")

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing feedback compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on feedback
            if sensor_type == 'vision':
                self.latest_states['sensory_qualia'] = {'sensory_data_json': json.dumps(processed), 'salience_score': random.uniform(0.3, 0.8)}
            elif sensor_type == 'sound':
                self.latest_states['emotion'] = {'mood': 'reactive' if random.random() < 0.5 else 'neutral', 'sentiment_score': random.uniform(-0.2, 0.2)}
            elif sensor_type == 'instructions':
                self.latest_states['motivation'] = {'dominant_goal_id': 'user_response', 'overall_drive_level': random.uniform(0.4, 0.8)}
            _log_debug(self.node_name, f"{sensor_type} input updated feedback context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.process_feedback(None)
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
                "personality": {"adaptability": {"value": 0.7, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}},
                "emotional_tendencies": {"empathy": {"value": 0.8, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}},
                "behavioral_preferences": {},
                "metadata": {"version": "1.0", "created": datetime.utcnow().isoformat() + 'Z', "last_updated": datetime.utcnow().isoformat() + 'Z', "update_history": []}
            }

    def _update_character_traits(self, trait_category: str, trait_key: str, new_value: float, source: str):
        """Update character traits and log changes with compassionate bias."""
        try:
            if trait_category not in self.character_traits:
                self.character_traits[trait_category] = {}
            if trait_key not in self.character_traits[trait_category]:
                self.character_traits[trait_category][trait_key] = {"value": 0.5, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}
            # Compassionate bias: Cap empathy/adaptability to prevent over-adaptation
            new_value = max(0.0, min(1.0, new_value)) if trait_key in ['adaptability', 'empathy'] else new_value
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

    # --- Callbacks (generic for ROS/dynamic) ---
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

    def sensory_qualia_callback(self, msg: Any):
        """Handle incoming sensory qualia data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'sensory_data_json': ('{}', 'sensory_data_json'),
            'salience_score': (0.0, 'salience_score'),
        }
        self.latest_states['sensory_qualia'] = parse_message_data(msg, fields_map, self.node_name)

    def emotion_state_callback(self, msg: Any):
        """Handle incoming emotion state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'mood': ('neutral', 'mood'),
            'sentiment_score': (0.0, 'sentiment_score'),
        }
        self.latest_states['emotion'] = parse_message_data(msg, fields_map, self.node_name)

    def motivation_state_callback(self, msg: Any):
        """Handle incoming motivation state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'dominant_goal_id': ('none', 'dominant_goal_id'),
            'overall_drive_level': (0.0, 'overall_drive_level'),
        }
        self.latest_states['motivation'] = parse_message_data(msg, fields_map, self.node_name)

    def value_drift_state_callback(self, msg: Any):
        """Handle incoming value drift state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'drift_score': (0.0, 'drift_score'),
        }
        self.latest_states['value_drift'] = parse_message_data(msg, fields_map, self.node_name)

    def world_model_callback(self, msg: Any):
        """Handle incoming world model data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'entities_json': ('[]', 'entities_json'),
        }
        self.latest_states['world_model'] = parse_message_data(msg, fields_map, self.node_name)

    # --- Core Feedback Loop Logic ---
    async def _async_process_feedback(self):
        """Process action outcomes and generate feedback with compassionate bias."""
        actuator = self.latest_states.get('actuator', {'action_type': 'none', 'parameters': '{}', 'target': 'none', 'confidence_score': 0.0})
        sensory_qualia = self.latest_states.get('sensory_qualia', {'sensory_data_json': '{}', 'salience_score': 0.0})
        emotion = self.latest_states.get('emotion', {'mood': 'neutral', 'sentiment_score': 0.0})
        motivation = self.latest_states.get('motivation', {'dominant_goal_id': 'none', 'overall_drive_level': 0.0})
        value_drift = self.latest_states.get('value_drift', {'drift_score': 0.0})
        world_model = self.latest_states.get('world_model', {'entities_json': '[]'})

        sensory_data = json.loads(sensory_qualia.get('sensory_data_json', '{}') or '{}')
        entities = json.loads(world_model.get('entities_json', '[]') or '[]')
        action_parameters = json.loads(actuator.get('parameters', '{}') or '{}')

        # Simplified LLM input for feedback generation with compassionate bias
        llm_input = {
            'action_type': actuator.get('action_type'),
            'action_parameters': action_parameters,
            'action_target': actuator.get('target'),
            'action_confidence': actuator.get('confidence_score'),
            'sensory_context': sensory_data.get('context', 'neutral'),
            'salience_score': sensory_qualia.get('salience_score'),
            'mood': emotion.get('mood'),
            'goal_id': motivation.get('dominant_goal_id'),
            'drift_score': value_drift.get('drift_score'),
            'entities': entities,
            'recent_feedback': [{'type': e['type'], 'target_node': e['target_node']} for e in list(self.feedback_history)[-5:]],  # Limit history
            'traits': {
                'adaptability': self.character_traits['personality'].get('adaptability', {}).get('value', 0.7),
                'empathy': self.character_traits['emotional_tendencies'].get('empathy', {}).get('value', 0.8),
            },
            'ethical_compassion_bias': self.ethical_compassion_bias  # Guide toward compassionate feedback
        }

        # Query LLM for feedback
        try:
            async with self._async_session.post(self.llm_endpoint, json=llm_input, timeout=aiohttp.ClientTimeout(total=self.llm_timeout)) as response:
                if response.status == 200:
                    llm_output = await response.json()
                    feedback_type = llm_output.get('feedback_type', 'none')
                    target_node = llm_output.get('target_node', 'none')
                    adjustment = llm_output.get('adjustment', {})
                    confidence_score = llm_output.get('confidence', 0.5)
                    contributing_factors = llm_output.get('factors', {})
                else:
                    _log_warn(self.node_name, f"LLM request failed with status {response.status}")
                    feedback_type, target_node, adjustment, confidence_score, contributing_factors = 'none', 'none', {}, 0.3, {}
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            _log_error(self.node_name, f"LLM request failed: {e}")
            feedback_type, target_node, adjustment, confidence_score, contributing_factors = 'none', 'none', {}, 0.3, {}

        # Adjust feedback based on traits and inputs with compassionate bias
        adaptability = self.character_traits['personality'].get('adaptability', {}).get('value', 0.7)
        empathy = self.character_traits['emotional_tendencies'].get('empathy', {}).get('value', 0.8)

        if actuator.get('action_type') == 'speak' and sensory_data.get('user_reaction', 'neutral') == 'positive' and empathy > 0.7:
            feedback_type = 'positive_reinforcement' if feedback_type == 'none' else feedback_type
            target_node = 'cognitive_reasoning_node' if target_node == 'none' else target_node
            adjustment = {'action': 'increase_empathy', 'value': 0.05} if not adjustment else adjustment
            self._update_character_traits('personality', 'adaptability', min(1.0, adaptability + 0.05), 'positive_outcome')
        if actuator.get('action_type') != 'none' and sensory_data.get('user_reaction', 'neutral') == 'negative':
            feedback_type = 'corrective' if feedback_type == 'none' else feedback_type
            target_node = 'cognitive_control_node' if target_node == 'none' else target_node
            adjustment = {'action': 'reassess_directive', 'priority': 0.7} if not adjustment else adjustment
        if value_drift.get('drift_score', 0.0) > 0.5:
            feedback_type = 'ethical_adjustment' if feedback_type == 'none' else feedback_type
            target_node = 'value_drift_monitor_node' if target_node == 'none' else target_node
            adjustment = {'action': 'reassess_values', 'priority': 0.8} if not adjustment else adjustment
        if motivation.get('dominant_goal_id') == 'user_assistance' and adaptability > 0.6:
            adjustment['priority'] = adjustment.get('priority', 0.5) + 0.1
            confidence_score = min(1.0, confidence_score + 0.1)

        # Compassionate bias: If high emotion distress, add empathetic adjustment
        if emotion.get('sentiment_score', 0.0) < -0.5 and self.ethical_compassion_bias > 0.2:
            adjustment['compassionate'] = True
            feedback_type = 'empathetic_adjustment' if feedback_type == 'none' else feedback_type

        adjustment_json = json.dumps(adjustment)
        return feedback_type, target_node, adjustment_json, confidence_score, contributing_factors

    def process_feedback(self, event: Any = None):
        """Periodic feedback processing wrapper."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM feedback processing active. Skipping cycle.")
            return

        self.active_llm_task = asyncio.run_coroutine_threadsafe(
            self._async_process_feedback(), self._async_loop
        )
        future = self.active_llm_task
        try:
            result = future.result(timeout=1.0)  # Short timeout for real-time
            if result:
                feedback_type, target_node, adjustment_json, confidence_score, contributing_factors = result
                timestamp = str(self._get_current_time())

                self.current_feedback = {
                    'type': feedback_type,
                    'target_node': target_node,
                    'adjustment': adjustment_json
                }
                self.feedback_history.append({
                    'type': feedback_type,
                    'target_node': target_node,
                    'adjustment': adjustment_json,
                    'timestamp': timestamp
                })

                # Publish feedback as directive
                self.publish_directive(timestamp, feedback_type, target_node, adjustment_json, confidence_score, contributing_factors)

                # Publish feedback to action_feedback topic
                self.publish_feedback(timestamp, feedback_type, target_node, adjustment_json, confidence_score, contributing_factors)

                # Log state
                sensory_snapshot = json.dumps(self.sensory_data)
                self.log_buffer.append((
                    timestamp,
                    feedback_type,
                    target_node,
                    adjustment_json,
                    confidence_score,
                    json.dumps(contributing_factors),
                    sensory_snapshot
                ))

                _log_info(self.node_name, f"Feedback: {feedback_type}, Target: {target_node}, Confidence: {confidence_score:.2f}")
        except asyncio.TimeoutError:
            _log_warn(self.node_name, "Feedback processing timed out; skipping.")
        except Exception as e:
            _log_error(self.node_name, f"Feedback processing failed: {e}")

    def publish_directive(self, timestamp: str, feedback_type: str, target_node: str, adjustment_json: str, confidence_score: float, contributing_factors: Dict):
        """Publish feedback as a cognitive directive."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_directive:
                if hasattr(CognitiveDirective, 'data'):  # String fallback
                    directive_data = {
                        'timestamp': timestamp,
                        'directive_type': feedback_type,
                        'target_node': target_node,
                        'command_payload': adjustment_json,
                        'confidence_score': confidence_score,
                        'contributing_factors': json.dumps(contributing_factors),
                    }
                    self.pub_directive.publish(String(data=json.dumps(directive_data)))
                else:
                    msg = CognitiveDirective()
                    msg.timestamp = timestamp
                    msg.directive_type = feedback_type
                    msg.target_node = target_node
                    msg.command_payload = adjustment_json
                    msg.confidence_score = confidence_score
                    msg.contributing_factors_json = json.dumps(contributing_factors)
                    self.pub_directive.publish(msg)
        except Exception as e:
            _log_error(self.node_name, f"Failed to publish directive: {e}")

    def publish_feedback(self, timestamp: str, feedback_type: str, target_node: str, adjustment_json: str, confidence_score: float, contributing_factors: Dict):
        """Publish feedback to action_feedback topic."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_feedback:
                feedback_data = {
                    'timestamp': timestamp,
                    'feedback_type': feedback_type,
                    'target_node': target_node,
                    'adjustment': adjustment_json,
                    'confidence_score': confidence_score,
                    'contributing_factors': contributing_factors,
                }
                self.pub_feedback.publish(String(data=json.dumps(feedback_data)))
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
                INSERT INTO feedback_log (timestamp, feedback_type, target_node, adjustment, confidence_score, contributing_factors, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', [(e[0], e[1], e[2], e[3], e[4], e[5], e[6]) for e in entries])
            self.conn.commit()
            _log_info(self.node_name, f"Flushed {len(entries)} feedback log entries to DB.")
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
        _log_info(self.node_name, "Shutting down FeedbackLoopNode.")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Feedback Loop Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = FeedbackLoopNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate inputs
            node.actuator_callback({'data': json.dumps({'action_type': 'move', 'confidence_score': 0.8})})
            node.emotion_state_callback({'data': json.dumps({'mood': 'satisfied', 'sentiment_score': 0.7})})
            time.sleep(2)
            print("Feedback loop simulated.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
