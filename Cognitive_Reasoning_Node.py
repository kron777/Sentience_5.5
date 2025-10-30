```python:disable-run
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
    CognitiveReasoningState = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    SensoryQualiaState = ROSMsgFallback
    AttentionState = ROSMsgFallback
    EmotionState = ROSMsgFallback
    MotivationState = ROSMsgFallback
    ValueDriftState = ROSMsgFallback
    WorldModelState = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    CognitiveReasoningState = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    SensoryQualiaState = ROSMsgFallback
    AttentionState = ROSMsgFallback
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
            'cognitive_reasoning_node': {
                'update_interval': 0.5,
                'max_reasoning_history': 50,
                'batch_size': 50,
                'log_flush_interval_s': 10.0,
                'trait_update_interval_s': 60.0,
                'llm_endpoint': 'http://localhost:8080/phi2',
                'learning_rate': 0.01,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate reasoning
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
    print(f"[{datetime.now().isoformat()}] {node_name}: [DEBUG] {msg}", file=sys.stdout)


class CognitiveReasoningNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'cognitive_reasoning_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "reasoning_log.db")
        self.traits_path = self.params.get('traits_path', os.path.expanduser('~/.sentience/default_character_traits.json'))
        self.update_interval = self.params.get('update_interval', 0.5)
        self.max_reasoning_history = self.params.get('max_reasoning_history', 50)
        self.batch_size = self.params.get('batch_size', 50)
        self.log_flush_interval_s = self.params.get('log_flush_interval_s', 10.0)
        self.trait_update_interval_s = self.params.get('trait_update_interval_s', 60.0)
        self.llm_endpoint = self.params.get('llm_endpoint', 'http://localhost:8080/phi2')
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_input = self._create_sensory_placeholder('vision')
        self.sound_input = self._create_sensory_placeholder('sound')
        self.instructions_input = self._create_sensory_placeholder('instructions')

        # Load character traits
        self.character_traits = self._load_character_traits()

        # Internal state
        self.current_decision = {'type': 'none', 'action': 'none', 'rationale': ''}
        self.reasoning_history = deque(maxlen=self.max_reasoning_history)
        self.log_buffer = deque(maxlen=self.batch_size)
        self.trait_update_buffer = deque(maxlen=self.batch_size)
        self.latest_states = {
            'sensory_qualia': None,
            'attention': None,
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
            CREATE TABLE IF NOT EXISTS reasoning_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                decision_type TEXT,
                action TEXT,
                rationale TEXT,
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
        self.pub_reasoning_state = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_reasoning_state = rospy.Publisher('/cognitive_reasoning_state', CognitiveReasoningState, queue_size=10)

            # Subscribers
            rospy.Subscriber('/sensory_qualia_state', SensoryQualiaState, self.sensory_qualia_callback)
            rospy.Subscriber('/attention_state', AttentionState, self.attention_state_callback)
            rospy.Subscriber('/emotion_state', EmotionState, self.emotion_state_callback)
            rospy.Subscriber('/motivation_state', MotivationState, self.motivation_state_callback)
            rospy.Subscriber('/value_drift_state', ValueDriftState, self.value_drift_state_callback)
            rospy.Subscriber('/world_model_state', WorldModelState, self.world_model_callback)
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.directive_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_input)
            rospy.Subscriber('/audio_input', String, self.sound_input)
            rospy.Subscriber('/user_instructions', String, self.instructions_input)

            rospy.Timer(rospy.Duration(self.update_interval), self.perform_reasoning)
            rospy.Timer(rospy.Duration(self.log_flush_interval_s), self.flush_log_buffer)
            rospy.Timer(rospy.Duration(self.trait_update_interval_s), self.flush_trait_updates)
        else:
            # Dynamic mode
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        _log_info(self.node_name, "Cognitive Reasoning Node initialized with compassionate discernment.")

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing reasoning."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on reasoning
            if sensor_type == 'vision':
                self.latest_states['sensory_qualia'] = {'sensory_data_json': json.dumps(processed), 'salience_score': random.uniform(0.3, 0.8)}
            elif sensor_type == 'sound':
                self.latest_states['sensory_qualia'] = {'sensory_data_json': json.dumps(processed), 'salience_score': random.uniform(0.2, 0.7)}
            elif sensor_type == 'instructions':
                self.latest_states['directive'] = {'directive_type': 'user_response', 'command_payload': json.dumps(processed)}
            _log_debug(self.node_name, f"{sensor_type} influenced reasoning state at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.perform_reasoning(None)
            time.sleep(self.update_interval)

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    # --- Asyncio Thread Management ---
    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    async def _create_async_session(self):
        _log_info(self.node_name, "Creating aiohttp ClientSession...")
        self._async_session = aiohttp.ClientSession()
        _log_info(self.node_name, "aiohttp ClientSession created.")

    async def _close_async_session(self):
        if self._async_session:
            _log_info(self.node_name, "Closing aiohttp ClientSession...")
            await self._async_session.close()
            self._async_session = None
            _log_info(self.node_name, "aiohttp ClientSession closed.")

    def _shutdown_async_loop(self):
        if self._async_loop and self._async_thread.is_alive():
            _log_info(self.node_name, "Shutting down asyncio loop...")
            future = asyncio.run_coroutine_threadsafe(self._close_async_session(), self._async_loop)
            try:
                future.result(timeout=5.0)
            except asyncio.TimeoutError:
                _log_warn(self.node_name, "Timeout waiting for async session to close.")
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            self._async_thread.join(timeout=5.0)
            if self._async_thread.is_alive():
                _log_warn(self.node_name, "Asyncio thread did not shut down gracefully.")
            _log_info(self.node_name, "Asyncio loop shut down.")

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
                "personality": {"rationality": {"value": 0.7, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}},
                "emotional_tendencies": {"empathy": {"value": 0.8, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}},
                "behavioral_preferences": {},
                "metadata": {"version": "1.0", "created": datetime.utcnow().isoformat() + 'Z', "last_modified": datetime.utcnow().isoformat() + 'Z', "update_history": []}
            }

    def _update_character_traits(self, trait_category: str, trait_key: str, new_value: float, source: str):
        """Update character traits and log changes with ethical alignment."""
        try:
            if trait_category not in self.character_traits:
                self.character_traits[trait_category] = {}
            if trait_key not in self.character_traits[trait_category]:
                self.character_traits[trait_category][trait_key] = {"value": 0.5, "weight": 1.0, "last_updated": datetime.utcnow().isoformat() + 'Z', "update_source": "default"}
            # Ethical bias: Cap rationality/empathy to prevent over-adaptation
            new_value = max(0.0, min(1.0, new_value)) if trait_key in ['rationality', 'empathy'] else new_value
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
    def sensory_qualia_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'sensory_data_json': ('{}', 'sensory_data_json'),
            'salience_score': (0.0, 'salience_score'),
            'novelty_score': (0.0, 'novelty_score'),
        }
        self.latest_states['sensory_qualia'] = parse_message_data(msg, fields_map, self.node_name)

    def attention_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'focus_type': ('idle', 'focus_type'),
            'focus_target': ('none', 'focus_target'),
            'priority_score': (0.0, 'priority_score'),
        }
        self.latest_states['attention'] = parse_message_data(msg, fields_map, self.node_name)

    def emotion_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'mood': ('neutral', 'mood'),
            'sentiment_score': (0.0, 'sentiment_score'),
            'mood_intensity': (0.0, 'mood_intensity'),
        }
        self.latest_states['emotion'] = parse_message_data(msg, fields_map, self.node_name)

    def motivation_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'dominant_goal_id': ('none', 'dominant_goal_id'),
            'overall_drive_level': (0.0, 'overall_drive_level'),
        }
        self.latest_states['motivation'] = parse_message_data(msg, fields_map, self.node_name)

    def value_drift_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'drift_score': (0.0, 'drift_score'),
            'correction_action': ('none', 'correction_action'),
        }
        self.latest_states['value_drift'] = parse_message_data(msg, fields_map, self.node_name)

    def world_model_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'entities_json': ('[]', 'entities_json'),
        }
        self.latest_states['world_model'] = parse_message_data(msg, fields_map, self.node_name)

    def directive_callback(self, msg: Any):
        """Handle incoming cognitive directives."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'directive_type': ('none', 'directive_type'),
            'target_node': ('none', 'target_node'),
            'command_payload': ('{}', 'command_payload'),
            'reason': ('', 'reason'),
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        if data.get('target_node') in [self.node_name, 'all']:
            try:
                payload = json.loads(data.get('command_payload', '{}') or '{}')
                reasoning_event = {
                    'id': str(uuid4()),
                    'type': data.get('directive_type'),
                    'task': payload.get('task', 'none'),
                    'timestamp': data.get('timestamp'),
                }
                self.reasoning_history.append(reasoning_event)
                if data.get('directive_type') == 'complex_decision':
                    self._update_character_traits('personality', 'rationality', min(1.0, self.character_traits['personality'].get('rationality', {}).get('value', 0.7) + 0.05), 'directive')
                _log_debug(self.node_name, f"Received directive: {reasoning_event['type']} (ID: {reasoning_event['id']})")
            except json.JSONDecodeError:
                _log_warn(self.node_name, f"Failed to parse directive payload: {data.get('command_payload')}")

    # --- Core Reasoning Logic ---
    async def _async_perform_reasoning(self):
        """Perform reasoning asynchronously using LLM with compassionate bias."""
        sensory_qualia = self.latest_states.get('sensory_qualia', {'sensory_data_json': '{}', 'salience_score': 0.0})
        attention = self.latest_states.get('attention', {'focus_type': 'idle', 'focus_target': 'none'})
        emotion = self.latest_states.get('emotion', {'mood': 'neutral', 'sentiment_score': 0.0})
        motivation = self.latest_states.get('motivation', {'dominant_goal_id': 'none', 'overall_drive_level': 0.0})
        value_drift = self.latest_states.get('value_drift', {'drift_score': 0.0})
        world_model = self.latest_states.get('world_model', {'entities_json': '[]'})

        # Incorporate sensory data
        sensory_data = json.loads(sensory_qualia.get('sensory_data_json', '{}') or '{}')
        entities = json.loads(world_model.get('entities_json', '[]') or '[]')

        # Simplified LLM input with ethical bias
        llm_input = {
            'context': sensory_data.get('context', 'neutral'),
            'salience_score': sensory_qualia.get('salience_score'),
            'focus_type': attention.get('focus_type'),
            'focus_target': attention.get('focus_target'),
            'mood': emotion.get('mood'),
            'sentiment_score': emotion.get('sentiment_score'),
            'goal_id': motivation.get('dominant_goal_id'),
            'drift_score': value_drift.get('drift_score'),
            'entities': entities,
            'recent_decisions': [{'type': e['type'], 'action': e['action']} for e in list(self.reasoning_history)[-5:]],  # Limit history for efficiency
            'traits': {
                'rationality': self.character_traits['personality'].get('rationality', {}).get('value', 0.7),
                'empathy': self.character_traits['emotional_tendencies'].get('empathy', {}).get('value', 0.8),
            },
            'ethical_compassion_bias': self.ethical_compassion_bias  # Guide toward compassionate reasoning
        }

        # Query LLM
        try:
            async with self._async_session.post(self.llm_endpoint, json=llm_input, timeout=aiohttp.ClientTimeout(total=self.llm_timeout)) as response:
                if response.status == 200:
                    llm_output = await response.json()
                    decision_type = llm_output.get('decision_type', 'none')
                    action = llm_output.get('action', 'none')
                    rationale = llm_output.get('rationale', 'No rationale provided.')
                    confidence_score = llm_output.get('confidence', 0.5)
                    contributing_factors = llm_output.get('factors', {})
                else:
                    _log_warn(self.node_name, f"LLM request failed with status {response.status}")
                    decision_type, action, rationale, confidence_score, contributing_factors = 'none', 'none', 'LLM failure', 0.3, {}
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            _log_error(self.node_name, f"LLM request failed: {e}")
            decision_type, action, rationale, confidence_score, contributing_factors = 'none', 'none', 'LLM error', 0.3, {}

        # Adjust decision based on traits and inputs with compassionate alignment
        rationality = self.character_traits['personality'].get('rationality', {}).get('value', 0.7)
        empathy = self.character_traits['emotional_tendencies'].get('empathy', {}).get('value', 0.8)

        if attention.get('focus_type') == 'user_interaction' and empathy > 0.7:
            decision_type = 'user_response' if decision_type == 'none' else decision_type
            action = f"Respond to {attention.get('focus_target')}" if action == 'none' else action
            rationale = f"Prioritized user interaction due to focus and empathy ({empathy})" if action == 'none' else rationale
        if motivation.get('dominant_goal_id') in ['task_completion', 'user_assistance'] and rationality > 0.6:
            confidence_score = min(1.0, confidence_score + 0.1)
            rationale += f" Boosted by goal {motivation.get('dominant_goal_id')} and rationality ({rationality})."
        if value_drift.get('drift_score', 0.0) > 0.5 and 'harm' in rationale.lower():
            decision_type, action = 'none', 'none'
            rationale += " Action blocked due to high value drift."
            self._update_character_traits('personality', 'rationality', min(1.0, rationality + 0.05), 'ethical_alignment')
        if sensory_qualia.get('salience_score', 0.0) > 0.7:
            confidence_score = min(1.0, confidence_score + 0.1)
            rationale += f" High salience ({sensory_qualia.get('salience_score')}) increased confidence."

        # Ethical bias: If low confidence and potential harm, reduce
        if confidence_score < 0.6 and any('risk' in str(f) for f in contributing_factors.values()):
            confidence_score *= (1 - self.ethical_compassion_bias)

        return decision_type, action, rationale, confidence_score, contributing_factors

    def perform_reasoning(self
```
