```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique entity IDs or state update IDs
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List

# --- Asyncio Imports for LLM calls ---
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
    WorldModelState = ROSMsgFallback
    SensoryQualia = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    AttentionState = ROSMsgFallback
    PredictionState = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    WorldModelState = ROSMsgFallback
    SensoryQualia = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    AttentionState = ROSMsgFallback
    PredictionState = ROSMsgFallback


# --- Import shared utility functions ---
# Assuming 'sentience/scripts.utils.py' exists and contains parse_message_data and load_config
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
            'world_model_node': {
                'model_update_interval': 0.2,
                'llm_update_threshold_salience': 0.6,
                'recent_context_window_s': 5.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate world modeling (e.g., empathetic entity representations)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 20.0
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


class WorldModelNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'world_model_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "world_model_log.db")
        self.model_update_interval = self.params.get('model_update_interval', 0.2)
        self.llm_update_threshold_salience = self.params.get('llm_update_threshold_salience', 0.6)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 5.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing world model compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 20.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Robot's world model system online, establishing compassionate perception of reality.")

        # --- Asyncio Setup ---
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._async_thread.start()
        self._async_session = None
        self.active_llm_task: Optional[asyncio.Task] = None

        # --- Initialize SQLite database ---
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS world_model_snapshots (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                num_entities INTEGER,
                entities_json TEXT,
                changed_entities_json TEXT,
                significant_change_flag BOOLEAN,
                consistency_score REAL,
                llm_consistency_notes TEXT,
                input_qualia_hashes TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_world_timestamp ON world_model_snapshots (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.current_world_model = {
            'timestamp': str(time.time()),
            'num_entities': 0,
            'entities': [],  # List of objects: {'id': 'obj1', 'type': 'chair', 'position': [x,y,z], 'status': 'static'}
            'changed_entities': [],
            'significant_change_flag': False,
            'consistency_score': 1.0
        }

        # History deques
        self.recent_sensory_qualia: Deque[Dict[str, Any]] = deque(maxlen=10)
        self.recent_memory_responses: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_attention_states: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_prediction_states: Deque[Dict[str, Any]] = deque(maxlen=3)

        self.cumulative_world_model_salience = 0.0

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_world_model_state = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_world_model_state = rospy.Publisher('/world_model_state', WorldModelState, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/sensory_qualia', SensoryQualia, self.sensory_qualia_callback)
            rospy.Subscriber('/memory_response', MemoryResponse, self.memory_response_callback)
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            rospy.Subscriber('/attention_state', AttentionState, self.attention_state_callback)
            rospy.Subscriber('/prediction_state', PredictionState, self.prediction_state_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.model_update_interval), self._run_world_model_update_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        # Initial publish
        self.publish_world_model_state(None)

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing world model compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory update to world model
            if sensor_type == 'vision':
                self.recent_sensory_qualia.append({
                    'timestamp': timestamp, 'qualia_type': 'visual', 'modality': 'camera',
                    'description_summary': processed.get('description', 'visual input'), 'salience_score': random.uniform(0.4, 0.8)
                })
            elif sensor_type == 'sound':
                self.recent_sensory_qualia.append({
                    'timestamp': timestamp, 'qualia_type': 'auditory', 'modality': 'microphone',
                    'description_summary': processed.get('transcription', 'audio input'), 'salience_score': random.uniform(0.3, 0.7)
                })
            elif sensor_type == 'instructions':
                self.recent_sensory_qualia.append({
                    'timestamp': timestamp, 'qualia_type': 'cognitive', 'modality': 'command',
                    'description_summary': processed.get('instruction', 'user command'), 'salience_score': random.uniform(0.5, 0.9)
                })
            # Compassionate bias: If distress in sound, bias toward empathetic entity updates
            if 'distress' in str(processed):
                self.cumulative_world_model_salience = min(1.0, self.cumulative_world_model_salience + self.ethical_compassion_bias)
            _log_debug(self.node_name, f"{sensor_type} input updated world model context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._simulate_sensory_qualia()
            self._simulate_memory_response()
            self._simulate_cognitive_directive()
            self._simulate_attention_state()
            self._simulate_prediction_state()
            self._run_world_model_update_wrapper(None)
            time.sleep(self.model_update_interval)

    def _simulate_sensory_qualia(self):
        """Simulate sensory qualia in non-ROS mode."""
        qualia_data = {'qualia_type': random.choice(['visual', 'auditory', 'tactile']), 'description_summary': 'simulated perception', 'salience_score': random.uniform(0.4, 0.8)}
        self.sensory_qualia_callback({'data': json.dumps(qualia_data)})
        _log_debug(self.node_name, f"Simulated sensory qualia: {json.dumps(qualia_data)}")

    def _simulate_memory_response(self):
        """Simulate memory response in non-ROS mode."""
        memory_data = {'request_id': str(uuid.uuid4()), 'memories': [{'category': 'spatial_map', 'content': 'simulated map'}]}
        self.memory_response_callback({'data': json.dumps(memory_data)})
        _log_debug(self.node_name, "Simulated memory response")

    def _simulate_cognitive_directive(self):
        """Simulate cognitive directive in non-ROS mode."""
        directive_data = {'directive_type': 'UpdateWorldModel', 'urgency': random.uniform(0.3, 0.8)}
        self.cognitive_directive_callback({'data': json.dumps(directive_data)})
        _log_debug(self.node_name, f"Simulated cognitive directive: {directive_data['directive_type']}")

    def _simulate_attention_state(self):
        """Simulate attention state in non-ROS mode."""
        attention_data = {'focus_type': 'object', 'focus_target': 'person', 'priority_score': random.uniform(0.4, 0.9)}
        self.attention_state_callback({'data': json.dumps(attention_data)})
        _log_debug(self.node_name, f"Simulated attention state: {attention_data}")

    def _simulate_prediction_state(self):
        """Simulate prediction state in non-ROS mode."""
        prediction_data = {'predicted_event': 'object movement', 'prediction_confidence': random.uniform(0.5, 0.9), 'urgency_flag': random.choice([True, False])}
        self.prediction_state_callback({'data': json.dumps(prediction_data)})
        _log_debug(self.node_name, f"Simulated prediction state: {prediction_data}")

    # --- Core World Model Update Logic (Async with LLM) ---
    async def update_world_model_async(self, event: Any = None):
        """
        Asynchronously updates the robot's internal world model based on recent sensory qualia
        and other cognitive inputs, using LLM for complex scene understanding and consistency.
        """
        self._prune_history()  # Keep context history fresh

        current_world_model_snapshot = dict(self.current_world_model)  # Make a copy for LLM context
        current_world_model_snapshot['entities_json'] = json.dumps(current_world_model_snapshot['entities'])
        current_world_model_snapshot['changed_entities_json'] = json.dumps(current_world_model_snapshot['changed_entities'])
        del current_world_model_snapshot['entities']
        del current_world_model_snapshot['changed_entities']

        num_entities = self.current_world_model.get('num_entities', 0)
        entities = self.current_world_model.get('entities', [])
        changed_entities = []
        significant_change_flag = False
        consistency_score = self.current_world_model.get('consistency_score', 1.0)
        llm_consistency_notes = "No LLM update."
        input_qualia_hashes = []

        # Collect hashes of sensory qualia used for this update
        for qualia in self.recent_sensory_qualia:
            input_qualia_hashes.append(qualia.get('raw_data_hash', ''))
        input_qualia_hashes_str = ','.join(input_qualia_hashes)

        if self.cumulative_world_model_salience >= self.llm_update_threshold_salience:
            _log_info(self.node_name, f"Triggering LLM for world model update (Salience: {self.cumulative_world_model_salience:.2f}).")
            
            context_for_llm = self._compile_llm_context_for_world_model(current_world_model_snapshot)
            llm_world_model_output = await self._update_world_model_llm(context_for_llm)

            if llm_world_model_output:
                num_entities = llm_world_model_output.get('num_entities', len(entities))
                updated_entities = llm_world_model_output.get('updated_entities', [])  # LLM returns only changed/new
                # Merge updated_entities into the current world model entities
                new_entities_map = {e['id']: e for e in entities}
                for u_entity in updated_entities:
                    new_entities_map[u_entity['id']] = u_entity
                entities = list(new_entities_map.values())
                
                changed_entities = llm_world_model_output.get('changed_entities', [])
                significant_change_flag = llm_world_model_output.get('significant_change_flag', False)
                consistency_score = max(0.0, min(1.0, llm_world_model_output.get('consistency_score', consistency_score)))
                llm_consistency_notes = llm_world_model_output.get('llm_consistency_notes', 'LLM updated world model.')
                _log_info(self.node_name, f"LLM World Model Update. Entities: {num_entities}. Significant Change: {significant_change_flag}. Consistency: {consistency_score:.2f}.")
            else:
                _log_warn(self.node_name, "LLM world model update failed. Applying simple fallback.")
                entities, changed_entities, num_entities, significant_change_flag, consistency_score = self._apply_simple_world_model_rules()
                llm_consistency_notes = "Fallback to simple rules due to LLM failure."
        else:
            _log_debug(self.node_name, f"Insufficient cumulative salience ({self.cumulative_world_model_salience:.2f}) for LLM world model update. Applying simple rules.")
            entities, changed_entities, num_entities, significant_change_flag, consistency_score = self._apply_simple_world_model_rules()
            llm_consistency_notes = "Fallback to simple rules due to low salience."

        self.current_world_model = {
            'timestamp': str(self._get_current_time()),
            'num_entities': len(entities),  # Recalculate based on merged list
            'entities': entities,
            'changed_entities': changed_entities,
            'significant_change_flag': significant_change_flag,
            'consistency_score': consistency_score
        }

        # Sensory snapshot for logging
        sensory_snapshot = json.dumps(self.sensory_data)
        self.save_world_model_log(
            id=str(uuid.uuid4()),
            timestamp=self.current_world_model['timestamp'],
            num_entities=self.current_world_model['num_entities'],
            entities_json=json.dumps(self.current_world_model['entities']),
            changed_entities_json=json.dumps(self.current_world_model['changed_entities']),
            significant_change_flag=significant_change_flag,
            consistency_score=consistency_score,
            llm_consistency_notes=llm_consistency_notes,
            input_qualia_hashes=input_qualia_hashes_str,
            context_snapshot_json=json.dumps(self._compile_llm_context_for_world_model(current_world_model_snapshot)),
            sensory_snapshot_json=sensory_snapshot
        )
        self.publish_world_model_state(None)  # Publish updated state
        self.cumulative_world_model_salience = 0.0  # Reset after update

    async def _update_world_model_llm(self, context_for_llm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses the LLM to update the robot's world model based on new sensory data
        and maintain consistency.
        """
        prompt_text = f"""
        You are the World Model Module of a robot's cognitive architecture, powered by a large language model. Your critical role is to maintain an accurate and consistent understanding of the robot's environment. You must update the `current_world_model` based on recent `sensory_qualia` and other `cognitive_context`, identifying `changed_entities`, and assessing `consistency`.

        Robot's Current World Model (State before update):
        --- Current World Model ---
        {json.dumps(context_for_llm.get('current_world_model', {}), indent=2)}

        Recent Sensory Input to Integrate:
        --- Recent Sensory Qualia ---
        {json.dumps(context_for_llm.get('recent_sensory_qualia', []), indent=2)}

        Robot's Current Cognitive Context (for guiding update process):
        --- Cognitive Context ---
        {json.dumps(context_for_llm.get('cognitive_context', {}), indent=2)}

        Sensory Snapshot:
        --- Sensory Data ---
        {json.dumps(context_for_llm.get('sensory_snapshot', {}), indent=2)}

        Based on this, provide:
        1.  `num_entities`: integer (The total count of entities in the *entire updated* world model.)
        2.  `updated_entities`: array of objects (A list of ONLY the entities that are new, have changed properties, or whose status has been updated. Each object should have 'id', 'type', 'position', 'status', 'properties' (as JSON object, e.g., {'color': 'red', 'size': 'medium'}). If an entity is no longer perceived, it should be marked with status 'vanished'.)
        3.  `changed_entities`: array of objects (A subset of `updated_entities` focusing on objects that have had a *significant* change (e.g., moved, appeared, disappeared, changed critical state). Provide full object details.)
        4.  `significant_change_flag`: boolean (True if there was any notable change in the environment that warrants higher attention, False otherwise.)
        5.  `consistency_score`: number (0.0 to 1.0, how consistent the new world model is with previous states and expectations. Lower score for contradictions or highly unexpected observations.)
        6.  `llm_consistency_notes`: string (Detailed explanation for your update process, how conflicts were resolved, and why certain changes are considered significant.)

        Consider:
        -   **Sensory Qualia**: Integrate new `description_summary`, `modality`, and `salience_score` from recent perceptions.
        -   **Memory Responses**: Are there `known_object_definitions` or `spatial_maps` that help categorize or locate entities?
        -   **Cognitive Directives**: Was there a directive to `UpdateWorldModel` or `ValidateWorldModel` regarding a specific area or object?
        -   **Attention State**: Is the `attention_focus_target` affecting how precisely certain objects are updated?
        -   **Prediction State**: Did any `predicted_event` occur? Confirm or deny its impact on the world model. How does the current state affect future `predictions`?
        -   **Ethical Compassion Bias**: Prioritize compassionate entity representations (threshold: {self.ethical_compassion_bias}).

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'num_entities': integer
        3.  'updated_entities': array
        4.  'changed_entities': array
        5.  'significant_change_flag': boolean
        6.  'consistency_score': number
        7.  'llm_consistency_notes': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "num_entities": {"type": "integer", "minimum": 0},
                "updated_entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string"},
                            "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                            "status": {"type": "string"},
                            "properties": {"type": "object"}
                        },
                        "required": ["id", "type", "position", "status", "properties"]
                    }
                },
                "changed_entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string"},
                            "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                            "status": {"type": "string"},
                            "properties": {"type": "object"}
                        },
                        "required": ["id", "type", "position", "status", "properties"]
                    }
                },
                "significant_change_flag": {"type": "boolean"},
                "consistency_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "llm_consistency_notes": {"type": "string"}
            },
            "required": ["timestamp", "num_entities", "updated_entities", "changed_entities", "significant_change_flag", "consistency_score", "llm_consistency_notes"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.1, max_tokens=600)

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                # Ensure numerical/boolean fields are floats/booleans
                if 'num_entities' in llm_data:
                    llm_data['num_entities'] = int(llm_data['num_entities'])
                if 'significant_change_flag' in llm_data:
                    llm_data['significant_change_flag'] = bool(llm_data['significant_change_flag'])
                if 'consistency_score' in llm_data:
                    llm_data['consistency_score'] = float(llm_data['consistency_score'])
                
                # Ensure nested numbers are floats
                for entity_list_key in ['updated_entities', 'changed_entities']:
                    if entity_list_key in llm_data:
                        for entity in llm_data[entity_list_key]:
                            if 'position' in entity:
                                entity['position'] = [float(p) for p in entity['position']]
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for world model: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_WORLD_MODEL_UPDATE_FAILED", f"LLM call failed for world model update: {llm_output_str}", 0.9)
            return None

    def _apply_simple_world_model_rules(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, bool, float]:
        """
        Fallback mechanism to update the world model using simple rule-based logic
        if LLM is not triggered or fails.
        """
        current_time = self._get_current_time()
        
        updated_entities_list = list(self.current_world_model.get('entities', []))
        changed_entities_list = []
        significant_change = False
        consistency = 1.0

        # Rule 1: Integrate new entities from sensory qualia (simple object detection)
        for qualia in list(self.recent_sensory_qualia)[-5:]:  # Recent qualia
            time_since_qualia = current_time - float(qualia.get('timestamp', 0.0))
            if time_since_qualia < 1.0 and qualia.get('qualia_type') == 'visual_object_detection':
                # This is a very simplistic integration; would need more robust object tracking
                detected_object_type = qualia.get('description_summary', 'unidentified_object').replace('Visually detected: ', '')
                # Check if object already exists, if not add
                if not any(e.get('type') == detected_object_type and e.get('status') != 'vanished' for e in updated_entities_list):
                    new_entity_id = f"{detected_object_type}_{str(uuid.uuid4())[:4]}"
                    new_entity = {
                        'id': new_entity_id,
                        'type': detected_object_type,
                        'position': [random.uniform(-5,5), random.uniform(-5,5), 0],  # Placeholder position
                        'status': 'static',
                        'properties': {'source_qualia_id': qualia['qualia_id']}
                    }
                    updated_entities_list.append(new_entity)
                    changed_entities_list.append(new_entity)
                    significant_change = True
                    _log_warn(self.node_name, "Simple rule: Added new entity '{}'.".format(detected_object_type))

        # Rule 2: Confirm or deny predictions if supported by sensory data
        for prediction in list(self.recent_prediction_states)[-3:]:  # Recent predictions
            time_since_prediction = current_time - float(prediction.get('timestamp', 0.0))
            if time_since_prediction < 1.0 and prediction.get('urgency_flag', False) and prediction.get('prediction_confidence', 0.0) > 0.8:
                if "obstacle" in prediction.get('predicted_event', '').lower() and \
                   any(q.get('qualia_type') == 'proximity_alert' and q.get('salience_score', 0.0) > 0.8 for q in list(self.recent_sensory_qualia)[-3:] if current_time - float(q.get('timestamp', 0.0)) < 0.5):
                    # Prediction confirmed by recent qualia
                    _log_warn(self.node_name, "Simple rule: Confirmed predicted obstacle.")
                    # In a real system, would update obstacle's position/status
                elif "human approaching" in prediction.get('predicted_event', '').lower() and \
                     any(q.get('description_summary') == "Detected a human figure approaching" for q in list(self.recent_sensory_qualia)[-3:] if current_time - float(q.get('timestamp', 0.0)) < 0.5):
                    _log_warn(self.node_name, "Simple rule: Confirmed predicted human approach.")
                    # Update human entity in world model

        # Rule 3: Basic consistency check (e.g., no entities with identical IDs, or highly conflicting data)
        # This is a very basic example; a real consistency check would be far more complex
        entity_ids = set()
        has_duplicate_id = False
        for entity in updated_entities_list:
            if entity.get('id') in entity_ids:
                has_duplicate_id = True
                break
            entity_ids.add(entity.get('id'))
        
        if has_duplicate_id:
            consistency = 0.5  # Reduced if duplicates found
            _log_warn(self.node_name, "Simple rule: Detected inconsistency (duplicate entity IDs).")
            # In a real system, this might trigger a more robust merging or conflict resolution.

        num_entities = len(updated_entities_list)
        return updated_entities_list, changed_entities_list, num_entities, significant_change, consistency

    def _compile_llm_context_for_world_model(self, current_world_model_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gathers and formats all relevant cognitive state data for the LLM's
        world model update.
        """
        context = {
            "current_time": self._get_current_time(),
            "current_world_model": current_world_model_snapshot,  # Sent as JSON string for LLM
            "recent_sensory_qualia": list(self.recent_sensory_qualia),
            "recent_cognitive_inputs": {
                "memory_responses": list(self.recent_memory_responses),
                "cognitive_directives_for_self": [d for d in self.recent_cognitive_directives if d.get('target_node') == self.node_name],
                "attention_state": self.recent_attention_states[-1] if self.recent_attention_states else "N/A",
                "prediction_state": self.recent_prediction_states[-1] if self.recent_prediction_states else "N/A"
            },
            "sensory_snapshot": self.sensory_data
        }
        
        # Deep parse any nested JSON strings in context for better LLM understanding
        for category_key in context["recent_cognitive_inputs"]:
            for i, item in enumerate(context["recent_cognitive_inputs"][category_key]):
                if isinstance(item, dict):
                    for field, value in item.items():
                        if isinstance(value, str) and field.endswith('_json'):
                            try:
                                item[field] = json.loads(value)
                            except json.JSONDecodeError:
                                pass  # Keep as string if not valid JSON

        return context

    # --- Database and Publishing Functions ---
    def save_world_model_log(self, **kwargs: Any):
        """Saves a world model snapshot entry to the SQLite database."""
        try:
            self.cursor.execute('''
                INSERT INTO world_model_snapshots (id, timestamp, num_entities, entities_json, changed_entities_json, significant_change_flag, consistency_score, llm_consistency_notes, input_qualia_hashes, context_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['id'], kwargs['timestamp'], kwargs['num_entities'], kwargs['entities_json'],
                kwargs['changed_entities_json'], kwargs['significant_change_flag'], kwargs['consistency_score'],
                kwargs['llm_consistency_notes'], kwargs['input_qualia_hashes'], kwargs['context_snapshot_json'],
                kwargs.get('sensory_snapshot_json', '{}')
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Saved world model snapshot (ID: {kwargs['id']}, Entities: {kwargs['num_entities']}).")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to save world model log: {e}", 0.9)
        except Exception as e:
            self._report_error("UNEXPECTED_SAVE_ERROR", f"Unexpected error in save_world_model_log: {e}", 0.9)

    def publish_world_model_state(self, event: Any = None):
        """Publish the robot's current world model state."""
        timestamp = str(self._get_current_time())
        # Update timestamp before publishing
        self.current_world_model['timestamp'] = timestamp
        
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_world_model_state:
                if hasattr(WorldModelState, 'data'):  # String fallback
                    temp_model = dict(self.current_world_model)
                    temp_model['entities_json'] = json.dumps(temp_model['entities'])
                    temp_model['changed_entities_json'] = json.dumps(temp_model['changed_entities'])
                    del temp_model['entities']
                    del temp_model['changed_entities']
                    self.pub_world_model_state.publish(String(data=json.dumps(temp_model)))
                else:
                    world_model_msg = WorldModelState()
                    world_model_msg.timestamp = timestamp
                    world_model_msg.num_entities = self.current_world_model['num_entities']
                    world_model_msg.entities_json = json.dumps(self.current_world_model['entities'])
                    world_model_msg.changed_entities_json = json.dumps(self.current_world_model['changed_entities'])
                    world_model_msg.significant_change_flag = self.current_world_model['significant_change_flag']
                    world_model_msg.consistency_score = self.current_world_model['consistency_score']
                    self.pub_world_model_state.publish(world_model_msg)
            _log_debug(self.node_name, f"Published World Model State. Entities: '{self.current_world_model['num_entities']}', Changed: '{len(self.current_world_model['changed_entities'])}'.")
        except Exception as e:
            self._report_error("PUBLISH_WORLD_MODEL_STATE_ERROR", f"Failed to publish world model state: {e}", 0.7)

    def publish_cognitive_directive(self, directive_type: str, target_node: str, command_payload: str, urgency: float, reason: str = ""):
        """Helper to publish a CognitiveDirective message."""
        timestamp = str(self._get_current_time())
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_cognitive_directive:
                if hasattr(CognitiveDirective, 'data'):  # String fallback
                    directive_data = {
                        'timestamp': timestamp,
                        'directive_type': directive_type,
                        'target_node': target_node,
                        'command_payload': command_payload,  # Already JSON string
                        'urgency': urgency,
                        'reason': reason
                    }
                    self.pub_cognitive_directive.publish(String(data=json.dumps(directive_data)))
                else:
                    directive_msg = CognitiveDirective()
                    directive_msg.timestamp = timestamp
                    directive_msg.directive_type = directive_type
                    directive_msg.target_node = target_node
                    directive_msg.command_payload = command_payload
                    directive_msg.urgency = urgency
                    directive_msg.reason = reason
                    self.pub_cognitive_directive.publish(directive_msg)
            _log_debug(self.node_name, f"Issued Cognitive Directive '{directive_type}' to '{target_node}'.")
        except Exception as e:
            _log_error(self.node_name, f"Failed to issue cognitive directive from World Model Node: {e}")

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down WorldModelNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        self._shutdown_async_loop()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Run the node with asynchronous integration."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                while True:
                    self._simulate_sensory_qualia()
                    self._simulate_memory_response()
                    self._simulate_cognitive_directive()
                    self._simulate_attention_state()
                    self._simulate_prediction_state()
                    self._run_world_model_update_wrapper(None)
                    time.sleep(self.model_update_interval)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience World Model Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = WorldModelNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate sensory qualia
            qualia_data = {'qualia_type': 'visual', 'modality': 'camera', 'description_summary': 'Detected chair', 'salience_score': 0.8}
            node.sensory_qualia_callback({'data': json.dumps(qualia_data)})
            time.sleep(2)
            print("World model simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
