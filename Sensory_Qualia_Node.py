```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique qualia IDs
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque

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
    SensoryQualia = ROSMsgFallback
    RawSensorData = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    AttentionState = ROSMsgFallback
    WorldModelState = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    SensoryQualia = ROSMsgFallback
    RawSensorData = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    AttentionState = ROSMsgFallback
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
            'sensory_qualia_node': {
                'processing_interval': 0.1,
                'llm_interpretation_threshold_salience': 0.7,
                'recent_context_window_s': 5.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate qualia (e.g., empathetic interpretations)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 15.0
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


class SensoryQualiaNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'sensory_qualia_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "sensory_log.db")
        self.processing_interval = self.params.get('processing_interval', 0.1)
        self.llm_interpretation_threshold_salience = self.params.get('llm_interpretation_threshold_salience', 0.7)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 5.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (for dynamic simulation)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 15.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Robot's sensory qualia system online, interpreting perceptions with compassionate nuance.")

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
            CREATE TABLE IF NOT EXISTS sensory_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                qualia_type TEXT,
                modality TEXT,
                description_summary TEXT,
                salience_score REAL,
                llm_interpretation_notes TEXT,
                raw_data_hash TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensory_timestamp ON sensory_log (timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensory_modality ON sensory_log (modality)')
        self.conn.commit()

        # --- Internal State ---
        self.raw_sensor_data_queue: Deque[Dict[str, Any]] = deque()

        # History deques
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_attention_states: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_world_model_states: Deque[Dict[str, Any]] = deque(maxlen=3)

        self.cumulative_sensory_salience = 0.0

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_sensory_qualia = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_sensory_qualia = rospy.Publisher('/sensory_qualia', SensoryQualia, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/raw_sensor_data', RawSensorData, self.raw_sensor_data_callback)
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            rospy.Subscriber('/attention_state', AttentionState, self.attention_state_callback)
            rospy.Subscriber('/world_model_state', WorldModelState, self.world_model_state_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.processing_interval), self._run_sensory_processing_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing qualia compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on qualia
            if sensor_type == 'vision':
                self.raw_sensor_data_queue.append({
                    'timestamp': timestamp, 'modality': 'camera', 'raw_data_json': json.dumps(processed),
                    'urgency': random.uniform(0.3, 0.8), 'data_hash': str(uuid.uuid4())
                })
            elif sensor_type == 'sound':
                self.raw_sensor_data_queue.append({
                    'timestamp': timestamp, 'modality': 'microphone', 'raw_data_json': json.dumps({'transcription': processed.get('transcription', 'audio input')}),
                    'urgency': random.uniform(0.4, 0.9), 'data_hash': str(uuid.uuid4())
                })
            elif sensor_type == 'instructions':
                self.raw_sensor_data_queue.append({
                    'timestamp': timestamp, 'modality': 'command', 'raw_data_json': json.dumps({'instruction': processed.get('instruction', 'user command')}),
                    'urgency': random.uniform(0.5, 0.9), 'data_hash': str(uuid.uuid4())
                })
            # Compassionate bias: If distress in sound, boost urgency for empathetic qualia
            if 'distress' in str(processed):
                self.cumulative_sensory_salience = min(1.0, self.cumulative_sensory_salience + self.ethical_compassion_bias)
            _log_debug(self.node_name, f"{sensor_type} input queued for qualia processing at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._run_sensory_processing_wrapper(None)
            time.sleep(self.processing_interval)

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    # --- Asyncio Thread Management ---
    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_until_complete(self._create_async_session())
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

    def _run_sensory_processing_wrapper(self, event: Any = None):
        """Wrapper to run the async sensory processing from a timer."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM sensory processing task already active. Skipping new cycle.")
            return

        if self.raw_sensor_data_queue:
            raw_data = self.raw_sensor_data_queue.popleft()
            self.active_llm_task = asyncio.run_coroutine_threadsafe(
                self.process_raw_sensor_data_async(raw_data, event), self._async_loop
            )
        else:
            _log_debug(self.node_name, "No raw sensor data in queue.")

    # --- Error Reporting Utility ---
    def _report_error(self, error_type: str, description: str, severity: float = 0.5, context: Optional[Dict] = None):
        timestamp = str(self._get_current_time())
        error_msg_data = {
            'timestamp': timestamp, 'source_node': self.node_name, 'error_type': error_type,
            'description': description, 'severity': severity, 'context': context or {}
        }
        if ROS_AVAILABLE and self.ros_enabled and self.pub_error_report:
            try:
                self.pub_error_report.publish(String(data=json.dumps(error_msg_data)))
                rospy.logerr(f"{self.node_name}: REPORTED ERROR: {error_type} - {description}")
            except Exception as e:
                _log_error(self.node_name, f"Failed to publish error report: {e}")
        else:
            _log_error(self.node_name, f"REPORTED ERROR: {error_type} - {description} (Severity: {severity})")

    # --- LLM Call Function ---
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.3, max_tokens: int = 200) -> str:
        """
        Asynchronously calls the local LLM inference server (e.g., llama.cpp compatible API).
        Can optionally request a structured JSON response. Low temperature for factual interpretation.
        """
        if not self._async_session:
            await self._create_async_session()
            if not self._async_session:
                self._report_error("LLM_SESSION_ERROR", "aiohttp session not available for LLM call.", 0.8)
                return "Error: LLM session not ready."

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,  # Low temperature for factual interpretation of sensory data
            "max_tokens": max_tokens,
            "stream": False
        }
        headers = {'Content-Type': 'application/json'}

        if response_schema:
            prompt_text += "\n\nProvide the response in JSON format according to this schema:\n" + json.dumps(response_schema, indent=2)
            payload["messages"] = [{"role": "user", "content": prompt_text}]

        api_url = self.llm_base_url

        try:
            async with self._async_session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=self.llm_timeout), headers=headers) as response:
                response.raise_for_status()  # Raise an exception for bad status codes
                result = await response.json()

                if result.get('choices') and result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
                    return result['choices'][0]['message']['content']
                
                self._report_error("LLM_RESPONSE_EMPTY", "LLM response had no content from local server.", 0.5, {'prompt_snippet': prompt_text[:100]})
                return "Error: LLM response empty."
        except aiohttp.ClientError as e:
            self._report_error("LLM_API_ERROR", f"LLM API request failed (aiohttp ClientError to local server): {e}", 0.9, {'url': api_url})
            return f"Error: LLM API request failed: {e}"
        except asyncio.TimeoutError:
            self._report_error("LLM_TIMEOUT", f"LLM API request timed out after {self.llm_timeout} seconds (local server).", 0.8, {'prompt_snippet': prompt_text[:100]})
            return "Error: LLM API request timed out."
        except json.JSONDecodeError:
            self._report_error("LLM_JSON_PARSE_ERROR", "Failed to parse local LLM response JSON.", 0.7)
            return "Error: Failed to parse LLM response."
        except Exception as e:
            self._report_error("UNEXPECTED_LLM_ERROR", f"An unexpected error occurred during local LLM call: {e}", 0.9, {'prompt_snippet': prompt_text[:100]})
            return f"Error: An unexpected error occurred: {e}"

    # --- Utility to accumulate input salience ---
    def _update_cumulative_salience(self, score: float):
        """Accumulates salience from new inputs for triggering LLM interpretation."""
        self.cumulative_sensory_salience += score
        self.cumulative_sensory_salience = min(1.0, self.cumulative_sensory_salience)  # Clamp at 1.0

    # --- Pruning old history ---
    def _prune_history(self):
        """Removes old entries from history deques based on recent_context_window_s."""
        current_time = self._get_current_time()
        for history_deque in [
            self.recent_cognitive_directives, self.recent_attention_states,
            self.recent_world_model_states
        ]:
            while history_deque and (current_time - float(history_deque[0].get('timestamp', 0.0))) > self.recent_context_window_s:
                history_deque.popleft()

    # --- Callbacks (generic, ROS or direct) ---
    def raw_sensor_data_callback(self, msg: Any):
        """Handle incoming raw sensor data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'sensor_id': ('', 'sensor_id'),
            'modality': ('unknown', 'modality'), 'raw_data_hash': ('{}', 'raw_data_hash'),
            'data_hash': ('', 'data_hash'), 'urgency': (0.0, 'urgency')  # Urgency for this specific raw data
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        # Parse raw_data_hash if it's a string
        if isinstance(data.get('raw_data_hash'), str):
            try:
                data['raw_data_parsed'] = json.loads(data['raw_data_hash'])
            except json.JSONDecodeError:
                data['raw_data_parsed'] = {}  # Fallback if not valid JSON
        else:
            data['raw_data_parsed'] = data.get('raw_data_hash', {})  # Ensure it's a dict

        self.raw_sensor_data_queue.append(data)
        # Salience of the raw sensor data influences LLM trigger
        self._update_cumulative_salience(data.get('urgency', 0.0) * 0.8)  # High urgency for direct sensor data
        _log_debug(self.node_name, f"Queued raw sensor data (Modality: {data['modality']}, Sensor: {data['sensor_id']}). Queue size: {len(self.raw_sensor_data_queue)}.")

    def cognitive_directive_callback(self, msg: Any):
        """Handle incoming cognitive directives."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'directive_type': ('', 'directive_type'),
            'target_node': ('', 'target_node'), 'command_payload': ('{}', 'command_payload'),
            'urgency': (0.0, 'urgency'), 'reason': ('', 'reason')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        if data.get('target_node') == self.node_name and data.get('directive_type') == 'FocusSensoryProcessing':
            try:
                payload = json.loads(data.get('command_payload', '{}'))
                # This directive doesn't go into a queue, it directly influences the next processing cycle
                self._update_cumulative_salience(data.get('urgency', 0.0) * 0.9)  # High urgency for focus directives
                _log_info(self.node_name, f"Received directive to focus sensory processing on reason: '{data.get('reason', 'N/A')}'.")
            except json.JSONDecodeError as e:
                self._report_error("JSON_DECODE_ERROR", f"Failed to decode command_payload in CognitiveDirective: {e}", 0.5, {'payload': data.get('command_payload')})
            except Exception as e:
                self._report_error("DIRECTIVE_PROCESSING_ERROR", f"Error processing CognitiveDirective for sensory: {e}", 0.7, {'directive': data})
        
        self.recent_cognitive_directives.append(data)  # Store all directives for context
        _log_debug(self.node_name, "Cognitive Directive received for context/action.")

    def attention_state_callback(self, msg: Any):
        """Handle incoming attention state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'focus_type': ('idle', 'focus_type'),
            'focus_target': ('environment', 'focus_target'), 'priority_score': (0.0, 'priority_score')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_attention_states.append(data)
        # What the robot is attending to influences what sensory input is considered salient
        if data.get('priority_score', 0.0) > 0.5:
            self._update_cumulative_salience(data.get('priority_score', 0.0) * 0.3)
        _log_debug(self.node_name, f"Received Attention State. Focus: {data.get('focus_target', 'N/A')}.")

    def world_model_state_callback(self, msg: Any):
        """Handle incoming world model state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'num_entities': (0, 'num_entities'),
            'changed_entities_json': ('[]', 'changed_entities_json'),
            'significant_change_flag': (False, 'significant_change_flag'),
            'consistency_score': (1.0, 'consistency_score')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        if isinstance(data.get('changed_entities_json'), str):
            try:
                data['changed_entities'] = json.loads(data['changed_entities_json'])
            except json.JSONDecodeError:
                data['changed_entities'] = []
        self.recent_world_model_states.append(data)
        # World model provides context for interpreting sensory data (e.g., expecting a human in a certain area)
        if data.get('significant_change_flag', False):
            self._update_cumulative_salience(0.2)
        _log_debug(self.node_name, f"Received World Model State. Significant Change: {data.get('significant_change_flag', False)}.")

    # --- Core Sensory Processing Logic (Async with LLM) ---
    async def process_raw_sensor_data_async(self, raw_data: Dict[str, Any], event: Any = None):
        """
        Asynchronously processes raw sensor data to extract 'qualia' (meaningful, salient perceptions),
        using LLM for higher-level interpretation with compassionate bias for empathetic sensory nuances.
        """
        self._prune_history()  # Keep context history fresh

        qualia_id = str(uuid.uuid4())
        timestamp = raw_data.get('timestamp', str(self._get_current_time()))
        modality = raw_data.get('modality', 'unknown')
        raw_data_hash = raw_data.get('data_hash', '')

        qualia_type = 'unspecified_perception'
        description_summary = "Raw data processed."
        salience_score = 0.0  # Default low, will be updated
        llm_interpretation_notes = "No LLM interpretation."

        if self.cumulative_sensory_salience >= self.llm_interpretation_threshold_salience:
            _log_info(self.node_name, f"Triggering LLM for sensory interpretation (Modality: {modality}, Salience: {self.cumulative_sensory_salience:.2f}).")
            
            context_for_llm = self._compile_llm_context_for_sensory_interpretation(raw_data)
            llm_qualia_output = await self._interpret_sensory_data_llm(raw_data['raw_data_parsed'], modality, context_for_llm)

            if llm_qualia_output:
                qualia_type = llm_qualia_output.get('qualia_type', 'unspecified_perception')
                description_summary = llm_qualia_output.get('description_summary', 'No summary.')
                salience_score = max(0.0, min(1.0, llm_qualia_output.get('salience_score', 0.0)))
                llm_interpretation_notes = llm_qualia_output.get('llm_interpretation_notes', 'LLM interpreted sensory data.')
                _log_info(self.node_name, f"LLM Interpreted Qualia: '{description_summary}' (Salience: {salience_score:.2f}).")
            else:
                _log_warn(self.node_name, "LLM sensory interpretation failed. Applying simple fallback.")
                qualia_type, description_summary, salience_score = self._apply_simple_sensory_rules(raw_data)
                llm_interpretation_notes = "Fallback to simple rules due to LLM failure."
        else:
            _log_debug(self.node_name, f"Insufficient cumulative salience ({self.cumulative_sensory_salience:.2f}) for LLM sensory interpretation. Applying simple rules.")
            qualia_type, description_summary, salience_score = self._apply_simple_sensory_rules(raw_data)
            llm_interpretation_notes = "Fallback to simple rules due to low salience."

        # Update salience_score based on the raw data's urgency if it was not high enough for LLM
        # This ensures high urgency raw data still gets a decent salience_score even without LLM
        salience_score = max(salience_score, raw_data.get('urgency', 0.0))

        # Compassionate bias: Boost salience for empathetic sensory inputs (e.g., social cues)
        if 'social' in modality.lower() or 'voice' in content.lower():
            salience_score = min(1.0, salience_score + self.ethical_compassion_bias * 0.1)

        self.save_sensory_log(
            id=qualia_id,
            timestamp=timestamp,
            qualia_type=qualia_type,
            modality=modality,
            description_summary=description_summary,
            salience_score=salience_score,
            llm_interpretation_notes=llm_interpretation_notes,
            raw_data_hash=raw_data_hash,
            context_snapshot_json=json.dumps(self._compile_llm_context_for_sensory_interpretation(raw_data)),
            sensory_snapshot_json=json.dumps(self.sensory_data)
        )
        self.publish_sensory_qualia(
            timestamp=timestamp,
            qualia_id=qualia_id,
            qualia_type=qualia_type,
            modality=modality,
            description_summary=description_summary,
            salience_score=salience_score,
            raw_data_hash=raw_data_hash
        )
        self.cumulative_sensory_salience = 0.0  # Reset after processing

    async def _interpret_sensory_data_llm(self, raw_data_parsed: Dict[str, Any], modality: str, context_for_llm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses the LLM to interpret raw sensor data into meaningful sensory qualia with compassionate bias for empathetic interpretations.
        """
        prompt_text = f"""
        You are the Sensory Qualia Module of a robot's cognitive architecture, powered by a large language model. Your role is to interpret `raw_sensor_data` from a specific `modality` into high-level `SensoryQualia`. This involves describing the perception, categorizing its `qualia_type`, and assigning a `salience_score` based on its importance and current `cognitive_context`, with compassionate bias for empathetic/social nuances.

        Raw Sensor Data:
        --- Raw Data (from {modality} sensor) ---
        {json.dumps(raw_data_parsed, indent=2)}

        Robot's Current Cognitive Context (for interpreting sensory data):
        --- Cognitive Context ---
        {json.dumps(context_for_llm, indent=2)}

        Sensory Snapshot:
        --- Sensory Data ---
        {json.dumps(context_for_llm.get('sensory_snapshot', {}), indent=2)}

        Based on this, provide:
        1.  `qualia_type`: string (The type of perception, e.g., 'visual_perception', 'auditory_stimulus', 'tactile_sensation', 'temperature_change', 'proximity_detection'.)
        2.  `description_summary`: string (A concise, human-readable summary of the sensory experience, e.g., "Detected a human figure approaching", "Heard a loud bang", "Felt a warm surface".)
        3.  `salience_score`: number (0.0 to 1.0, indicating how attention-grabbing or important this sensory input is. Higher score for unexpected, urgent, or goal-relevant perceptions.)
        4.  `llm_interpretation_notes`: string (Brief notes on your interpretation process and why certain details were highlighted, with compassionate nuance.)

        Consider:
        -   **Raw Data**: What are the key features in the raw data? (e.g., for camera: presence of objects, colors, movement; for audio: loudness, frequency, speech; for lidar: distance, obstacles).
        -   **Modality**: How does the modality influence interpretation?
        -   **Cognitive Directives**: Was there a directive to `FocusSensoryProcessing` on something specific (e.g., "look for red objects")?
        -   **Attention State**: What is the robot's current `focus_target` and `priority_score`? Does this sensory input align with it, increasing its salience?
        -   **World Model State**: Does this sensory data confirm or contradict the current `world_model_state`? Is it a `significant_change_flag`?
        -   **Ethical Compassion Bias**: Prioritize compassionate, empathetic interpretations (threshold: {self.ethical_compassion_bias}).

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'qualia_type': string
        3.  'description_summary': string
        4.  'salience_score': number
        5.  'llm_interpretation_notes': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "qualia_type": {"type": "string"},
                "description_summary": {"type": "string"},
                "salience_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "llm_interpretation_notes": {"type": "string"}
            },
            "required": ["timestamp", "qualia_type", "description_summary", "salience_score", "llm_interpretation_notes"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.3, max_tokens=250)

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                # Ensure numerical fields are floats
                if 'salience_score' in llm_data:
                    llm_data['salience_score'] = float(llm_data['salience_score'])
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for sensory qualia: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_SENSORY_INTERPRETATION_FAILED", f"LLM call failed for sensory interpretation: {llm_output_str}", 0.9)
            return None

    def _apply_simple_sensory_rules(self, raw_data: Dict[str, Any]) -> tuple[str, str, float]:
        """
        Fallback mechanism to process raw sensor data into simple qualia using rule-based logic
        if LLM is not triggered or fails.
        """
        modality = raw_data.get('modality', 'unknown')
        raw_data_parsed = raw_data.get('raw_data_parsed', {})
        urgency = raw_data.get('urgency', 0.0)
        
        qualia_type = 'generic_perception'
        description_summary = f"Processed raw data from {modality} sensor."
        salience_score = urgency * 0.5  # Base salience on urgency

        # Example simple rules based on modality and hypothetical content
        if modality == 'camera':
            if 'object_detected' in raw_data_parsed and raw_data_parsed['object_detected']:
                qualia_type = 'visual_object_detection'
                description_summary = f"Visually detected: {raw_data_parsed.get('object_type', 'an object')}."
                salience_score = max(salience_score, 0.4)
            elif 'motion_detected' in raw_data_parsed and raw_data_parsed['motion_detected']:
                qualia_type = 'visual_motion'
                description_summary = "Detected visual motion."
                salience_score = max(salience_score, 0.3)
        elif modality == 'microphone':
            if 'sound_level' in raw_data_parsed and raw_data_parsed['sound_level'] > 70:  # dB threshold
                qualia_type = 'auditory_loud_sound'
                description_summary = "Heard a loud sound."
                salience_score = max(salience_score, 0.6)
            elif 'speech_detected' in raw_data_parsed and raw_data_parsed['speech_detected']:
                qualia_type = 'auditory_speech'
                description_summary = "Detected human speech."
                salience_score = max(salience_score, 0.5)
        elif modality == 'lidar':
            if 'closest_distance' in raw_data_parsed and raw_data_parsed['closest_distance'] < 0.5:  # meters
                qualia_type = 'proximity_alert'
                description_summary = "Obstacle detected very close."
                salience_score = max(salience_score, 0.8)

        # Compassionate bias: Boost salience for empathetic/social sensory inputs
        if 'social' in modality.lower() or 'voice' in str(raw_data_parsed).lower():
            salience_score = min(1.0, salience_score + self.ethical_compassion_bias * 0.1)

        _log_warn(self.node_name, f"Simple rule: Generated fallback qualia for '{modality}'. Summary: {description_summary}.")
        return qualia_type, description_summary, salience_score

    def _compile_llm_context_for_sensory_interpretation(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gathers and formats all relevant cognitive state data for the LLM's
        sensory interpretation.
        """
        context = {
            "current_time": self._get_current_time(),
            "raw_sensor_data_source": {
                "timestamp": raw_data.get('timestamp'),
                "sensor_id": raw_data.get('sensor_id'),
                "modality": raw_data.get('modality'),
                "urgency_from_sensor": raw_data.get('urgency', 0.0)
            },
            "recent_cognitive_inputs": {
                "cognitive_directives_for_self": [d for d in self.recent_cognitive_directives if d.get('target_node') == self.node_name],
                "attention_state": self.recent_attention_states[-1] if self.recent_attention_states else "N/A",
                "world_model_state": self.recent_world_model_states[-1] if self.recent_world_model_states else "N/A"
            },
            "sensory_snapshot": self.sensory_data
        }
        
        # Deep parse any nested JSON strings in context for better LLM understanding
        for category_key in context["recent_cognitive_inputs"]:
            for item in context["recent_cognitive_inputs"][category_key]:
                if isinstance(item, dict):
                    for field, value in list(item.items()):
                        if isinstance(value, str) and field.endswith('_json'):
                            try:
                                item[field] = json.loads(value)
                            except json.JSONDecodeError:
                                pass  # Keep as string if not valid JSON
        return context

    # --- Database and Publishing Functions ---
    def save_sensory_log(self, **kwargs: Any):
        """Saves a sensory qualia entry to the SQLite database."""
        try:
            self.cursor.execute('''
                INSERT INTO sensory_log (id, timestamp, qualia_type, modality, description_summary, salience_score, llm_interpretation_notes, raw_data_hash, context_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['id'], kwargs['timestamp'], kwargs['qualia_type'], kwargs['modality'],
                kwargs['description_summary'], kwargs['salience_score'], kwargs['llm_interpretation_notes'],
                kwargs['raw_data_hash'], kwargs['context_snapshot_json'], kwargs.get('sensory_snapshot_json', '{}')
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Saved sensory log (ID: {kwargs['id']}, Type: {kwargs['qualia_type']}).")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to save sensory log: {e}", 0.9)
        except Exception as e:
            self._report_error("UNEXPECTED_SAVE_ERROR", f"Unexpected error in save_sensory_log: {e}", 0.9)

    def publish_sensory_qualia(self, timestamp: str, qualia_id: str, qualia_type: str, modality: str, description_summary: str, salience_score: float, raw_data_hash: str):
        """Publishes the processed sensory qualia."""
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
            _log_debug(self.node_name, f"Published Sensory Qualia. Type: '{qualia_type}', Summary: '{description_summary}'.")
        except Exception as e:
            self._report_error("PUBLISH_SENSORY_QUALIA_ERROR", f"Failed to publish sensory qualia for '{modality}': {e}", 0.7)

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
            _log_error(self.node_name, f"Failed to issue cognitive directive from Sensory Qualia Node: {e}")

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down SensoryQualiaNode.")
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
                    self._run_sensory_processing_wrapper(None)
                    time.sleep(1)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Sensory Qualia Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = SensoryQualiaNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate raw sensor data
            raw_data = {'modality': 'camera', 'raw_data_hash': json.dumps({'object': 'person'}), 'urgency': 0.8}
            node.raw_sensor_data_callback(raw_data)
            time.sleep(2)
            print("Sensory qualia simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
