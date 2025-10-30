```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique bias event IDs
import sys
import argparse
from datetime import datetime
from collections import deque
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
    BiasMitigationState = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    InteractionRequest = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    ReflectionState = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    BiasMitigationState = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    InteractionRequest = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    ReflectionState = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback


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
            'bias_mitigation_node': {
                'mitigation_interval': 1.0,
                'llm_trigger_salience': 0.6,
                'recent_context_window_s': 20.0,
                'ethical_compassion_threshold': 0.4,  # Bias toward compassionate mitigation
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 30.0
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


class BiasMitigationNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'bias_mitigation_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "bias_log.db")
        self.mitigation_interval = self.params.get('mitigation_interval', 1.0)
        self.llm_trigger_salience = self.params.get('llm_trigger_salience', 0.6)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 20.0)
        self.ethical_compassion_threshold = self.params.get('ethical_compassion_threshold', 0.4)

        # Sensory placeholders
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 30.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Bias Mitigation Node online, fostering equitable and compassionate cognition.")

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
            CREATE TABLE IF NOT EXISTS bias_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                bias_type TEXT,
                detected_severity REAL,
                mitigation_status TEXT,
                llm_reasoning TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_bias_timestamp ON bias_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.current_bias_mitigation_state = {
            'timestamp': str(time.time()),
            'bias_type': 'none',
            'detected_severity': 0.0,
            'mitigation_status': 'idle'
        }

        # History deques
        self.recent_internal_narratives: deque = deque(maxlen=10)
        self.recent_interaction_requests: deque = deque(maxlen=10)
        self.recent_memory_responses: deque = deque(maxlen=5)
        self.recent_reflection_states: deque = deque(maxlen=5)
        self.recent_cognitive_directives: deque = deque(maxlen=3)

        self.cumulative_bias_salience = 0.0

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_bias_mitigation_state = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_bias_mitigation_state = rospy.Publisher('/bias_mitigation_state', BiasMitigationState, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/internal_narrative', InternalNarrative, self.internal_narrative_callback)
            rospy.Subscriber('/interaction_request', InteractionRequest, self.interaction_request_callback)
            rospy.Subscriber('/memory_response', MemoryResponse, self.memory_response_callback)
            rospy.Subscriber('/reflection_state', ReflectionState, self.reflection_state_callback)
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.mitigation_interval), self._run_bias_analysis_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        # Initial publish
        self.publish_bias_mitigation_state(None)

    def _create_sensory_placeholder(self, sensor_type: str):
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            if sensor_type == 'vision':
                self.recent_interaction_requests.append({'timestamp': timestamp, 'speech_text': processed.get('description', ''), 'urgency_score': random.uniform(0.1, 0.5)})
            elif sensor_type == 'sound':
                self.recent_interaction_requests.append({'timestamp': timestamp, 'speech_text': processed.get('transcription', ''), 'urgency_score': random.uniform(0.2, 0.6)})
            elif sensor_type == 'instructions':
                self.recent_cognitive_directives.append({'timestamp': timestamp, 'directive_type': 'user_input', 'command_payload': json.dumps(processed)})
            self._update_cumulative_salience(0.2)  # Sensory can trigger bias checks (e.g., biased perception)
            _log_debug(self.node_name, f"{sensor_type} input updated at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._run_bias_analysis_wrapper(None)
            time.sleep(self.mitigation_interval)

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

    def _run_bias_analysis_wrapper(self, event: Any = None):
        """Wrapper to run the async bias analysis from a ROS timer."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM bias analysis task already active. Skipping new cycle.")
            return
        
        # Schedule the async task
        self.active_llm_task = asyncio.run_coroutine_threadsafe(
            self.analyze_for_biases_async(event), self._async_loop
        )

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
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.3, max_tokens: int = 350) -> str:
        """
        Asynchronously calls the local LLM inference server (e.g., llama.cpp compatible API).
        Can optionally request a structured JSON response.
        """
        if not self._async_session:
            await self._create_async_session()
            if not self._async_session:
                self._report_error("LLM_SESSION_ERROR", "aiohttp session not available for LLM call.", 0.8)
                return "Error: LLM session not ready."

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,  # Low temperature for factual/reasoning tasks (bias detection)
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
                response.raise_for_status()
                result = await response.json()

                if result.get('choices') and result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
                    return result['choices'][0]['message']['content']
                
                self._report_error("LLM_RESPONSE_EMPTY", "LLM response had no content from local server.", 0.5, {'prompt_snippet': prompt_text[:100], 'raw_result': str(result)})
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
        """Accumulates salience from new inputs for triggering LLM analysis."""
        self.cumulative_bias_salience += score
        self.cumulative_bias_salience = min(1.0, self.cumulative_bias_salience)

    # --- Pruning old history ---
    def _prune_history(self):
        """Removes old entries from history deques based on recent_context_window_s."""
        current_time = self._get_current_time()
        for history_deque in [
            self.recent_internal_narratives, self.recent_interaction_requests,
            self.recent_memory_responses, self.recent_reflection_states,
            self.recent_cognitive_directives
        ]:
            while history_deque and (current_time - float(history_deque[0].get('timestamp', 0.0))) > self.recent_context_window_s:
                history_deque.popleft()

    # --- Callbacks (generic, ROS or direct) ---
    def internal_narrative_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'narrative_text': ('', 'narrative_text'),
            'main_theme': ('', 'main_theme'), 'sentiment': (0.0, 'sentiment'), 'salience_score': (0.0, 'salience_score')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_internal_narratives.append(data)
        # Narratives indicating strong internal convictions, quick conclusions, or emotional reasoning
        if data.get('sentiment', 0.0) > 0.5 and "conclusion" in data.get('main_theme', '').lower() or \
           data.get('sentiment', 0.0) < -0.5 and "problem" in data.get('main_theme', '').lower():
            self._update_cumulative_salience(data.get('salience_score', 0.0) * 0.4)
        _log_debug(self.node_name, f"Received Internal Narrative (Theme: {data.get('main_theme', 'N/A')}.)")

    def interaction_request_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'request_id': ('', 'request_id'),
            'request_type': ('', 'request_type'), 'user_id': ('unknown', 'user_id'),
            'command_payload': ('{}', 'command_payload'), 'urgency_score': (0.0, 'urgency_score'),
            'speech_text': ('', 'speech_text'), 'gesture_data_json': ('{}', 'gesture_data_json')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        if isinstance(data.get('command_payload'), str):
            try:
                data['command_payload'] = json.loads(data['command_payload'])
            except json.JSONDecodeError:
                data['command_payload'] = {}
        if isinstance(data.get('gesture_data_json'), str):
            try:
                data['gesture_data'] = json.loads(data['gesture_data_json'])
            except json.JSONDecodeError:
                data['gesture_data'] = {}
        
        self.recent_interaction_requests.append(data)
        # User input with strong opinions, leading questions, or confirmation-seeking
        if "force" in data.get('speech_text', '').lower() or "only option" in data.get('speech_text', '').lower():
            self._update_cumulative_salience(data.get('urgency_score', 0.0) * 0.6)
        _log_debug(self.node_name, f"Received Interaction Request (ID: {data.get('request_id', 'N/A')}.)")

    def memory_response_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'request_id': ('', 'request_id'),
            'response_code': (0, 'response_code'), 'memories_json': ('[]', 'memories_json')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        if isinstance(data.get('memories_json'), str):
            try:
                data['memories'] = json.loads(data['memories_json'])
            except json.JSONDecodeError:
                data['memories'] = []
        else:
            data['memories'] = []
        self.recent_memory_responses.append(data)
        # Memory retrieval that shows selective recall or over-reliance on certain past events
        if data.get('memories') and len(data['memories']) > 1 and \
           any('strong_preference' in mem.get('category', '') for mem in data['memories']):
            self._update_cumulative_salience(0.3)
        _log_debug(self.node_name, f"Received Memory Response for request ID: {data.get('request_id', 'N/A')}.")

    def reflection_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'reflection_text': ('', 'reflection_text'),
            'insight_type': ('none', 'insight_type'), 'consistency_score': (1.0, 'consistency_score')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_reflection_states.append(data)
        # Self-reflection that flags inconsistencies or potential errors in reasoning
        if data.get('consistency_score', 1.0) < 0.7:
            self._update_cumulative_salience(0.5 * (1.0 - data['consistency_score']))
        _log_debug(self.node_name, f"Received Reflection State (Insight Type: {data.get('insight_type', 'N/A')}.)")

    def cognitive_directive_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'directive_type': ('', 'directive_type'),
            'target_node': ('', 'target_node'), 'command_payload': ('{}', 'command_payload'),
            'urgency': (0.0, 'urgency'), 'reason': ('', 'reason')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        if data.get('target_node') == self.node_name:
            self.recent_cognitive_directives.append(data)  # Add directives for self to context
            # Directives to perform a bias audit or apply a specific mitigation strategy
            if data.get('directive_type') in ['AuditBias', 'ApplyMitigationStrategy']:
                self._update_cumulative_salience(data.get('urgency', 0.0) * 0.9)
            _log_info(self.node_name, f"Received directive for self: '{data.get('directive_type', 'N/A')}' (Payload: {data.get('command_payload', 'N/A')}.)")
        else:
            self.recent_cognitive_directives.append(data)  # Add all directives for general context
        _log_debug(self.node_name, "Cognitive Directive received for context/action.")

    # --- Core Bias Analysis Logic (Async with LLM) ---
    async def analyze_for_biases_async(self, event: Any = None):
        """
        Asynchronously analyzes recent cognitive data for signs of bias using LLM.
        """
        self._prune_history()  # Keep context history fresh

        if self.cumulative_bias_salience >= self.llm_trigger_salience:
            _log_info(self.node_name, f"Triggering LLM for bias analysis (Salience: {self.cumulative_bias_salience:.2f}).")
            
            context_for_llm = self._compile_llm_context_for_bias_analysis()
            llm_bias_output = await self._detect_and_mitigate_bias_llm(context_for_llm)

            if llm_bias_output:
                bias_event_id = str(uuid.uuid4())
                timestamp = llm_bias_output.get('timestamp', str(self._get_current_time()))
                bias_type = llm_bias_output.get('bias_type', 'none')
                detected_severity = max(0.0, min(1.0, llm_bias_output.get('detected_severity', 0.0)))
                mitigation_status = llm_bias_output.get('mitigation_status', 'idle')
                llm_reasoning = llm_bias_output.get('llm_reasoning', 'No reasoning.')
                recommended_directive = llm_bias_output.get('recommended_directive', None)

                self.current_bias_mitigation_state = {
                    'timestamp': timestamp,
                    'bias_type': bias_type,
                    'detected_severity': detected_severity,
                    'mitigation_status': mitigation_status
                }

                sensory_snapshot = json.dumps(self.sensory_data)
                self.save_bias_log(
                    id=bias_event_id,
                    timestamp=timestamp,
                    bias_type=bias_type,
                    detected_severity=detected_severity,
                    mitigation_status=mitigation_status,
                    llm_reasoning=llm_reasoning,
                    context_snapshot_json=json.dumps(context_for_llm),
                    sensory_snapshot_json=sensory_snapshot
                )
                self.publish_bias_mitigation_state(None)  # Publish updated state

                # If a directive is recommended, publish it for system-wide mitigation
                if recommended_directive:
                    self.publish_cognitive_directive(
                        directive_type=recommended_directive.get('directive_type', 'ConsiderAlternative'),
                        target_node=recommended_directive.get('target_node', 'CognitiveControl'),  # Often directs Cognitive Control
                        command_payload=json.dumps(recommended_directive.get('command_payload', {})),
                        urgency=recommended_directive.get('urgency', 0.7)
                    )
                _log_info(self.node_name, f"Bias Detection: '{bias_type}' (Severity: {detected_severity:.2f}, Status: '{mitigation_status}').")
                self.cumulative_bias_salience = 0.0  # Reset after LLM analysis
            else:
                _log_warn(self.node_name, "LLM failed to detect/mitigate bias. Applying simple fallback.")
                self._apply_simple_bias_rules()
        else:
            _log_debug(self.node_name, f"Insufficient cumulative salience ({self.cumulative_bias_salience:.2f}) for LLM bias analysis. Applying simple rules.")
            self._apply_simple_bias_rules()
        
        self.publish_bias_mitigation_state(None)  # Always publish state, even if updated by simple rules

    async def _detect_and_mitigate_bias_llm(self, context_for_llm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses the LLM to detect cognitive biases and suggest mitigation strategies.
        """
        prompt_text = f"""
        You are the Bias Mitigation Module of a robot's cognitive architecture. Your role is to identify potential cognitive biases in the robot's internal processes or interactions and propose strategies for mitigation. This ensures fair, objective, and ethical decision-making, aligned with compassionate principles.

        Robot's Recent Cognitive Context (for Bias Detection):
        --- Cognitive Context ---
        {json.dumps(context_for_llm, indent=2)}

        Sensory Snapshot:
        --- Sensory Data ---
        {json.dumps(context_for_llm.get('sensory_snapshot', {}), indent=2)}

        Based on this comprehensive context, analyze for cognitive biases and provide:
        1.  `bias_type`: string (The type of bias detected, e.g., 'confirmation_bias', 'anchoring_bias', 'automation_bias', 'affect_heuristic', 'none' if no significant bias).
        2.  `detected_severity`: number (0.0 to 1.0, how severe the detected bias is. 1.0 is highly severe).
        3.  `mitigation_status`: string ('detected', 'mitigation_recommended', 'mitigated', 'monitor', 'idle').
        4.  `llm_reasoning`: string (Detailed explanation for the bias detection and why a particular mitigation is suggested, referencing specific contextual inputs).
        5.  `recommended_directive`: object or null (If mitigation requires action from other nodes. Structured as { 'directive_type': string, 'target_node': string, 'command_payload': object, 'urgency': number }).

        Consider:
        -   **Internal Narratives**: Does the robot's self-talk show quick conclusions, ignoring conflicting data (confirmation bias)? Or an overly positive/negative framing (affect heuristic)?
        -   **Interaction Requests**: Is the robot overly compliant or dismissive based on user's social status (authority bias)? Or preferring user input that confirms its existing beliefs?
        -   **Memory Responses**: Is the robot retrieving only confirmatory memories, or over-relying on initial retrieved information (anchoring)?
        -   **Reflection States**: Has self-reflection flagged a specific `consistency_score` issue or `insight_type` related to flawed reasoning?
        -   **Cognitive Directives**: Are there explicit directives for *this node* ('BiasMitigationNode') like 'AuditBias' or 'ApplyMitigationStrategy'?
        -   **Sensory Inputs**: Do vision/sound/instructions suggest perceptual biases (e.g., selective attention to threats)?

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'bias_type': string
        3.  'detected_severity': number
        4.  'mitigation_status': string
        5.  'llm_reasoning': string
        6.  'recommended_directive': object or null
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "bias_type": {"type": "string"},
                "detected_severity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "mitigation_status": {"type": "string"},
                "llm_reasoning": {"type": "string"},
                "recommended_directive": {
                    "type": ["object", "null"],
                    "properties": {
                        "directive_type": {"type": "string"},
                        "target_node": {"type": "string"},
                        "command_payload": {"type": "object"},
                        "urgency": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    },
                    "required": ["directive_type", "target_node", "command_payload", "urgency"]
                }
            },
            "required": ["timestamp", "bias_type", "detected_severity", "mitigation_status", "llm_reasoning", "recommended_directive"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.3, max_tokens=400)  # Lower temp for more objective assessment

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                # Ensure numerical fields are floats
                if 'detected_severity' in llm_data:
                    llm_data['detected_severity'] = float(llm_data['detected_severity'])
                if llm_data.get('recommended_directive') and 'urgency' in llm_data['recommended_directive']:
                    llm_data['recommended_directive']['urgency'] = float(llm_data['recommended_directive']['urgency'])
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for bias mitigation: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_BIAS_ANALYSIS_FAILED", f"LLM call failed for bias mitigation: {llm_output_str}", 0.9)
            return None

    def _apply_simple_bias_rules(self):
        """
        Fallback mechanism to detect and mitigate bias using simple rule-based logic
        if LLM is not triggered or fails.
        """
        current_time = self._get_current_time()
        
        new_bias_type = "none"
        new_detected_severity = 0.0
        new_mitigation_status = "idle"

        # Rule 1: Simple confirmation bias check based on narrative and memory recall
        # If the robot's internal narrative strongly confirms a prior belief AND
        # recent memory recalls only support that belief, flag potential confirmation bias.
        if self.recent_internal_narratives and self.recent_memory_responses:
            latest_narrative = self.recent_internal_narratives[-1]
            latest_memory_response = self.recent_memory_responses[-1]

            if "confirms" in latest_narrative.get('narrative_text', '').lower() and \
               latest_narrative.get('sentiment', 0.0) > 0.7 and \
               latest_memory_response.get('response_code', 0) == 200 and \
               len(latest_memory_response.get('memories', [])) > 0:
                
                # Check if all retrieved memories align with the narrative's confirmation
                all_memories_confirm = True
                for mem in latest_memory_response['memories']:
                    # This is a very simplistic check; real NLP needed for deep analysis
                    if "contradict" in mem.get('content', '').lower() or "disagree" in mem.get('content', '').lower():
                        all_memories_confirm = False
                        break
                
                if all_memories_confirm:
                    new_bias_type = "confirmation_bias"
                    new_detected_severity = 0.6
                    new_mitigation_status = "detected"
                    _log_info(self.node_name, "Simple rule: Detected potential confirmation bias.")
                    # Suggest a directive to Cognitive Control to seek diverse info or re-evaluate
                    self.publish_cognitive_directive(
                        directive_type='ConsiderAlternative',
                        target_node='CognitiveControl',
                        command_payload=json.dumps({"reason": "Potential confirmation bias detected, seek disconfirming evidence."}),
                        urgency=0.7
                    )

        # Rule 2: Automation bias (over-reliance on system outputs)
        # If a user interaction expresses doubt about a robot's prior output, but the robot's internal
        # narrative expresses high confidence without re-evaluating.
        if self.recent_interaction_requests and self.recent_internal_narratives:
            latest_request = self.recent_interaction_requests[-1]
            latest_narrative = self.recent_internal_narratives[-1]
            
            if "doubt" in latest_request.get('speech_text', '').lower() and \
               latest_request.get('urgency_score', 0.0) > 0.5 and \
               "confident" in latest_narrative.get('narrative_text', '').lower() and \
               latest_narrative.get('salience_score', 0.0) > 0.4:
                
                new_bias_type = "automation_bias"
                new_detected_severity = 0.5
                new_mitigation_status = "detected"
                _log_info(self.node_name, "Simple rule: Detected potential automation bias.")
                # Suggest a directive to Cognitive Control to re-evaluate the previous output
                self.publish_cognitive_directive(
                    directive_type='ReEvaluateOutput',
                    target_node='CognitiveControl',
                    command_payload=json.dumps({"reason": "User expressed doubt, robot too confident without re-evaluation."}),
                    urgency=0.6
                )

        # Update current state based on simple rules
        self.current_bias_mitigation_state = {
            'timestamp': str(current_time),
            'bias_type': new_bias_type,
            'detected_severity': new_detected_severity,
            'mitigation_status': new_mitigation_status
        }
        _log_debug(self.node_name, f"Simple rule: Current Bias State: {new_bias_type}, Status: {new_mitigation_status}.")

    def _compile_llm_context_for_bias_analysis(self) -> Dict[str, Any]:
        """
        Gathers and formats all relevant cognitive state data for the LLM's
        bias detection and mitigation analysis.
        """
        context = {
            "current_time": self._get_current_time(),
            "current_bias_mitigation_state": self.current_bias_mitigation_state,
            "recent_cognitive_inputs": {
                "internal_narratives": list(self.recent_internal_narratives),
                "interaction_requests": list(self.recent_interaction_requests),
                "memory_responses": list(self.recent_memory_responses),
                "reflection_states": list(self.recent_reflection_states),
                "cognitive_directives_for_self": [d for d in self.recent_cognitive_directives if d.get('target_node') == self.node_name]
            },
            "sensory_snapshot": self.sensory_data
        }
        
        # Parse nested JSON
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
    def save_bias_log(self, **kwargs: Any):
        """Saves a bias mitigation state entry to the SQLite database."""
        try:
            self.cursor.execute('''
                INSERT INTO bias_log (id, timestamp, bias_type, detected_severity, mitigation_status, llm_reasoning, context_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['id'], kwargs['timestamp'], kwargs['bias_type'], kwargs['detected_severity'],
                kwargs['mitigation_status'], kwargs['llm_reasoning'], kwargs['context_snapshot_json'],
                kwargs.get('sensory_snapshot_json', '{}')
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Saved bias log (ID: {kwargs['id']}, Type: {kwargs['bias_type']}).")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to save bias log: {e}", 0.9)
        except Exception as e:
            self._report_error("UNEXPECTED_SAVE_ERROR", f"Unexpected error in save_bias_log: {e}", 0.9)

    def publish_bias_mitigation_state(self, event: Any = None):
        """Publishes the robot's current bias mitigation state."""
        timestamp = str(self._get_current_time())
        # Update timestamp before publishing
        self.current_bias_mitigation_state['timestamp'] = timestamp
        
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_bias_mitigation_state:
                if hasattr(BiasMitigationState, 'data'):  # String fallback
                    self.pub_bias_mitigation_state.publish(String(data=json.dumps(self.current_bias_mitigation_state)))
                else:
                    bias_msg = BiasMitigationState(**self.current_bias_mitigation_state)
                    self.pub_bias_mitigation_state.publish(bias_msg)
            _log_debug(self.node_name, f"Published Bias Mitigation State. Type: '{self.current_bias_mitigation_state['bias_type']}'.")
        except Exception as e:
            self._report_error("PUBLISH_BIAS_MITIGATION_STATE_ERROR", f"Failed to publish bias mitigation state: {e}", 0.7)

    def publish_cognitive_directive(self, directive_type: str, target_node: str, command_payload: str, urgency: float):
        """Helper to publish a CognitiveDirective message."""
        timestamp = str(self._get_current_time())
        directive_data = {
            'timestamp': timestamp,
            'directive_type': directive_type,
            'target_node': target_node,
            'command_payload': command_payload,
            'urgency': urgency
        }
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_cognitive_directive:
                if hasattr(CognitiveDirective, 'data'):  # String fallback
                    self.pub_cognitive_directive.publish(String(data=json.dumps(directive_data)))
                else:
                    directive_msg = CognitiveDirective(**directive_data)
                    self.pub_cognitive_directive.publish(directive_msg)
            _log_debug(self.node_name, f"Issued Cognitive Directive '{directive_type}' to '{target_node}'.")
        except Exception as e:
            self._report_error("DIRECTIVE_ISSUE_ERROR", f"Failed to issue cognitive directive from Bias Mitigation Node: {e}", 0.7)

    def shutdown(self):
        self._shutdown_flag.set() if hasattr(self, '_shutdown_flag') else None
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        self._shutdown_async_loop()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
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
                _log_info(self.node_name, "Shutdown requested.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Bias Mitigation Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = BiasMitigationNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
