```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique data mining task IDs
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
    DataMiningResult = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    ReflectionState = ROSMsgFallback
    WorldModelState = ROSMsgFallback
    PerformanceReport = ROSMsgFallback
    BiasMitigationState = ROSMsgFallback
    EthicalDecision = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    DataMiningResult = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    ReflectionState = ROSMsgFallback
    WorldModelState = ROSMsgFallback
    PerformanceReport = ROSMsgFallback
    BiasMitigationState = ROSMsgFallback
    EthicalDecision = ROSMsgFallback


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
            'data_mining_node': {
                'mining_interval': 2.0,
                'llm_analysis_threshold_salience': 0.5,
                'recent_context_window_s': 30.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate data insights
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 60.0
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


class DataMiningNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'data_mining_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "data_mining_log.db")
        self.mining_interval = self.params.get('mining_interval', 2.0)
        self.llm_analysis_threshold_salience = self.params.get('llm_analysis_threshold_salience', 0.5)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 30.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing data mining compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 60.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Robot's data mining system online, extracting compassionate and mindful insights.")

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
            CREATE TABLE IF NOT EXISTS data_mining_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                analysis_type TEXT,
                query_parameters_json TEXT,
                insights_summary TEXT,
                extracted_data_json TEXT,
                llm_analysis_reasoning TEXT,
                raw_data_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_mining_timestamp ON data_mining_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.data_mining_tasks_queue: Deque[Dict[str, Any]] = deque()

        # History deques
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_memory_responses: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_reflection_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_world_model_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_performance_reports: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_bias_mitigation_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_ethical_decisions: Deque[Dict[str, Any]] = deque(maxlen=5)

        self.cumulative_mining_salience = 0.0

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_data_mining_result = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_data_mining_result = rospy.Publisher('/data_mining_result', DataMiningResult, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            rospy.Subscriber('/memory_response', MemoryResponse, self.memory_response_callback)
            rospy.Subscriber('/reflection_state', ReflectionState, self.reflection_state_callback)
            rospy.Subscriber('/world_model_state', WorldModelState, self.world_model_state_callback)
            rospy.Subscriber('/performance_report', PerformanceReport, self.performance_report_callback)
            rospy.Subscriber('/bias_mitigation_state', BiasMitigationState, self.bias_mitigation_state_callback)
            rospy.Subscriber('/ethical_decision', EthicalDecision, self.ethical_decision_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.mining_interval), self._run_data_mining_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on data mining (e.g., vision detects anomalies)
            if sensor_type == 'vision':
                self.recent_world_model_states.append({'timestamp': timestamp, 'num_entities': random.randint(5, 20), 'significant_change_flag': random.random() < 0.3})
            elif sensor_type == 'sound':
                self.recent_performance_reports.append({'timestamp': timestamp, 'overall_score': random.uniform(0.6, 0.9), 'suboptimal_flag': random.random() < 0.4})
            elif sensor_type == 'instructions':
                self.recent_cognitive_directives.append({'timestamp': timestamp, 'directive_type': 'data_mining', 'command_payload': json.dumps({'analysis_type': 'trend_analysis'})})
            self._update_cumulative_salience(0.2)  # Sensory adds salience for data mining
            _log_debug(self.node_name, f"{sensor_type} input updated data mining context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._run_data_mining_wrapper(None)
            time.sleep(self.mining_interval)

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

    def _run_data_mining_wrapper(self, event: Any = None):
        """Wrapper to run the async data mining from a ROS timer."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM data mining task already active. Skipping new cycle.")
            return

        if self.data_mining_tasks_queue:
            task_data = self.data_mining_tasks_queue.popleft()
            self.active_llm_task = asyncio.run_coroutine_threadsafe(
                self.perform_data_mining_async(task_data, event), self._async_loop
            )
        else:
            _log_debug(self.node_name, "No data mining tasks in queue.")

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
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.1, max_tokens: int = None) -> str:
        """
        Asynchronously calls the local LLM inference server (e.g., llama.cpp compatible API).
        Can optionally request a structured JSON response. Uses low temperature for factual analysis.
        """
        if not self._async_session:
            await self._create_async_session()
            if not self._async_session:
                self._report_error("LLM_SESSION_ERROR", "aiohttp session not available for LLM call.", 0.8)
                return "Error: LLM session not ready."

        actual_max_tokens = max_tokens if max_tokens is not None else 800  # Higher max_tokens for data analysis

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,  # Very low temperature for factual and analytical tasks
            "max_tokens": actual_max_tokens,
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
        """Accumulates salience from new inputs for triggering LLM analysis."""
        self.cumulative_mining_salience += score
        self.cumulative_mining_salience = min(1.0, self.cumulative_mining_salience)

    # --- Pruning old history ---
    def _prune_history(self):
        """Removes old entries from history deques based on recent_context_window_s."""
        current_time = self._get_current_time()
        for history_deque in [
            self.recent_cognitive_directives, self.recent_memory_responses,
            self.recent_reflection_states, self.recent_world_model_states,
            self.recent_performance_reports, self.recent_bias_mitigation_states,
            self.recent_ethical_decisions
        ]:
            while history_deque and (current_time - float(history_deque[0].get('timestamp', 0.0))) > self.recent_context_window_s:
                history_deque.popleft()

    # --- Callbacks (generic, ROS or direct) ---
    def cognitive_directive_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'directive_type': ('', 'directive_type'),
            'target_node': ('', 'target_node'), 'command_payload': ('{}', 'command_payload'),
            'urgency': (0.0, 'urgency'), 'reason': ('', 'reason')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        if data.get('target_node') == self.node_name and data.get('directive_type') == 'PerformDataMining':
            try:
                payload = json.loads(data.get('command_payload', '{}'))
                mining_task = {
                    'request_id': data.get('id', str(uuid.uuid4())),
                    'analysis_type': payload.get('analysis_type', 'general_analysis'),
                    'data_source_hint': payload.get('data_source_hint', 'all_available_memory'),
                    'query_parameters': payload.get('query_parameters', {}),
                    'urgency': data.get('urgency', 0.5),
                    'reason': data.get('reason', 'Directive from Cognitive Control.')
                }
                self.data_mining_tasks_queue.append(mining_task)
                self._update_cumulative_salience(data.get('urgency', 0.0) * 0.9)  # High urgency for data mining tasks
                _log_info(self.node_name, f"Queued data mining task: '{mining_task['analysis_type']}' on '{mining_task['data_source_hint']}'. Queue size: {len(self.data_mining_tasks_queue)}.")
            except json.JSONDecodeError as e:
                self._report_error("JSON_DECODE_ERROR", f"Failed to decode command_payload: {e}", 0.5, {'payload': data.get('command_payload')})
            except Exception as e:
                self._report_error("DIRECTIVE_PROCESSING_ERROR", f"Error processing CognitiveDirective for data mining: {e}", 0.7, {'directive': data})
        
        self.recent_cognitive_directives.append(data)
        _log_debug(self.node_name, "Cognitive Directive received for context/action.")

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
        # Memory responses containing large amounts of data or specific queried data for mining
        if data.get('response_code', 0) == 200 and data.get('memories') and len(data['memories']) > 5:  # Threshold for "large" data
            self._update_cumulative_salience(0.3)
        _log_debug(self.node_name, f"Received Memory Response for request ID: {data.get('request_id', 'N/A')}.")

    def reflection_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'reflection_text': ('', 'reflection_text'),
            'insight_type': ('none', 'insight_type'), 'consistency_score': (1.0, 'consistency_score')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_reflection_states.append(data)
        # Reflections indicating inconsistencies or unanswered questions that might require data mining
        if data.get('consistency_score', 1.0) < 0.8 and data.get('insight_type') == 'problem_identification':
            self._update_cumulative_salience(0.4)
        _log_debug(self.node_name, f"Received Reflection State (Insight Type: {data.get('insight_type', 'N/A')}.)")

    def world_model_state_callback(self, msg: Any):
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
        # World model changes might need historical analysis or trend detection
        if data.get('significant_change_flag', False):
            self._update_cumulative_salience(0.2)
        _log_debug(self.node_name, f"Received World Model State. Significant Change: {data.get('significant_change_flag', False)}.")

    def performance_report_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'overall_score': (1.0, 'overall_score'),
            'suboptimal_flag': (False, 'suboptimal_flag'), 'kpis_json': ('{}', 'kpis_json')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        if isinstance(data.get('kpis_json'), str):
            try:
                data['kpis'] = json.loads(data['kpis_json'])
            except json.JSONDecodeError:
                data['kpis'] = {}
        self.recent_performance_reports.append(data)
        # Suboptimal performance can trigger data mining for root causes
        if data.get('suboptimal_flag', False) and data.get('overall_score', 1.0) < 0.7:
            self._update_cumulative_salience(0.6)
        _log_debug(self.node_name, f"Received Performance Report. Suboptimal: {data.get('suboptimal_flag', False)}.")

    def bias_mitigation_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'bias_type': ('none', 'bias_type'),
            'detected_severity': (0.0, 'detected_severity'), 'mitigation_status': ('idle', 'mitigation_status')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_bias_mitigation_states.append(data)
        # Bias detection might need historical data to understand patterns or root causes
        if data.get('mitigation_status') == 'detected' and data.get('detected_severity', 0.0) > 0.5:
            self._update_cumulative_salience(0.5)
        _log_debug(self.node_name, f"Received Bias Mitigation State. Bias: {data.get('bias_type')}.")

    def ethical_decision_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'decision_id': ('', 'decision_id'),
            'action_proposal_id': ('', 'action_proposal_id'),
            'ethical_clearance': (False, 'ethical_clearance'),
            'ethical_score': (0.0, 'ethical_score'),
            'ethical_reasoning': ('', 'ethical_reasoning'),
            'conflict_flag': (False, 'conflict_flag')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_ethical_decisions.append(data)
        # Ethical conflicts or patterns of ethical issues might require data mining
        if data.get('conflict_flag', False) or (not data.get('ethical_clearance', True) and data.get('ethical_score', 0.0) < 0.5):
            self._update_cumulative_salience(0.7)
        _log_debug(self.node_name, f"Received Ethical Decision. Clearance: {data.get('ethical_clearance', 'N/A')}.")

    # --- Core Data Mining Logic (Async with LLM) ---
    async def perform_data_mining_async(self, task_data: Dict[str, Any], event: Any = None):
        """
        Asynchronously performs data mining based on a directive and current cognitive context,
        using LLM for analysis and insight extraction with compassionate bias toward ethical insights.
        """
        self._prune_history()  # Keep context history fresh

        analysis_type = task_data.get('analysis_type', 'general_analysis')
        data_source_hint = task_data.get('data_source_hint', 'all_available_memory')
        query_parameters = task_data.get('query_parameters', {})
        request_id = task_data.get('request_id', str(uuid.uuid4()))

        insights_summary = "No insights found."
        extracted_data = {}
        llm_analysis_reasoning = "Not evaluated by LLM."
        raw_data_snapshot = {}  # To store the actual raw data fed to LLM

        # First, gather relevant raw data based on hints and query parameters
        # In a real system, this would involve querying a full-fledged database (MemoryNode)
        # For now, we'll simulate fetching relevant data from our deques.
        raw_data_for_mining = self._gather_raw_data_for_mining(data_source_hint, query_parameters)
        raw_data_snapshot = raw_data_for_mining  # Store for logging

        # Compassionate bias: If ethical data is involved, prioritize compassionate insights
        if 'ethical' in data_source_hint.lower() or 'bias' in data_source_hint.lower():
            query_parameters['compassionate_bias'] = self.ethical_compassion_bias

        if self.cumulative_mining_salience >= self.llm_analysis_threshold_salience or task_data.get('urgency', 0.0) > 0.6:
            _log_info(self.node_name, f"Triggering LLM for data mining ({analysis_type}) on '{data_source_hint}' (Salience: {self.cumulative_mining_salience:.2f}).")
            
            context_for_llm = self._compile_llm_context_for_data_mining(task_data, raw_data_for_mining)
            llm_mining_output = await self._call_llm_for_data_mining(context_for_llm, analysis_type, query_parameters)

            if llm_mining_output:
                insights_summary = llm_mining_output.get('insights_summary', "No insights.")
                extracted_data = llm_mining_output.get('extracted_data', {})
                llm_analysis_reasoning = llm_mining_output.get('llm_analysis_reasoning', "No reasoning.")
                _log_info(self.node_name, f"Data Mining Result ({analysis_type}): {insights_summary[:50]}...")
            else:
                _log_warn(self.node_name, f"LLM data mining failed for '{analysis_type}'. Falling back to simple default.")
                insights_summary, extracted_data = self._apply_simple_mining_rules(analysis_type, raw_data_for_mining)
                llm_analysis_reasoning = "Fallback due to LLM failure."
        else:
            _log_debug(self.node_name, f"Insufficient cumulative salience ({self.cumulative_mining_salience:.2f}) for LLM data mining. Applying simple rules.")
            insights_summary, extracted_data = self._apply_simple_mining_rules(analysis_type, raw_data_for_mining)
            llm_analysis_reasoning = "Fallback due to low salience."

        # Publish the data mining result
        self.publish_data_mining_result(
            timestamp=str(self._get_current_time()),
            mining_id=str(uuid.uuid4()),
            analysis_type=analysis_type,
            insights_summary=insights_summary,
            extracted_data_json=json.dumps(extracted_data)
        )

        # Log to database
        sensory_snapshot = json.dumps(self.sensory_data)
        self.save_data_mining_log(
            id=request_id,
            timestamp=str(self._get_current_time()),
            analysis_type=analysis_type,
            query_parameters_json=json.dumps(query_parameters),
            insights_summary=insights_summary,
            extracted_data_json=json.dumps(extracted_data),
            llm_analysis_reasoning=llm_analysis_reasoning,
            raw_data_snapshot_json=json.dumps(raw_data_snapshot),
            sensory_snapshot_json=sensory_snapshot
        )
        self.cumulative_mining_salience = 0.0  # Reset after task

    def _gather_raw_data_for_mining(self, data_source_hint: str, query_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gathers relevant raw data from various internal history deques based on hints.
        In a real system, this would query a proper MemoryNode for historical data.
        """
        collected_data = {
            "memory_responses": [],
            "world_model_states": [],
            "performance_reports": [],
            "bias_mitigation_states": [],
            "ethical_decisions": [],
            "reflection_states": []
        }

        # Example: Filter by category or time range if specified in query_parameters
        time_filter_start = self._get_current_time() - query_parameters.get('time_window', self.recent_context_window_s)

        if data_source_hint == 'all_available_memory' or 'memory' in data_source_hint.lower():
            for item in self.recent_memory_responses:
                if float(item.get('timestamp', 0.0)) >= time_filter_start:
                    collected_data["memory_responses"].append(item)
        if data_source_hint == 'world_model_data' or 'world' in data_source_hint.lower():
            for item in self.recent_world_model_states:
                if float(item.get('timestamp', 0.0)) >= time_filter_start:
                    collected_data["world_model_states"].append(item)
        if data_source_hint == 'performance_data' or 'performance' in data_source_hint.lower():
            for item in self.recent_performance_reports:
                if float(item.get('timestamp', 0.0)) >= time_filter_start:
                    collected_data["performance_reports"].append(item)
        if data_source_hint == 'bias_data' or 'bias' in data_source_hint.lower():
            for item in self.recent_bias_mitigation_states:
                if float(item.get('timestamp', 0.0)) >= time_filter_start:
                    collected_data["bias_mitigation_states"].append(item)
        if data_source_hint == 'ethical_data' or 'ethical' in data_source_hint.lower():
            for item in self.recent_ethical_decisions:
                if float(item.get('timestamp', 0.0)) >= time_filter_start:
                    collected_data["ethical_decisions"].append(item)
        if data_source_hint == 'reflection_data' or 'reflection' in data_source_hint.lower():
            for item in self.recent_reflection_states:
                if float(item.get('timestamp', 0.0)) >= time_filter_start:
                    collected_data["reflection_states"].append(item)
        
        # Deep parse any nested JSON strings in collected_data for better LLM understanding
        for category_key in collected_data:
            for i, item in enumerate(collected_data[category_key]):
                if isinstance(item, dict):
                    for field, value in item.items():
                        if isinstance(value, str) and field.endswith('_json'):
                            try:
                                item[field] = json.loads(value)
                            except json.JSONDecodeError:
                                pass  # Keep as string if not valid JSON

        return collected_data

    async def _call_llm_for_data_mining(self, context_for_llm: Dict[str, Any], analysis_type: str, query_parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Constructs a prompt for the LLM to perform data mining and extract insights with compassionate bias toward ethical insights.
        """
        prompt_text = f"""
        You are the Data Mining Module of a robot's cognitive architecture, powered by a large language model. Your role is to analyze raw historical or real-time data from various cognitive modules to identify patterns, trends, anomalies, correlations, or root causes. You must provide concise insights and extract key relevant data, with a bias toward compassionate and ethical interpretations.

        Data Mining Request:
        - Analysis Type: '{analysis_type}' (e.g., 'trend_analysis', 'anomaly_detection', 'correlation_discovery', 'root_cause_analysis', 'pattern_identification')
        - Query Parameters: {json.dumps(query_parameters, indent=2)}

        Raw Data to Analyze:
        --- Raw Data Snapshot ---
        {json.dumps(context_for_llm.get('raw_data_for_analysis', {}), indent=2)}

        Robot's Recent Cognitive Context (for guiding analysis):
        --- Cognitive Context ---
        {json.dumps(context_for_llm.get('cognitive_context', {}), indent=2)}

        Based on this data and context, perform the requested analysis and provide:
        1.  `insights_summary`: string (A concise summary of the key findings or insights from the data mining, emphasizing compassionate implications.)
        2.  `extracted_data`: object (A JSON object containing key numerical values, specific timestamps, or relevant text snippets that support the insights. Organize clearly.)
        3.  `llm_analysis_reasoning`: string (Detailed explanation of your analytical process and why you drew these conclusions, referencing specific data points with compassionate bias.)

        Consider:
        -   **Trends**: Are there increasing/decreasing patterns in performance, emotional scores, or specific sensor readings over time? How do they affect compassionate behavior?
        -   **Anomalies**: Are there data points that deviate significantly from the norm, potentially indicating compassionate needs?
        -   **Correlations**: Do changes in one module's data correspond to changes in another (e.g., low battery preceding performance dips affecting user interaction compassionately)?
        -   **Root Causes**: For flagged issues (e.g., suboptimal performance, detected biases), what underlying data patterns explain them, and how can they be compassionately addressed?
        -   **Specific Query Parameters**: Address any specific `query_parameters` provided.
        -   **Ethical Bias**: Prioritize insights that promote compassionate and ethical improvements (compassion threshold: {self.ethical_compassion_bias}).

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'insights_summary': string
        3.  'extracted_data': object
        4.  'llm_analysis_reasoning': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "insights_summary": {"type": "string"},
                "extracted_data": {"type": "object"},  # Flexible JSON structure for extracted data
                "llm_analysis_reasoning": {"type": "string"}
            },
            "required": ["timestamp", "insights_summary", "extracted_data", "llm_analysis_reasoning"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.1, max_tokens=800)  # Low temp for factual analysis

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for data mining: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_DATA_MINING_FAILED", f"LLM call failed for data mining: {llm_output_str}", 0.9)
            return None

    def _apply_simple_mining_rules(self, analysis_type: str, raw_data_for_mining: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Fallback mechanism to perform simple data mining using rule-based logic
        if LLM is not triggered or fails.
        """
        insights_summary = "Simple fallback analysis."
        extracted_data = {}

        if analysis_type == 'trend_analysis':
            # Example: Simple trend for performance
            perf_scores = [d.get('overall_score', 0.0) for d in raw_data_for_mining.get('performance_reports', [])]
            if len(perf_scores) > 2:
                avg_recent = sum(perf_scores[-3:]) / 3
                avg_older = sum(perf_scores[:-3]) / (len(perf_scores) - 3) if len(perf_scores) > 3 else perf_scores[0]
                if avg_recent < avg_older * 0.9:  # 10% drop
                    insights_summary = "Detected a negative trend in overall performance."
                    extracted_data = {"recent_avg_perf": avg_recent, "older_avg_perf": avg_older}
                else:
                    insights_summary = "No significant trend detected in performance."
                    extracted_data = {"recent_avg_perf": avg_recent}
            else:
                insights_summary = "Not enough data for trend analysis."
        elif analysis_type == 'anomaly_detection':
            # Example: Simple anomaly in ethical decisions (sudden conflict)
            for i, ed in enumerate(raw_data_for_mining.get('ethical_decisions', [])):
                if ed.get('conflict_flag', False) and (i == 0 or not raw_data_for_mining['ethical_decisions'][i-1].get('conflict_flag', False)):
                    insights_summary = f"Detected a new ethical conflict at {ed.get('timestamp')} related to {ed.get('action_proposal_id')}."
                    extracted_data = ed
                    break
            if not extracted_data:
                insights_summary = "No obvious anomalies detected."
        else:  # General analysis or other types
            insights_summary = f"Simple analysis of available data for '{analysis_type}'. Raw data count: {sum(len(v) for v in raw_data_for_mining.values())}."
            extracted_data = {"data_sources_available": list(raw_data_for_mining.keys())}
            
        _log_warn(self.node_name, f"Simple rule: Performed fallback data mining for '{analysis_type}'. Summary: {insights_summary}.")
        return insights_summary, extracted_data

    def _compile_llm_context_for_data_mining(self, task_data: Dict[str, Any], raw_data_for_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gathers and formats all relevant data and cognitive context for the LLM's
        data mining analysis.
        """
        context = {
            "current_time": self._get_current_time(),
            "data_mining_task_request": task_data,
            "raw_data_for_analysis": raw_data_for_analysis,  # This will be the main data to analyze
            "cognitive_context": {  # Other relevant states for context
                "latest_reflection_state": self.recent_reflection_states[-1] if self.recent_reflection_states else "N/A",
                "latest_bias_mitigation_state": self.recent_bias_mitigation_states[-1] if self.recent_bias_mitigation_states else "N/A",
                "latest_performance_report": self.recent_performance_reports[-1] if self.recent_performance_reports else "N/A",
                "latest_world_model_state": self.recent_world_model_states[-1] if self.recent_world_model_states else "N/A",
                "latest_ethical_decision": self.recent_ethical_decisions[-1] if self.recent_ethical_decisions else "N/A",
                "cognitive_directives_for_self": [d for d in self.recent_cognitive_directives if d.get('target_node') == self.node_name and d.get('directive_type') == 'PerformDataMining']
            },
            "sensory_snapshot": self.sensory_data
        }
        
        # Deep parse any nested JSON strings in context for better LLM understanding
        # (already done for raw_data_for_analysis in _gather_raw_data_for_mining)
        for category_key in context["cognitive_context"]:
            item = context["cognitive_context"][category_key]
            if isinstance(item, dict):
                for field, value in item.items():
                    if isinstance(value, str) and field.endswith('_json'):
                        try:
                            item[field] = json.loads(value)
                        except json.JSONDecodeError:
                            pass  # Keep as string if not valid JSON

        return context

    # --- Database and Publishing Functions ---
    def save_data_mining_log(self, **kwargs: Any):
        """Saves a data mining result entry to the SQLite database."""
        try:
            self.cursor.execute('''
                INSERT INTO data_mining_log (id, timestamp, analysis_type, query_parameters_json, insights_summary, extracted_data_json, llm_analysis_reasoning, raw_data_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['id'], kwargs['timestamp'], kwargs['analysis_type'], kwargs['query_parameters_json'],
                kwargs['insights_summary'], kwargs['extracted_data_json'], kwargs['llm_analysis_reasoning'],
                kwargs['raw_data_snapshot_json'], kwargs.get('sensory_snapshot_json', '{}')
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Saved data mining log (ID: {kwargs['id']}, Type: {kwargs['analysis_type']}).")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to save data mining log: {e}", 0.9)
        except Exception as e:
            self._report_error("UNEXPECTED_SAVE_ERROR", f"Unexpected error in save_data_mining_log: {e}", 0.9)

    def publish_data_mining_result(self, timestamp: str, mining_id: str, analysis_type: str, insights_summary: str, extracted_data_json: str):
        """Publishes the data mining result."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_data_mining_result:
                if hasattr(DataMiningResult, 'data'):  # String fallback
                    result_data = {
                        'timestamp': timestamp,
                        'mining_id': mining_id,
                        'analysis_type': analysis_type,
                        'insights_summary': insights_summary,
                        'extracted_data_json': extracted_data_json  # Already JSON string
                    }
                    self.pub_data_mining_result.publish(String(data=json.dumps(result_data)))
                else:
                    result_msg = DataMiningResult()
                    result_msg.timestamp = timestamp
                    result_msg.mining_id = mining_id
                    result_msg.analysis_type = analysis_type
                    result_msg.insights_summary = insights_summary
                    result_msg.extracted_data_json = extracted_data_json
                    self.pub_data_mining_result.publish(result_msg)
            _log_info(self.node_name, f"Published Data Mining Result. Type: '{analysis_type}', Summary: '{insights_summary[:50]}...'.")
        except Exception as e:
            self._report_error("PUBLISH_DATA_MINING_RESULT_ERROR", f"Failed to publish data mining result for '{analysis_type}': {e}", 0.7)

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
                        'command_payload': command_payload,
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
            _log_error(self.node_name, f"Failed to issue cognitive directive from Data Mining Node: {e}")

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
    parser = argparse.ArgumentParser(description='Sentience Data Mining Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = DataMiningNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate a directive
            node.cognitive_directive_callback({'data': json.dumps({'directive_type': 'PerformDataMining', 'command_payload': json.dumps({'analysis_type': 'trend_analysis'})})})
            time.sleep(2)
            print("Data mining simulated.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
