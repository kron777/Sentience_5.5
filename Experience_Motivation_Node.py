```python:disable-run
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique goal IDs
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
    MotivationState = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    EmotionState = ROSMsgFallback
    PerformanceReport = ROSMsgFallback
    WorldModelState = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    SocialCognitionState = ROSMsgFallback
    PredictionState = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    MotivationState = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    EmotionState = ROSMsgFallback
    PerformanceReport = ROSMsgFallback
    WorldModelState = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    SocialCognitionState = ROSMsgFallback
    PredictionState = ROSMsgFallback


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
            'experience_motivation_node': {
                'motivation_update_interval': 1.0,
                'llm_motivation_threshold_salience': 0.6,
                'recent_context_window_s': 15.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate goal inference
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


class ExperienceMotivationNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'experience_motivation_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized configuration ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "motivation_log.db")
        self.motivation_update_interval = self.params.get('motivation_update_interval', 1.0)
        self.llm_motivation_threshold_salience = self.params.get('llm_motivation_threshold_salience', 0.6)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 15.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing motivation compassionately)
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

        _log_info(self.node_name, "Robot's experience motivation system online, inspiring compassionate and mindful goal-driven pursuit.")

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
            CREATE TABLE IF NOT EXISTS motivation_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                dominant_goal_id TEXT,
                overall_drive_level REAL,
                active_goals_json TEXT,
                llm_reasoning TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_motivation_timestamp ON motivation_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.current_motivation_state = {
            'timestamp': str(time.time()),
            'dominant_goal_id': 'explore_environment',  # Default passive goal
            'overall_drive_level': 0.1,
            'active_goals': [{'goal_id': 'explore_environment', 'priority': 0.1, 'status': 'active'}]
        }

        # History deques
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_emotion_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_performance_reports: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_world_model_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_memory_responses: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_internal_narratives: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_social_cognition_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_prediction_states: Deque[Dict[str, Any]] = deque(maxlen=5)

        self.cumulative_motivation_salience = 0.0

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_motivation_state = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_motivation_state = rospy.Publisher('/motivation_state', MotivationState, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            rospy.Subscriber('/emotion_state', EmotionState, self.emotion_state_callback)
            rospy.Subscriber('/performance_report', PerformanceReport, self.performance_report_callback)
            rospy.Subscriber('/world_model_state', WorldModelState, self.world_model_state_callback)
            rospy.Subscriber('/memory_response', MemoryResponse, self.memory_response_callback)
            rospy.Subscriber('/internal_narrative', InternalNarrative, self.internal_narrative_callback)
            rospy.Subscriber('/social_cognition_state', SocialCognitionState, self.social_cognition_state_callback)
            rospy.Subscriber('/prediction_state', PredictionState, self.prediction_state_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.motivation_update_interval), self._run_motivation_analysis_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        # Initial publish
        self.publish_motivation_state(None)

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing motivation compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on motivation
            if sensor_type == 'vision':
                self.recent_world_model_states.append({'timestamp': timestamp, 'significant_change_flag': random.random() < 0.3, 'num_entities': random.randint(1, 10)})
            elif sensor_type == 'sound':
                self.recent_emotion_states.append({'timestamp': timestamp, 'mood': random.choice(['curious', 'anxious', 'neutral']), 'mood_intensity': random.uniform(0.2, 0.8)})
            elif sensor_type == 'instructions':
                self.recent_cognitive_directives.append({'timestamp': timestamp, 'directive_type': 'set_goal', 'command_payload': json.dumps({'goal_id': 'user_request', 'priority': random.uniform(0.4, 0.9)})})
            self._update_cumulative_salience(0.2)  # Sensory adds salience for motivation
            _log_debug(self.node_name, f"{sensor_type} input updated motivation context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._run_motivation_analysis_wrapper(None)
            time.sleep(self.motivation_update_interval)

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

    def _run_motivation_analysis_wrapper(self, event: Any = None):
        """Wrapper to run the async motivation analysis from a ROS timer."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM motivation analysis task already active. Skipping new cycle.")
            return
        
        # Schedule the async task
        self.active_llm_task = asyncio.run_coroutine_threadsafe(
            self.analyze_motivation_async(event), self._async_loop
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
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.5, max_tokens: int = 300) -> str:
        """
        Asynchronously calls the local LLM inference server (e.g., llama.cpp compatible API).
        Can optionally request a structured JSON response. Moderate temperature for goal-setting nuance.
        """
        if not self._async_session:
            await self._create_async_session()
            if not self._async_session:
                self._report_error("LLM_SESSION_ERROR", "aiohttp session not available for LLM call.", 0.8)
                return "Error: LLM session not ready."

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,  # Moderate temperature for goal-setting nuance
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
        self.cumulative_motivation_salience += score
        self.cumulative_motivation_salience = min(1.0, self.cumulative_motivation_salience)

    # --- Pruning old history ---
    def _prune_history(self):
        """Removes old entries from history deques based on recent_context_window_s."""
        current_time = self._get_current_time()
        for history_deque in [
            self.recent_cognitive_directives, self.recent_emotion_states,
            self.recent_performance_reports, self.recent_world_model_states,
            self.recent_memory_responses, self.recent_internal_narratives,
            self.recent_social_cognition_states, self.recent_prediction_states
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
        
        if data.get('target_node') == self.node_name:
            self.recent_cognitive_directives.append(data)  # Add directives for self to context
            # Directives to set new goals or adjust motivation are highly salient
            if data.get('directive_type') in ['SetGoal', 'AdjustMotivation']:
                self._update_cumulative_salience(data.get('urgency', 0.0) * 0.9)
            _log_info(self.node_name, f"Received directive for self: '{data.get('directive_type', 'N/A')}' (Payload: {data.get('command_payload', 'N/A')}.)")
        else:
            self.recent_cognitive_directives.append(data)  # Add all directives for general context
        _log_debug(self.node_name, "Cognitive Directive received for context/action.")

    def emotion_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'mood': ('neutral', 'mood'),
            'sentiment_score': (0.0, 'sentiment_score'), 'mood_intensity': (0.0, 'mood_intensity')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_emotion_states.append(data)
        # Strong emotions can influence motivation (e.g., frustration leading to change goal, happiness reinforcing current goal)
        if data.get('mood_intensity', 0.0) > 0.5:
            self._update_cumulative_salience(data.get('mood_intensity', 0.0) * 0.3)
        _log_debug(self.node_name, f"Received Emotion State. Mood: {data.get('mood', 'N/A')}.")

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
        # Suboptimal performance can decrease motivation for current goal, high performance can increase
        if data.get('suboptimal_flag', False) or data.get('overall_score', 1.0) < 0.7:
            self._update_cumulative_salience(0.5)  # Need to re-evaluate if not doing well
        elif data.get('overall_score', 0.0) > 0.9:
            self._update_cumulative_salience(0.2)  # Reinforce success
        _log_debug(self.node_name, f"Received Performance Report. Suboptimal: {data.get('suboptimal_flag', False)}.")

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
        # Changes in world model can present new opportunities or obstacles to current goals
        if data.get('significant_change_flag', False):
            self._update_cumulative_salience(0.4)
        _log_debug(self.node_name, f"Received World Model State. Significant Change: {data.get('significant_change_flag', False)}.")

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
                data['memories']
```
