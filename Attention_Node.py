```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique attention log IDs
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
    AttentionState = ROSMsgFallback
    SensoryQualia = ROSMsgFallback
    SocialCognitionState = ROSMsgFallback
    EmotionState = ROSMsgFallback
    MotivationState = ROSMsgFallback
    PerformanceReport = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    AttentionState = ROSMsgFallback
    SensoryQualia = ROSMsgFallback
    SocialCognitionState = ROSMsgFallback
    EmotionState = ROSMsgFallback
    MotivationState = ROSMsgFallback
    PerformanceReport = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback


# --- Import shared utility functions (renamed for generality) ---
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
        default_config = {
            'db_root_path': '/tmp/sentience_db',
            'default_log_level': 'INFO',
            'ros_enabled': False,  # Default to non-ROS for dynamic mode
            'attention_node': {
                'attention_update_interval': 0.1,
                'llm_attention_threshold_salience': 0.5,
                'recent_context_window_s': 5.0,
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
        }
        if node_name == "global":
            return default_config
        return default_config.get(node_name, {})


def _log_info(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [INFO] {msg}", file=sys.stdout)

def _log_warn(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [WARN] {msg}", file=sys.stderr)

def _log_error(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [ERROR] {msg}", file=sys.stderr)

def _log_debug(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [DEBUG] {msg}", file=sys.stdout)


class AttentionNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'attention_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "attention_log.db")
        self.attention_update_interval = self.params.get('attention_update_interval', 0.1)
        self.llm_attention_threshold_salience = self.params.get('llm_attention_threshold_salience', 0.5)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 5.0)
        self.sensory_sources = self.params.get('sensory_inputs', {})

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 20.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Robot's attention system online, ready to focus with mindful awareness.")

        # --- Dynamic Sensory Placeholders ---
        self.sensory_data: Dict[str, Any] = {
            'vision': {'data': None, 'timestamp': 0.0},
            'sound': {'data': None, 'timestamp': 0.0},
            'instructions': {'data': None, 'timestamp': 0.0}
        }
        self.vision_callback = self._create_sensory_callback('vision')
        self.sound_callback = self._create_sensory_callback('sound')
        self.instructions_callback = self._create_sensory_callback('instructions')

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
            CREATE TABLE IF NOT EXISTS attention_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                focus_type TEXT,
                focus_target TEXT,
                priority_score REAL,
                llm_reasoning TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_attention_timestamp ON attention_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.current_attention_state = {
            'timestamp': str(time.time()),
            'focus_type': 'idle',
            'focus_target': 'environment',
            'priority_score': 0.1
        }

        # History deques
        self.recent_sensory_qualia: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_social_cognition_states: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_emotion_states: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_motivation_states: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_performance_reports: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=3)

        self.cumulative_attention_salience = 0.0

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_attention_state = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_attention_state = rospy.Publisher('/attention_state', AttentionState, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/sensory_qualia', SensoryQualia, self.sensory_qualia_callback)
            rospy.Subscriber('/social_cognition_state', String, self.social_cognition_state_callback)
            rospy.Subscriber('/emotion_state', EmotionState, self.emotion_state_callback)
            rospy.Subscriber('/motivation_state', String, self.motivation_state_callback)
            rospy.Subscriber('/performance_report', PerformanceReport, self.performance_report_callback)
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.attention_update_interval), self._run_attention_analysis_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        # Initial publish
        self.publish_attention_state(None)

    def _create_sensory_callback(self, sensor_type: str):
        def callback(data: Any):
            timestamp = time.time()
            processed_data = data if isinstance(data, dict) else {'raw': data}
            self.sensory_data[sensor_type] = {'data': processed_data, 'timestamp': timestamp}
            self._update_cumulative_salience(0.1)  # Sensory input adds slight salience
            _log_debug(self.node_name, f"{sensor_type} input updated at {timestamp}")
        return callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._run_attention_analysis_wrapper(None)
            time.sleep(self.attention_update_interval)

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

    def _run_attention_analysis_wrapper(self, event):
        """Wrapper to run the async attention analysis from a ROS timer."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM attention analysis task already active. Skipping new cycle.")
            return
        
        # Schedule the async task
        self.active_llm_task = asyncio.run_coroutine_threadsafe(
            self.analyze_attention_async(event), self._async_loop
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
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.3, max_tokens: int = 200) -> str:
        """
        Asynchronously calls the local LLM inference server (e.g., llama.cpp compatible API).
        Can optionally request a structured JSON response. Moderate temperature for attention focus.
        """
        if not self._async_session:
            await self._create_async_session()
            if not self._async_session:
                self._report_error("LLM_SESSION_ERROR", "aiohttp session not available for LLM call.", 0.8)
                return "Error: LLM session not ready."

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,
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
            self._report_error("LLM_API_ERROR", f"LLM API request failed: {e}", 0.9, {'url': api_url})
            return f"Error: LLM API request failed: {e}"
        except asyncio.TimeoutError:
            self._report_error("LLM_TIMEOUT", f"LLM API request timed out after {self.llm_timeout} seconds.", 0.8, {'prompt_snippet': prompt_text[:100]})
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
        self.cumulative_attention_salience += score
        self.cumulative_attention_salience = min(1.0, self.cumulative_attention_salience)

    # --- Pruning old history ---
    def _prune_history(self):
        """Removes old entries from history deques based on recent_context_window_s."""
        current_time = self._get_current_time()
        for history_deque in [
            self.recent_sensory_qualia, self.recent_social_cognition_states,
            self.recent_emotion_states, self.recent_motivation_states,
            self.recent_performance_reports, self.recent_cognitive_directives
        ]:
            while history_deque and (current_time - float(history_deque[0].get('timestamp', 0.0))) > self.recent_context_window_s:
                history_deque.popleft()

    # --- Callbacks (generic, ROS or direct calls) ---
    def sensory_qualia_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'qualia_id': ('', 'qualia_id'),
            'qualia_type': ('none', 'qualia_type'), 'modality': ('none', 'modality'),
            'description_summary': ('', 'description_summary'), 'salience_score': (0.0, 'salience_score'),
            'raw_data_hash': ('', 'raw_data_hash')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_sensory_qualia.append(data)
        # High salience sensory events demand attention
        self._update_cumulative_salience(data.get('salience_score', 0.0) * 0.5)
        _log_debug(self.node_name, f"Received Sensory Qualia. Description: {data.get('description_summary', 'N/A')}.")

    def social_cognition_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'inferred_mood': ('neutral', 'inferred_mood'),
            'mood_confidence': (0.0, 'mood_confidence'), 'inferred_intent': ('none', 'inferred_intent'),
            'intent_confidence': (0.0, 'intent_confidence'), 'user_id': ('unknown', 'user_id')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_social_cognition_states.append(data)
        # Direct user commands or high-confidence user intent/distress override other attention priorities
        if data.get('inferred_intent') in ['command', 'request_help'] and data.get('intent_confidence', 0.0) > 0.7:
            self._update_cumulative_salience(data.get('intent_confidence', 0.0) * 0.9)
        elif data.get('inferred_mood') in ['distressed', 'anxious'] and data.get('mood_confidence', 0.0) > 0.7:
            self._update_cumulative_salience(data.get('mood_confidence', 0.0) * 0.8)
        _log_debug(self.node_name, f"Received Social Cognition State. Intent: {data.get('inferred_intent', 'N/A')}.")

    def emotion_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'mood': ('neutral', 'mood'),
            'sentiment_score': (0.0, 'sentiment_score'), 'mood_intensity': (0.0, 'mood_intensity')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_emotion_states.append(data)
        # Strong robot emotions can shift internal attention (e.g., curiosity to exploration, anxiety to self-preservation)
        if data.get('mood_intensity', 0.0) > 0.6:
            self._update_cumulative_salience(data.get('mood_intensity', 0.0) * 0.4)
        _log_debug(self.node_name, f"Received Emotion State. Mood: {data.get('mood', 'N/A')}.")

    def motivation_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'dominant_goal_id': ('none', 'dominant_goal_id'),
            'overall_drive_level': (0.0, 'overall_drive_level'), 'active_goals_json': ('{}', 'active_goals_json')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        if isinstance(data.get('active_goals_json'), str):
            try:
                data['active_goals'] = json.loads(data['active_goals_json'])
            except json.JSONDecodeError:
                data['active_goals'] = {}
        self.recent_motivation_states.append(data)
        # Current goals strongly direct attention (e.g., if goal is 'navigate_to_X', attend to path planning)
        if data.get('overall_drive_level', 0.0) > 0.6 and data.get('dominant_goal_id') != 'none':
            self._update_cumulative_salience(data.get('overall_drive_level', 0.0) * 0.6)
        _log_debug(self.node_name, f"Received Motivation State. Goal: {data.get('dominant_goal_id', 'N/A')}.")

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
        # Suboptimal performance can trigger attention to self-audit or problem-solving
        if data.get('suboptimal_flag', False) and data.get('overall_score', 1.0) < 0.7:
            self._update_cumulative_salience(0.7)
        _log_debug(self.node_name, f"Received Performance Report. Suboptimal: {data.get('suboptimal_flag', False)}.")

    def cognitive_directive_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'directive_type': ('', 'directive_type'),
            'target_node': ('', 'target_node'), 'command_payload': ('{}', 'command_payload'),
            'urgency': (0.0, 'urgency'), 'reason': ('', 'reason'), 'id': ('', 'id')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        if data.get('target_node') == self.node_name and data.get('directive_type') == 'ShiftAttention':
            try:
                payload = json.loads(data.get('command_payload', '{}'))
                self._update_cumulative_salience(data.get('urgency', 0.0) * 1.0)  # Explicit attention directives are highly salient
                _log_info(self.node_name, f"Received directive to shift attention to reason: '{data.get('reason', 'N/A')}'.")
            except json.JSONDecodeError as e:
                self._report_error("JSON_DECODE_ERROR", f"Failed to decode command_payload: {e}", 0.5, {'payload': data.get('command_payload')})
            except Exception as e:
                self._report_error("DIRECTIVE_PROCESSING_ERROR", f"Error processing CognitiveDirective for attention: {e}", 0.7, {'directive': data})
        
        self.recent_cognitive_directives.append(data)  # Store all directives for context
        _log_debug(self.node_name, "Cognitive Directive received for context/action.")

    # --- Core Attention Analysis Logic (Async with LLM) ---
    async def analyze_attention_async(self, event: Any = None):
        """
        Asynchronously analyzes recent cognitive states to determine and update the robot's
        current attention focus, using LLM for nuanced prioritization.
        """
        self._prune_history()  # Keep context history fresh

        focus_type = self.current_attention_state.get('focus_type', 'idle')
        focus_target = self.current_attention_state.get('focus_target', 'environment')
        priority_score = self.current_attention_state.get('priority_score', 0.1)
        llm_reasoning = "Not evaluated by LLM."
        
        if self.cumulative_attention_salience >= self.llm_attention_threshold_salience:
            _log_info(self.node_name, f"Triggering LLM for attention analysis (Salience: {self.cumulative_attention_salience:.2f}).")
            
            context_for_llm = self._compile_llm_context_for_attention()
            llm_attention_output = await self._infer_attention_state_llm(context_for_llm)

            if llm_attention_output:
                focus_type = llm_attention_output.get('focus_type', focus_type)
                focus_target = llm_attention_output.get('focus_target', focus_target)
                priority_score = max(0.0, min(1.0, llm_attention_output.get('priority_score', priority_score)))
                llm_reasoning = llm_attention_output.get('llm_reasoning', 'LLM provided no specific reasoning.')
                _log_info(self.node_name, f"LLM Inferred Attention. Type: '{focus_type}', Target: '{focus_target}' (Priority: {priority_score:.2f}).")
            else:
                _log_warn(self.node_name, "LLM attention analysis failed. Applying simple fallback.")
                focus_type, focus_target, priority_score = self._apply_simple_attention_rules()
                llm_reasoning = "Fallback to simple rules due to LLM failure."
        else:
            _log_debug(self.node_name, f"Insufficient cumulative salience ({self.cumulative_attention_salience:.2f}) for LLM attention analysis. Applying simple rules.")
            focus_type, focus_target, priority_score = self._apply_simple_attention_rules()
            llm_reasoning = "Fallback to simple rules due to low salience."

        self.current_attention_state = {
            'timestamp': str(self._get_current_time()),
            'focus_type': focus_type,
            'focus_target': focus_target,
            'priority_score': priority_score
        }

        sensory_snapshot = json.dumps(self.sensory_data)
        self.save_attention_log(
            id=str(uuid.uuid4()),
            timestamp=self.current_attention_state['timestamp'],
            focus_type=self.current_attention_state['focus_type'],
            focus_target=self.current_attention_state['focus_target'],
            priority_score=self.current_attention_state['priority_score'],
            llm_reasoning=llm_reasoning,
            context_snapshot_json=json.dumps(self._compile_llm_context_for_attention()),
            sensory_snapshot_json=sensory_snapshot
        )
        self.publish_attention_state(None)  # Publish updated state
        self.cumulative_attention_salience = 0.0  # Reset after analysis

    async def _infer_attention_state_llm(self, context_for_llm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses the LLM to infer the robot's current attention state, including
        focus type, focus target, and priority score.
        """
        prompt_text = f"""
        You are the Attention Module of a robot's cognitive architecture, powered by a large language model. Your role is to determine the robot's current `focus_type`, `focus_target`, and `priority_score` by synthesizing inputs from various cognitive modules. This module governs what information the robot prioritizes for processing and action.

        Robot's Recent Cognitive Context (for Attention Inference):
        --- Cognitive Context ---
        {json.dumps(context_for_llm, indent=2)}

        Sensory Snapshot:
        --- Sensory Data ---
        {json.dumps(context_for_llm.get('sensory_snapshot', {}), indent=2)}

        Based on this context, provide:
        1.  `focus_type`: string (The category of what the robot is attending to, e.g., 'sensory_event', 'user_interaction', 'internal_reflection', 'goal_driven', 'problem_solving', 'self_audit', 'idle').
        2.  `focus_target`: string (The specific entity, concept, or area the robot's attention is directed at, e.g., 'human_user', 'obstacle_X', 'battery_level', 'current_task_progress', 'memory_retrieval', 'environment').
        3.  `priority_score`: number (0.0 to 1.0, indicating the urgency or importance of this attention focus. 1.0 is highest priority.)
        4.  `llm_reasoning`: string (Detailed explanation for your attention shift decision, referencing specific contextual inputs that influenced the focus and its priority.)

        Consider:
        -   **Cognitive Directives**: Are there explicit directives like 'ShiftAttention' to a specific target or type? These are paramount.
        -   **Sensory Qualia**: Are there highly `salience_score` events with urgent `description_summary`? (e.g., "loud bang", "obstacle detected").
        -   **Social Cognition States**: Is the user showing `distressed` `inferred_mood` or giving a direct `command` via `inferred_intent`?
        -   **Emotion States**: Is the robot feeling intense `curiosity` (focus on exploration), `anxiety` (focus on safety/threats), or `frustration` (focus on problem)?
        -   **Motivation States**: What is the `dominant_goal_id` and `overall_drive_level`? Attention should align with achieving this goal.
        -   **Performance Reports**: Is `suboptimal_flag` true? Attention might shift to self-audit or problem-solving.
        -   **Sensory Inputs**: Vision/sound/instructions indicating immediate risks or cues?

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'focus_type': string
        3.  'focus_target': string
        4.  'priority_score': number
        5.  'llm_reasoning': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "focus_type": {"type": "string"},
                "focus_target": {"type": "string"},
                "priority_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "llm_reasoning": {"type": "string"}
            },
            "required": ["timestamp", "focus_type", "focus_target", "priority_score", "llm_reasoning"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.3, max_tokens=250)

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                # Ensure numerical fields are floats
                if 'priority_score' in llm_data:
                    llm_data['priority_score'] = float(llm_data['priority_score'])
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for attention: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_ATTENTION_ANALYSIS_FAILED", f"LLM call failed for attention analysis: {llm_output_str}", 0.9)
            return None

    def _apply_simple_attention_rules(self) -> tuple[str, str, float]:
        """
        Fallback mechanism to infer attention state using simple rule-based logic
        if LLM is not triggered or fails.
        """
        current_time = self._get_current_time()
        
        focus_type = "idle"
        focus_target = "environment"
        priority_score = 0.1

        # Rule 1: Prioritize explicit directives to shift attention
        for directive in reversed(self.recent_cognitive_directives):
            time_since_directive = current_time - float(directive.get('timestamp', 0.0))
            if time_since_directive < 1.0 and directive.get('target_node') == self.node_name and \
               directive.get('directive_type') == 'ShiftAttention':
                payload_str = directive.get('command_payload', '{}')
                try:
                    payload = json.loads(payload_str)
                    focus_type = payload.get('focus_type', 'directive_driven')
                    focus_target = payload.get('focus_target', 'unspecified_directive_target')
                    priority_score = max(0.8, directive.get('urgency', 0.0))  # High priority for direct commands
                    _log_debug(self.node_name, f"Simple rule: Direct attention shift from directive: {focus_target}.")
                    return focus_type, focus_target, priority_score
                except json.JSONDecodeError:
                    pass  # Skip invalid payload

        # Rule 2: Prioritize urgent sensory events
        if self.recent_sensory_qualia:
            latest_qualia = self.recent_sensory_qualia[-1]
            time_since_qualia = current_time - float(latest_qualia.get('timestamp', 0.0))
            if time_since_qualia < 0.2 and latest_qualia.get('salience_score', 0.0) > 0.8:
                focus_type = "sensory_event"
                focus_target = f"{latest_qualia.get('modality', 'unknown')}_salient_event"
                priority_score = latest_qualia.get('salience_score', 0.0)
                _log_debug(self.node_name, "Simple rule: Attention to salient sensory event.")
                return focus_type, focus_target, priority_score

        # Rule 3: Prioritize user interaction if a clear intent is detected
        if self.recent_social_cognition_states:
            latest_social = self.recent_social_cognition_states[-1]
            time_since_social = current_time - float(latest_social.get('timestamp', 0.0))
            if time_since_social < 0.5 and latest_social.get('inferred_intent') != 'none' and latest_social.get('intent_confidence', 0.0) > 0.6:
                focus_type = "user_interaction"
                focus_target = f"user_{latest_social.get('user_id', 'unknown')}_{latest_social.get('inferred_intent', 'N/A')}"
                priority_score = latest_social.get('intent_confidence', 0.0) * 0.8
                _log_debug(self.node_name, "Simple rule: Attention to user intent.")
                return focus_type, focus_target, priority_score
        
        # Rule 4: Prioritize current dominant goal if active
        if self.recent_motivation_states:
            latest_motivation = self.recent_motivation_states[-1]
            time_since_motivation = current_time - float(latest_motivation.get('timestamp', 0.0))
            if time_since_motivation < 1.0 and latest_motivation.get('dominant_goal_id') != 'none' and latest_motivation.get('overall_drive_level', 0.0) > 0.5:
                focus_type = "goal_driven"
                focus_target = latest_motivation.get('dominant_goal_id')
                priority_score = latest_motivation.get('overall_drive_level', 0.0) * 0.7
                _log_debug(self.node_name, "Simple rule: Attention to dominant goal.")
                return focus_type, focus_target, priority_score

        _log_debug(self.node_name, "Simple rule: Defaulting to idle environmental attention.")
        return focus_type, focus_target, priority_score

    def _compile_llm_context_for_attention(self) -> Dict[str, Any]:
        """
        Gathers and formats all relevant cognitive state data for the LLM's
        attention inference.
        """
        context = {
            "current_time": self._get_current_time(),
            "current_attention_state": self.current_attention_state,
            "recent_cognitive_inputs": {
                "sensory_qualia": list(self.recent_sensory_qualia),
                "social_cognition_states": list(self.recent_social_cognition_states),
                "emotion_states": list(self.recent_emotion_states),
                "motivation_states": list(self.recent_motivation_states),
                "performance_reports": list(self.recent_performance_reports),
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
    def save_attention_log(self, **kwargs: Any):
        """Saves an attention state entry to the SQLite database."""
        try:
            self.cursor.execute('''
                INSERT INTO attention_log (id, timestamp, focus_type, focus_target, priority_score, llm_reasoning, context_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['id'], kwargs['timestamp'], kwargs['focus_type'], kwargs['focus_target'],
                kwargs['priority_score'], kwargs['llm_reasoning'], kwargs['context_snapshot_json'],
                kwargs.get('sensory_snapshot_json', '{}')
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Saved attention log (ID: {kwargs['id']}, Target: {kwargs['focus_target']}).")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to save attention log: {e}", 0.9)
        except Exception as e:
            self._report_error("UNEXPECTED_SAVE_ERROR", f"Unexpected error in save_attention_log: {e}", 0.9)

    def publish_attention_state(self, event: Any = None):
        """Publishes the robot's current attention state."""
        timestamp = str(self._get_current_time())
        # Update timestamp before publishing
        self.current_attention_state['timestamp'] = timestamp
        
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_attention_state:
                if hasattr(AttentionState, 'data'):  # String fallback
                    self.pub_attention_state.publish(String(data=json.dumps(self.current_attention_state)))
                else:
                    attention_msg = AttentionState(**self.current_attention_state)
                    self.pub_attention_state.publish(attention_msg)
            else:
                # Dynamic: Log or queue
                _log_debug(self.node_name, f"Dynamic Attention State: Target: '{self.current_attention_state['focus_target']}', Score: {self.current_attention_state['priority_score']}.")
            _log_debug(self.node_name, f"Published Attention State. Target: '{self.current_attention_state['focus_target']}', Score: {self.current_attention_state['priority_score']}.")
        except Exception as e:
            self._report_error("PUBLISH_ATTENTION_STATE_ERROR", f"Failed to publish attention state: {e}", 0.7)

    def publish_cognitive_directive(self, directive_type: str, target_node: str, command_payload: str, urgency: float, reason: str = ""):
        """Helper to publish a CognitiveDirective message."""
        timestamp = str(self._get_current_time())
        directive_data = {
            'timestamp': timestamp,
            'directive_type': directive_type,
            'target_node': target_node,
            'command_payload': command_payload,
            'urgency': urgency,
            'reason': reason
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
            _log_error(self.node_name, f"Failed to issue cognitive directive from Attention Node: {e}")

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
    parser = argparse.ArgumentParser(description='Sentience Attention Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = AttentionNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
