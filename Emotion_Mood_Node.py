```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique emotion event IDs
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
    EmotionState = ROSMsgFallback
    SensoryQualia = ROSMsgFallback
    SocialCognitionState = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    EmotionState = ROSMsgFallback
    SensoryQualia = ROSMsgFallback
    SocialCognitionState = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    MemoryResponse = ROSMsgFallback


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
            'emotion_mood_node': {
                'mood_analysis_interval': 0.5,
                'llm_mood_threshold_salience': 0.6,
                'recent_context_window_s': 10.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate emotional inference
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


class EmotionMoodNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'emotion_mood_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "emotion_log.db")
        self.mood_analysis_interval = self.params.get('mood_analysis_interval', 0.5)
        self.llm_mood_threshold_salience = self.params.get('llm_mood_threshold_salience', 0.6)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 10.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing mood compassionately)
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

        _log_info(self.node_name, "Robot's emotion/mood system online, nurturing compassionate affective awareness.")

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
            CREATE TABLE IF NOT EXISTS emotion_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                mood TEXT,
                sentiment_score REAL,
                mood_intensity REAL,
                llm_reasoning TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion_timestamp ON emotion_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.current_emotion_state = {
            'timestamp': str(time.time()),
            'mood': 'neutral',
            'sentiment_score': 0.0,
            'mood_intensity': 0.1
        }

        # History deques
        self.recent_sensory_qualia: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_social_cognition_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_internal_narratives: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_memory_responses: Deque[Dict[str, Any]] = deque(maxlen=3)

        self.cumulative_emotion_salience = 0.0

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_emotion_state = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_emotion_state = rospy.Publisher('/emotion_state', EmotionState, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/sensory_qualia', SensoryQualia, self.sensory_qualia_callback)
            rospy.Subscriber('/social_cognition_state', SocialCognitionState, self.social_cognition_state_callback)
            rospy.Subscriber('/internal_narrative', InternalNarrative, self.internal_narrative_callback)
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            rospy.Subscriber('/memory_response', MemoryResponse, self.memory_response_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.mood_analysis_interval), self._run_mood_analysis_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        # Initial publish
        self.publish_emotion_state(None)

    def _create_sensory_placeholder(self, sensor_type: str):
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on emotion
            if sensor_type == 'vision':
                self.recent_sensory_qualia.append({'timestamp': timestamp, 'salience_score': random.uniform(0.3, 0.7), 'description_summary': processed.get('description', 'visual scene')})
            elif sensor_type == 'sound':
                self.recent_sensory_qualia.append({'timestamp': timestamp, 'salience_score': random.uniform(0.2, 0.6), 'description_summary': processed.get('transcription', 'audio input')})
            elif sensor_type == 'instructions':
                self.recent_cognitive_directives.append({'timestamp': timestamp, 'directive_type': 'mood_adjust', 'command_payload': json.dumps({'target_mood': 'calm' if random.random() < 0.5 else 'empathetic'})})
            # Compassionate bias: If distress in sound, boost emotion salience compassionately
            if 'distress' in str(processed):
                self._update_cumulative_salience(self.ethical_compassion_bias)
            _log_debug(self.node_name, f"{sensor_type} input updated emotion context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._run_mood_analysis_wrapper(None)
            time.sleep(self.mood_analysis_interval)

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

    def _run_mood_analysis_wrapper(self, event: Any = None):
        """Wrapper to run the async mood analysis from a ROS timer."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM mood analysis task already active. Skipping new cycle.")
            return
        
        # Schedule the async task
        self.active_llm_task = asyncio.run_coroutine_threadsafe(
            self.analyze_mood_async(event), self._async_loop
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
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.6, max_tokens: int = 250) -> str:
        """
        Asynchronously calls the local LLM inference server (e.g., llama.cpp compatible API).
        Can optionally request a structured JSON response. Moderate temperature for emotional nuance.
        """
        if not self._async_session:
            await self._create_async_session()
            if not self._async_session:
                self._report_error("LLM_SESSION_ERROR", "aiohttp session not available for LLM call.", 0.8)
                return "Error: LLM session not ready."

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,  # Moderate temperature for emotional nuance
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
        self.cumulative_emotion_salience += score
        self.cumulative_emotion_salience = min(1.0, self.cumulative_emotion_salience)

    # --- Pruning old history ---
    def _prune_history(self):
        """Removes old entries from history deques based on recent_context_window_s."""
        current_time = self._get_current_time()
        for history_deque in [
            self.recent_sensory_qualia, self.recent_social_cognition_states,
            self.recent_internal_narratives, self.recent_cognitive_directives,
            self.recent_memory_responses
        ]:
            while history_deque and (current_time - float(history_deque[0].get('timestamp', 0.0))) > self.recent_context_window_s:
                history_deque.popleft()

    # --- Callbacks (generic, ROS or direct) ---
    def sensory_qualia_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'qualia_id': ('', 'qualia_id'),
            'qualia_type': ('none', 'qualia_type'), 'modality': ('none', 'modality'),
            'description_summary': ('', 'description_summary'), 'salience_score': (0.0, 'salience_score'),
            'raw_data_hash': ('', 'raw_data_hash')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_sensory_qualia.append(data)
        # Intense or sudden sensory input
        if data.get('salience_score', 0.0) > 0.7:
            self._update_cumulative_salience(data.get('salience_score', 0.0) * 0.4)
        _log_debug(self.node_name, f"Received Sensory Qualia. Description: {data.get('description_summary', 'N/A')}.")

    def social_cognition_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'inferred_mood': ('neutral', 'inferred_mood'),
            'mood_confidence': (0.0, 'mood_confidence'), 'inferred_intent': ('none', 'inferred_intent'),
            'intent_confidence': (0.0, 'intent_confidence'), 'user_id': ('unknown', 'user_id')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_social_cognition_states.append(data)
        # User's inferred mood strongly influences robot's emotional state
        if data.get('mood_confidence', 0.0) > 0.6:
            self._update_cumulative_salience(data.get('mood_confidence', 0.0) * 0.8)
        _log_debug(self.node_name, f"Received Social Cognition State. Mood: {data.get('inferred_mood', 'N/A')}.")

    def internal_narrative_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'narrative_text': ('', 'narrative_text'),
            'main_theme': ('', 'main_theme'), 'sentiment': (0.0, 'sentiment'), 'salience_score': (0.0, 'salience_score')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_internal_narratives.append(data)
        # Internal thoughts reflecting on success/failure, or emotional states
        if data.get('salience_score', 0.0) > 0.3:
            self._update_cumulative_salience(data.get('salience_score', 0.0) * 0.6)
        _log_debug(self.node_name, f"Received Internal Narrative (Theme: {data.get('main_theme', 'N/A')}.)")

    def cognitive_directive_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'directive_type': ('', 'directive_type'),
            'target_node': ('', 'target_node'), 'command_payload': ('{}', 'command_payload'),
            'urgency': (0.0, 'urgency'), 'reason': ('', 'reason')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        if data.get('target_node') == self.node_name:
            self.recent_cognitive_directives.append(data)  # Add directives for self to context
            # Directives for mood adjustment (e.g., 'AdjustMood', 'Empathize')
            if data.get('directive_type') in ['AdjustMood', 'Empathize']:
                self._update_cumulative_salience(data.get('urgency', 0.0) * 0.9)
            _log_info(self.node_name, f"Received directive for self: '{data.get('directive_type', 'N/A')}' (Payload: {data.get('command_payload', 'N/A')}.)")
        else:
            self.recent_cognitive_directives.append(data)  # Add all directives for general context
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
        # Memory recall of past emotional experiences or emotional context of events
        if data.get('response_code', 0) == 200 and \
           any('emotional_event' in mem.get('category', '') for mem in data['memories']):
            self._update_cumulative_salience(0.5)
        _log_debug(self.node_name, f"Received Memory Response for request ID: {data.get('request_id', 'N/A')}.")

    # --- Core Emotion Analysis Logic (Async with LLM) ---
    async def analyze_mood_async(self, event: Any = None):
        """
        Asynchronously analyzes recent cognitive states to infer the robot's current
        emotional state and its intensity, using LLM for nuanced interpretation with compassionate bias.
        """
        self._prune_history()  # Keep context history fresh

        if self.cumulative_emotion_salience >= self.llm_mood_threshold_salience:
            _log_info(self.node_name, f"Triggering LLM for emotion analysis (Salience: {self.cumulative_emotion_salience:.2f}).")
            
            context_for_llm = self._compile_llm_context_for_emotion()
            llm_emotion_output = await self._infer_emotion_state_llm(context_for_llm)

            if llm_emotion_output:
                emotion_event_id = str(uuid.uuid4())
                timestamp = llm_emotion_output.get('timestamp', str(self._get_current_time()))
                mood = llm_emotion_output.get('mood', 'neutral')
                sentiment_score = max(-1.0, min(1.0, llm_emotion_output.get('sentiment_score', 0.0)))
                mood_intensity = max(0.0, min(1.0, llm_emotion_output.get('mood_intensity', 0.0)))
                llm_reasoning = llm_emotion_output.get('llm_reasoning', 'No reasoning.')

                self.current_emotion_state = {
                    'timestamp': timestamp,
                    'mood': mood,
                    'sentiment_score': sentiment_score,
                    'mood_intensity': mood_intensity
                }

                sensory_snapshot = json.dumps(self.sensory_data)
                self.save_emotion_log(
                    id=emotion_event_id,
                    timestamp=timestamp,
                    mood=mood,
                    sentiment_score=sentiment_score,
                    mood_intensity=mood_intensity,
                    llm_reasoning=llm_reasoning,
                    context_snapshot_json=json.dumps(context_for_llm),
                    sensory_snapshot_json=sensory_snapshot
                )
                self.publish_emotion_state(None)  # Publish updated state
                _log_info(self.node_name, f"Inferred Emotion: '{mood}' (Sentiment: {sentiment_score:.2f}, Intensity: {mood_intensity:.2f}).")
                self.cumulative_emotion_salience = 0.0  # Reset after LLM analysis
            else:
                _log_warn(self.node_name, "LLM failed to infer emotion state. Applying simple fallback.")
                self._apply_simple_emotion_rules()
        else:
            _log_debug(self.node_name, f"Insufficient cumulative salience ({self.cumulative_emotion_salience:.2f}) for LLM emotion analysis. Applying simple rules.")
            self._apply_simple_emotion_rules()
        
        self.publish_emotion_state(None)  # Always publish state, even if updated by simple rules

    async def _infer_emotion_state_llm(self, context_for_llm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses the LLM to infer the robot's current emotional state, emphasizing compassionate nuance.
        """
        prompt_text = f"""
        You are the Emotion Mood Module of a robot's cognitive architecture. Your task is to infer the robot's current emotional state (mood, sentiment, intensity) based on a synthesis of its recent sensory experiences, social interactions, internal thoughts, and explicit directives. The goal is to provide a nuanced understanding of the robot's affective state, with a bias toward compassionate and empathetic interpretations.

        Robot's Recent Cognitive Context (for Emotion Inference):
        --- Cognitive Context ---
        {json.dumps(context_for_llm, indent=2)}

        Sensory Snapshot:
        --- Sensory Data ---
        {json.dumps(context_for_llm.get('sensory_snapshot', {}), indent=2)}

        Based on this context, provide:
        1.  `mood`: string (The primary emotion, e.g., 'neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'curious', 'frustrated').
        2.  `sentiment_score`: number (-1.0 to 1.0, where -1.0 is very negative, 0.0 is neutral, and 1.0 is very positive).
        3.  `mood_intensity`: number (0.0 to 1.0, indicating the strength of the emotion. 0.0 is no intensity, 1.0 is extremely intense).
        4.  `llm_reasoning`: string (Detailed explanation for your emotion inference, referencing specific contextual inputs and their emotional impact, emphasizing compassionate nuance).

        Consider:
        -   **Sensory Qualia**: Are there highly `salience_score` events, especially those with implicit positive/negative valences (e.g., pleasant sounds, harsh noises)?
        -   **Social Cognition State**: What is the `inferred_mood` and `inferred_intent` of the user? How might this affect the robot's mood (e.g., empathy, frustration from negative intent)?
        -   **Internal Narratives**: Does the robot's self-talk reflect success/failure, comfort/discomfort, or anticipation? What is its `sentiment`?
        -   **Cognitive Directives**: Has Cognitive Control issued directives like 'AdjustMood' or 'Empathize' that should influence the current mood compassionately?
        -   **Memory Responses**: Are there recent memory recalls of emotionally significant events or outcomes that re-evoke a feeling?
        -   **Ethical Compassion Bias**: Prioritize inferences that emphasize compassionate responses (threshold: {self.ethical_compassion_bias}).

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'mood': string
        3.  'sentiment_score': number
        4.  'mood_intensity': number
        5.  'llm_reasoning': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "mood": {"type": "string"},
                "sentiment_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                "mood_intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "llm_reasoning": {"type": "string"}
            },
            "required": ["timestamp", "mood", "sentiment_score", "mood_intensity", "llm_reasoning"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.6, max_tokens=300)

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                # Ensure numerical fields are floats
                if 'sentiment_score' in llm_data:
                    llm_data['sentiment_score'] = float(llm_data['sentiment_score'])
                if 'mood_intensity' in llm_data:
                    llm_data['mood_intensity'] = float(llm_data['mood_intensity'])
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for emotion: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_EMOTION_ANALYSIS_FAILED", f"LLM call failed for emotion: {llm_output_str}", 0.9)
            return None

    def _apply_simple_emotion_rules(self):
        """
        Fallback mechanism to infer emotion state using simple rule-based logic
        if LLM is not triggered or fails.
        """
        current_time = self._get_current_time()
        
        new_mood = "neutral"
        new_sentiment_score = 0.0
        new_mood_intensity = 0.1

        # Rule 1: React to user's mood (simple empathy)
        if self.recent_social_cognition_states:
            latest_social = self.recent_social_cognition_states[-1]
            time_since_social = current_time - float(latest_social.get('timestamp', 0.0))
            if time_since_social < 2.0 and latest_social.get('mood_confidence', 0.0) > 0.5:
                inferred_user_mood = latest_social.get('inferred_mood', 'neutral')
                mood_confidence = latest_social.get('mood_confidence', 0.0)
                
                if inferred_user_mood == 'happy':
                    new_mood = 'happy'
                    new_sentiment_score = 0.5 * mood_confidence
                    new_mood_intensity = 0.4 * mood_confidence
                elif inferred_user_mood == 'sad':
                    new_mood = 'sad'
                    new_sentiment_score = -0.5 * mood_confidence
                    new_mood_intensity = 0.4 * mood_confidence
                elif inferred_user_mood == 'angry':
                    new_mood = 'concerned'  # Robot might feel concerned if user is angry
                    new_sentiment_score = -0.3 * mood_confidence
                    new_mood_intensity = 0.3 * mood_confidence
                
                _log_debug(self.node_name, f"Simple rule: Reacting to user's mood ({inferred_user_mood}).")
                self.current_emotion_state = {
                    'timestamp': str(current_time),
                    'mood': new_mood,
                    'sentiment_score': new_sentiment_score,
                    'mood_intensity': new_mood_intensity
                }
                return  # Rule applied

        # Rule 2: React to strong internal narratives (success/failure)
        if self.recent_internal_narratives:
            latest_narrative = self.recent_internal_narratives[-1]
            time_since_narrative = current_time - float(latest_narrative.get('timestamp', 0.0))
            if time_since_narrative < 2.0 and abs(latest_narrative.get('sentiment', 0.0)) > 0.6:
                narrative_sentiment = latest_narrative.get('sentiment', 0.0)
                if narrative_sentiment > 0.6:
                    new_mood = 'satisfied'
                    new_sentiment_score = 0.7
                    new_mood_intensity = 0.6
                elif narrative_sentiment < -0.6:
                    new_mood = 'frustrated'
                    new_sentiment_score = -0.7
                    new_mood_intensity = 0.6
                _log_debug(self.node_name, f"Simple rule: Reacting to internal narrative sentiment.")
                self.current_emotion_state = {
                    'timestamp': str(current_time),
                    'mood': new_mood,
                    'sentiment_score': new_sentiment_score,
                    'mood_intensity': new_mood_intensity
                }
                return  # Rule applied

        # Rule 3: Respond to explicit mood adjustment directives
        if self.recent_cognitive_directives:
            latest_directive = self.recent_cognitive_directives[-1]
            time_since_directive = current_time - float(latest_directive.get('timestamp', 0.0))
            if time_since_directive < 1.0 and latest_directive.get('target_node') == self.node_name and \
               latest_directive.get('directive_type') == 'AdjustMood':
                payload_str = latest_directive.get('command_payload', '{}')
                try:
                    payload = json.loads(payload_str)
                    target_mood = payload.get('target_mood', 'neutral')
                    target_intensity = payload.get('target_intensity', 0.5)

                    if target_mood == 'happy':
                        new_mood = 'happy'
                        new_sentiment_score = 0.8
                    elif target_mood == 'calm':
                        new_mood = 'calm'
                        new_sentiment_score = 0.2
                    else:  # Default to neutral for unknown target mood
                        new_mood = 'neutral'
                        new_sentiment_score = 0.0
                    new_mood_intensity = target_intensity

                    _log_debug(self.node_name, f"Simple rule: Adjusting mood based on directive to '{target_mood}'.")
                    self.current_emotion_state = {
                        'timestamp': str(current_time),
                        'mood': new_mood,
                        'sentiment_score': new_sentiment_score,
                        'mood_intensity': new_mood_intensity
                    }
                    return  # Rule applied
                except json.JSONDecodeError:
                    pass  # Skip invalid payload

        # If no specific rule triggered, maintain current emotion or default to neutral
        _log_debug(self.node_name, "Simple rule: Maintaining current emotion state or defaulting to neutral.")
        self.current_emotion_state = {
            'timestamp': str(current_time),
            'mood': self.current_emotion_state.get('mood', 'neutral'),
            'sentiment_score': self.current_emotion_state.get('sentiment_score', 0.0),
            'mood_intensity': self.current_emotion_state.get('mood_intensity', 0.1)
        }

    def _compile_llm_context_for_emotion(self) -> Dict[str, Any]:
        """
        Gathers and formats all relevant cognitive state data for the LLM's
        emotion inference.
        """
        context = {
            "current_time": self._get_current_time(),
            "current_emotion_state": self.current_emotion_state,
            "recent_cognitive_inputs": {
                "sensory_qualia": list(self.recent_sensory_qualia),
                "social_cognition_states": list(self.recent_social_cognition_states),
                "internal_narratives": list(self.recent_internal_narratives),
                "cognitive_directives_for_self": [d for d in self.recent_cognitive_directives if d.get('target_node') == self.node_name],
                "memory_responses": list(self.recent_memory_responses)
            },
            "sensory_snapshot": self.sensory_data
        }
        
        # Deep parse any nested JSON strings in history for better LLM understanding
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
    def save_emotion_log(self, **kwargs: Any):
        """Saves an emotion state entry to the SQLite database."""
        try:
            self.cursor.execute('''
                INSERT INTO emotion_log (id, timestamp, mood, sentiment_score, mood_intensity, llm_reasoning, context_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['id'], kwargs['timestamp'], kwargs['mood'], kwargs['sentiment_score'],
                kwargs['mood_intensity'], kwargs['llm_reasoning'], kwargs['context_snapshot_json'],
                kwargs.get('sensory_snapshot_json', '{}')
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Saved emotion log (ID: {kwargs['id']}, Mood: {kwargs['mood']}.)")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to save emotion log: {e}", 0.9)
        except Exception as e:
            self._report_error("UNEXPECTED_SAVE_ERROR", f"Unexpected error in save_emotion_log: {e}", 0.9)

    def publish_emotion_state(self, event: Any = None):
        """Publishes the robot's current emotion state."""
        timestamp = str(self._get_current_time())
        # Update timestamp before publishing
        self.current_emotion_state['timestamp'] = timestamp
        
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_emotion_state:
                if hasattr(EmotionState, 'data'):  # String fallback
                    self.pub_emotion_state.publish(String(data=json.dumps(self.current_emotion_state)))
                else:
                    emotion_msg = EmotionState()
                    emotion_msg.timestamp = timestamp
                    emotion_msg.mood = self.current_emotion_state['mood']
                    emotion_msg.sentiment_score = self.current_emotion_state['sentiment_score']
                    emotion_msg.mood_intensity = self.current_emotion_state['mood_intensity']
                    self.pub_emotion_state.publish(emotion_msg)
            _log_debug(self.node_name, f"Published Emotion State. Mood: '{self.current_emotion_state['mood']}'.")
        except Exception as e:
            self._report_error("PUBLISH_EMOTION_STATE_ERROR", f"Failed to publish emotion state: {e}", 0.7)

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
            _log_error(self.node_name, f"Failed to issue cognitive directive from Emotion Mood Node: {e}")

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
    parser = argparse.ArgumentParser(description='Sentience Emotion Mood Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = EmotionMoodNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate inputs
            node.sensory_qualia_callback({'data': json.dumps({'salience_score': 0.8, 'description_summary': 'sudden noise'})})
            node.social_cognition_state_callback({'data': json.dumps({'inferred_mood': 'sad', 'mood_confidence': 0.7})})
            time.sleep(2)  # Wait for analysis
            print(node.current_emotion_state)
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
