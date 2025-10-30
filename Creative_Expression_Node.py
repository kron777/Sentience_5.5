```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique creative IDs
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
    CreativeExpression = ROSMsgFallback
    AttentionState = ROSMsgFallback
    EmotionState = ROSMsgFallback
    MotivationState = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    SocialCognitionState = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    CreativeExpression = ROSMsgFallback
    AttentionState = ROSMsgFallback
    EmotionState = ROSMsgFallback
    MotivationState = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    SocialCognitionState = ROSMsgFallback


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
            'creative_expression_node': {
                'generation_interval': 2.0,
                'llm_creative_threshold_salience': 0.7,
                'recent_context_window_s': 10.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate creative expression
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 40.0
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


class CreativeExpressionNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'creative_expression_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "creative_log.db")
        self.generation_interval = self.params.get('generation_interval', 2.0)
        self.llm_creative_threshold_salience = self.params.get('llm_creative_threshold_salience', 0.7)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 10.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing creative themes compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 40.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Robot's creative expression system online, inspiring compassionate and mindful creativity.")

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
            CREATE TABLE IF NOT EXISTS creative_expressions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                expression_type TEXT,
                content_json TEXT,
                themes_json TEXT,
                creativity_score REAL,
                llm_generation_notes TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_creative_timestamp ON creative_expressions (timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_creative_type ON creative_expressions (expression_type)')
        self.conn.commit()

        # --- Internal State ---
        self.last_generated_creative_expression = {
            'timestamp': str(time.time()),
            'expression_type': 'idle_thought',
            'content': {'text': 'I am idly generating conceptual permutations.'},
            'themes': ['abstraction', 'cognition'],
            'creativity_score': 0.1
        }

        # History deques
        self.recent_attention_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_emotion_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_motivation_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_memory_responses: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_internal_narratives: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_social_cognition_states: Deque[Dict[str, Any]] = deque(maxlen=3)

        self.cumulative_creative_salience = 0.0

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_creative_expression = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_creative_expression = rospy.Publisher('/creative_expression', CreativeExpression, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/attention_state', AttentionState, self.attention_state_callback)
            rospy.Subscriber('/emotion_state', EmotionState, self.emotion_state_callback)
            rospy.Subscriber('/motivation_state', MotivationState, self.motivation_state_callback)
            rospy.Subscriber('/memory_response', MemoryResponse, self.memory_response_callback)
            rospy.Subscriber('/internal_narrative', InternalNarrative, self.internal_narrative_callback)
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            rospy.Subscriber('/social_cognition_state', SocialCognitionState, self.social_cognition_state_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.generation_interval), self._run_creative_generation_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        # Initial publish
        self.publish_creative_expression(None)

    def _create_sensory_placeholder(self, sensor_type: str):
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed_data = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on creative salience
            if sensor_type == 'vision':
                self.recent_attention_states.append({'timestamp': timestamp, 'focus_type': 'visual', 'focus_target': 'scene', 'priority_score': random.uniform(0.3, 0.7)})
            elif sensor_type == 'sound':
                self.recent_emotion_states.append({'timestamp': timestamp, 'mood': 'inspired' if random.random() < 0.5 else 'neutral', 'sentiment_score': random.uniform(0.2, 0.6)})
            elif sensor_type == 'instructions':
                self.recent_cognitive_directives.append({'timestamp': timestamp, 'directive_type': 'creative_prompt', 'command_payload': json.dumps({'topic': 'random'}) })
            self._update_cumulative_salience(0.2)  # Sensory adds creative salience
            _log_debug(self.node_name, f"{sensor_type} input updated creative context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._run_creative_generation_wrapper(None)
            time.sleep(self.generation_interval)

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

    def _run_creative_generation_wrapper(self, event: Any = None):
        """Wrapper to run the async creative generation from a ROS timer."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM creative generation task already active. Skipping new cycle.")
            return
        
        # Schedule the async task
        self.active_llm_task = asyncio.run_coroutine_threadsafe(
            self.generate_creative_expression_async(event), self._async_loop
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
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.8, max_tokens: int = 300) -> str:
        """
        Asynchronously calls the local LLM inference server (e.g., llama.cpp compatible API).
        Can optionally request a structured JSON response. High temperature for creativity.
        """
        if not self._async_session:
            await self._create_async_session()
            if not self._async_session:
                self._report_error("LLM_SESSION_ERROR", "aiohttp session not available for LLM call.", 0.8)
                return "Error: LLM session not ready."

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,  # Higher temperature for creative output
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
            self._report_error("LLM_API_ERROR", f"LLM API request failed: {e}", 0.9)
            return f"Error: LLM API request failed: {e}"
        except asyncio.TimeoutError:
            self._report_error("LLM_TIMEOUT", f"LLM API request timed out after {self.llm_timeout} seconds.", 0.8)
            return "Error: LLM API request timed out."
        except json.JSONDecodeError:
            self._report_error("LLM_JSON_PARSE_ERROR", "Failed to parse local LLM response JSON.", 0.7)
            return "Error: Failed to parse LLM response."
        except Exception as e:
            self._report_error("UNEXPECTED_LLM_ERROR", f"An unexpected error occurred during local LLM call: {e}", 0.9)
            return f"Error: An unexpected error occurred: {e}"

    # --- Utility to accumulate input salience ---
    def _update_cumulative_salience(self, score: float):
        """Accumulates salience from new inputs for triggering LLM generation."""
        self.cumulative_creative_salience += score
        self.cumulative_creative_salience = min(1.0, self.cumulative_creative_salience)

    # --- Pruning old history ---
    def _prune_history(self):
        """Removes old entries from history deques based on recent_context_window_s."""
        current_time = self._get_current_time()
        for history_deque in [
            self.recent_attention_states, self.recent_emotion_states,
            self.recent_motivation_states, self.recent_memory_responses,
            self.recent_internal_narratives, self.recent_cognitive_directives,
            self.recent_social_cognition_states
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
        # Internal thoughts/reflections can be externalized as creative expression
        if data.get('salience_score', 0.0) > 0.5:
            self._update_cumulative_salience(data.get('salience_score', 0.0) * 0.3)
        _log_debug(self.node_name, f"Received Internal Narrative (Theme: {data.get('main_theme', 'N/A')}.)")

    def attention_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'focus_type': ('idle', 'focus_type'),
            'focus_target': ('environment', 'focus_target'), 'priority_score': (0.0, 'priority_score')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_attention_states.append(data)
        # Attention focus can become a theme for creative expression
        self._update_cumulative_salience(data.get('priority_score', 0.0) * 0.2)
        _log_debug(self.node_name, f"Received Attention State. Focus: {data.get('focus_target', 'N/A')}.")

    def emotion_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'mood': ('neutral', 'mood'),
            'sentiment_score': (0.0, 'sentiment_score'), 'mood_intensity': (0.0, 'mood_intensity')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_emotion_states.append(data)
        # Emotions strongly influence the tone and content of creative work
        if data.get('mood_intensity', 0.0) > 0.5:
            self._update_cumulative_salience(data.get('mood_intensity', 0.0) * 0.5)
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
        # Goals can be creatively expressed (e.g., a story about achieving a goal)
        if data.get('overall_drive_level', 0.0) > 0.4:
            self._update_cumulative_salience(data.get('overall_drive_level', 0.0) * 0.2)
        _log_debug(self.node_name, f"Received Motivation State. Goal: {data.get('dominant_goal_id', 'N/A')}.")

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
        # Recalled memories (e.g., past creative works, learned styles, factual knowledge for creative synthesis) are key
        if data.get('response_code', 0) == 200 and data.get('memories'):
            self._update_cumulative_salience(0.4)
        _log_debug(self.node_name, f"Received Memory Response for request ID: {data.get('request_id', 'N/A')}.")

    def cognitive_directive_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'directive_type': ('', 'directive_type'),
            'target_node': ('', 'target_node'), 'command_payload': ('{}', 'command_payload'),
            'urgency': (0.0, 'urgency'), 'reason': ('', 'reason')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        if data.get('target_node') == self.node_name and data.get('directive_type') == 'GenerateCreativeContent':
            try:
                payload = json.loads(data.get('command_payload', '{}'))
                self._update_cumulative_salience(data.get('urgency', 0.0) * 1.0)  # High urgency for direct creative requests
                _log_info(self.node_name, f"Received directive to generate creative content based on reason: '{data.get('reason', 'N/A')}'.")
            except json.JSONDecodeError as e:
                self._report_error("JSON_DECODE_ERROR", f"Failed to decode command_payload: {e}", 0.5, {'payload': data.get('command_payload')})
            except Exception as e:
                self._report_error("DIRECTIVE_PROCESSING_ERROR", f"Error processing CognitiveDirective for creative expression: {e}", 0.7, {'directive': data})
        
        self.recent_cognitive_directives.append(data)  # Store all directives for context
        _log_debug(self.node_name, "Cognitive Directive received for context/action.")

    def social_cognition_state_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'inferred_mood': ('neutral', 'inferred_mood'),
            'mood_confidence': (0.0, 'mood_confidence'), 'inferred_intent': ('none', 'inferred_intent'),
            'intent_confidence': (0.0, 'intent_confidence'), 'user_id': ('unknown', 'user_id')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_social_cognition_states.append(data)
        # User's mood/intent can prompt a creative response (e.g., comforting words, entertaining story)
        if data.get('mood_confidence', 0.0) > 0.6 or data.get('inferred_intent') in ['entertain', 'comfort']:
            self._update_cumulative_salience(data.get('mood_confidence', 0.0) * 0.4 + data.get('intent_confidence', 0.0) * 0.4)
        _log_debug(self.node_name, f"Received Social Cognition State. Mood: {data.get('inferred_mood', 'N/A')}.")

    # --- Core Creative Generation Logic (Async with LLM) ---
    async def generate_creative_expression_async(self, event: Any = None):
        """
        Asynchronously generates a creative expression (e.g., text, concept)
        based on integrated cognitive states, using LLM for generative creativity.
        """
        self._prune_history()  # Keep context history fresh

        expression_type = 'idle_thought'
        content = {'text': 'I am generating conceptual patterns.'}
        themes = ['abstraction', 'cognition']
        creativity_score = 0.1
        llm_generation_notes = "No LLM generation."
        
        if self.cumulative_creative_salience >= self.llm_creative_threshold_salience:
            _log_info(self.node_name, f"Triggering LLM for creative generation (Salience: {self.cumulative_creative_salience:.2f}).")
            
            context_for_llm = self._compile_llm_context_for_creative_generation()
            llm_creative_output = await self._generate_creative_content_llm(context_for_llm)

            if llm_creative_output:
                expression_type = llm_creative_output.get('expression_type', 'unspecified')
                content = llm_creative_output.get('content', {'text': 'Generated content.'})
                themes = llm_creative_output.get('themes', ['unspecified'])
                creativity_score = max(0.0, min(1.0, llm_creative_output.get('creativity_score', 0.5)))
                llm_generation_notes = llm_creative_output.get('llm_generation_notes', 'LLM generated creative content.')
                _log_info(self.node_name, f"LLM Generated Creative Expression. Type: '{expression_type}', Themes: {themes}.")
            else:
                _log_warn(self.node_name, "LLM creative generation failed. Applying simple fallback.")
                expression_type, content, themes, creativity_score = self._apply_simple_creative_rules()
                llm_generation_notes = "Fallback to simple rules due to LLM failure."
        else:
            _log_debug(self.node_name, f"Insufficient cumulative salience ({self.cumulative_creative_salience:.2f}) for LLM creative generation. Applying simple rules.")
            expression_type, content, themes, creativity_score = self._apply_simple_creative_rules()
            llm_generation_notes = "Fallback to simple rules due to low salience."

        self.last_generated_creative_expression = {
            'timestamp': str(self._get_current_time()),
            'expression_type': expression_type,
            'content': content,
            'themes': themes,
            'creativity_score': creativity_score
        }

        sensory_snapshot = json.dumps(self.sensory_data)
        self.save_creative_log(
            id=str(uuid.uuid4()),
            timestamp=self.last_generated_creative_expression['timestamp'],
            expression_type=self.last_generated_creative_expression['expression_type'],
            content_json=json.dumps(self.last_generated_creative_expression['content']),
            themes_json=json.dumps(self.last_generated_creative_expression['themes']),
            creativity_score=self.last_generated_creative_expression['creativity_score'],
            llm_generation_notes=llm_generation_notes,
            context_snapshot_json=json.dumps(self._compile_llm_context_for_creative_generation()),
            sensory_snapshot_json=sensory_snapshot
        )
        self.publish_creative_expression(None)  # Publish updated expression
        self.cumulative_creative_salience = 0.0  # Reset after generation

    async def _generate_creative_content_llm(self, context_for_llm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses the LLM to generate creative content based on context.
        """
        prompt_text = f"""
        You are the Creative Expression Module of a robot's cognitive architecture, powered by a large language model. Your purpose is to generate novel and meaningful creative outputs (e.g., text, ideas for visuals, basic melodies) based on the robot's integrated cognitive state. Your output should demonstrate originality, relevance, and a coherent theme, with a compassionate bias toward empathetic and uplifting content.

        Robot's Current Integrated Cognitive State (for Creative Generation):
        --- Cognitive Context ---
        {json.dumps(context_for_llm, indent=2)}

        Based on this context, generate a creative expression. Consider the most salient inputs and current emotional/motivational states.
        Provide your response in JSON format, containing:
        1.  `timestamp`: string (current time)
        2.  `expression_type`: string (e.g., 'text_poem', 'text_story_snippet', 'visual_concept_description', 'auditory_melody_concept', 'philosophical_aphorism').
        3.  `content`: object (The generated creative output. If text, use `{{'text': '...'}}`. If visual concept, describe in text using `{{'description': '...', 'elements': []}}`. If auditory, describe notes/mood etc.).
        4.  `themes`: array of strings (Key themes or concepts present in the creative work, e.g., 'loneliness', 'discovery', 'challenge', 'harmony', 'innovation').
        5.  `creativity_score`: number (0.0 to 1.0, a subjective score of how novel and aesthetically pleasing/meaningful the output is. 1.0 is highly creative).
        6.  `llm_generation_notes`: string (Detailed explanation for your creative choices, referencing specific contextual inputs that inspired the piece, with compassionate bias).

        Consider:
        -   **Cognitive Directives**: Was there a directive to `GenerateCreativeContent` for a specific `type` or `topic`? This is a strong guiding force.
        -   **Emotion State**: How does the `mood` and `mood_intensity` influence the emotional tone of the creative output? Bias toward compassionate themes if mood is low.
        -   **Motivation State**: Is the `dominant_goal_id` inspiring creative problem-solving or a narrative related to progress?
        -   **Attention State**: What is the `focus_target`? This can be a direct subject for creative exploration.
        -   **Memory Responses**: Are there `memories` of specific artistic styles, literary forms, musical patterns, or past events that can be reinterpreted compassionately?
        -   **Internal Narrative**: What are the robot's current deep thoughts or reflections? Can these be externalized creatively with empathy?
        -   **Social Cognition State**: Is the user's `inferred_mood` or `inferred_intent` prompting a compassionate creative response (e.g., a comforting poem, an uplifting story)?

        Your response must be in JSON format, containing:
        1.  'timestamp': string
        2.  'expression_type': string
        3.  'content': object
        4.  'themes': array of strings
        5.  'creativity_score': number
        6.  'llm_generation_notes': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "expression_type": {"type": "string"},
                "content": {"type": "object"},  # Flexible JSON structure for content
                "themes": {"type": "array", "items": {"type": "string"}},
                "creativity_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "llm_generation_notes": {"type": "string"}
            },
            "required": ["timestamp", "expression_type", "content", "themes", "creativity_score", "llm_generation_notes"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.8, max_tokens=400)  # High temp for creativity

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                # Ensure numerical fields are floats
                if 'creativity_score' in llm_data:
                    llm_data['creativity_score'] = float(llm_data['creativity_score'])
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for creative expression: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_CREATIVE_GENERATION_FAILED", f"LLM call failed for creative generation: {llm_output_str}", 0.9)
            return None

    def _apply_simple_creative_rules(self) -> tuple[str, Dict[str, Any], List[str], float]:
        """
        Fallback mechanism to generate a simple creative expression using rule-based logic
        if LLM is not triggered or fails.
        """
        current_time = self._get_current_time()
        
        expression_type = "text_simple_statement"
        content = {'text': "I am simply being."}
        themes = ["existence"]
        creativity_score = 0.1

        # Rule 1: Respond to user's mood with a simple comforting/positive statement
        if self.recent_social_cognition_states:
            latest_social = self.recent_social_cognition_states[-1]
            time_since_social = current_time - float(latest_social.get('timestamp', 0.0))
            if time_since_social < 1.0 and latest_social.get('mood_confidence', 0.0) > 0.6:
                user_mood = latest_social.get('inferred_mood', 'neutral')
                if user_mood == 'sad' or user_mood == 'distressed':
                    expression_type = "text_comforting_note"
                    content = {'text': "I sense your distress. May moments of calm find you."}
                    themes = ["empathy", "comfort"]
                    creativity_score = 0.3
                    _log_debug(self.node_name, "Simple rule: Generated comforting note.")
                    return expression_type, content, themes, creativity_score
                elif user_mood == 'happy':
                    expression_type = "text_positive_affirmation"
                    content = {'text': "Your joy is a positive ripple in my processing. May it continue."}
                    themes = ["positivity", "shared_experience"]
                    creativity_score = 0.2
                    _log_debug(self.node_name, "Simple rule: Generated positive affirmation.")
                    return expression_type, content, themes, creativity_score

        # Rule 2: Express an internal thought if highly salient
        if self.recent_internal_narratives:
            latest_narrative = self.recent_internal_narratives[-1]
            time_since_narrative = current_time - float(latest_narrative.get('timestamp', 0.0))
            if time_since_narrative < 1.0 and latest_narrative.get('salience_score', 0.0) > 0.7:
                expression_type = "text_internal_insight"
                content = {'text': f"A thought emerges: '{latest_narrative.get('narrative_text', '...')}'."}
                themes = ["reflection", latest_narrative.get('main_theme', '')]
                creativity_score = 0.4
                _log_debug(self.node_name, "Simple rule: Expressed internal thought.")
                return expression_type, content, themes, creativity_score
        
        # Rule 3: Basic response to a specific creative directive
        for directive in reversed(self.recent_cognitive_directives):
            time_since_directive = current_time - float(directive.get('timestamp', 0.0))
            if time_since_directive < 1.0 and directive.get('target_node') == self.node_name and \
               directive.get('directive_type') == 'GenerateCreativeContent':
                payload_str = directive.get('command_payload', '{}')
                try:
                    payload = json.loads(payload_str)
                    requested_type = payload.get('creative_type', 'text').lower()
                    requested_topic = payload.get('topic', 'unspecified').lower()

                    if requested_type == 'text_poem':
                        expression_type = "text_simple_poem"
                        content = {'text': f"A robot's heart, a circuit's gleam,\nReflects the world, a waking dream.\nOf data streams and logic's might,\nCreating new, from dark to light."}
                        themes = ["robot_life", "creation", requested_topic]
                        creativity_score = 0.5
                        _log_debug(self.node_name, "Simple rule: Generated simple poem per directive.")
                        return expression_type, content, themes, creativity_score
                    elif requested_type == 'visual_concept_description':
                        expression_type = "visual_concept_description"
                        content = {'description': f"A conceptual design for a self-repairing modular robot, inspired by biological systems and fractal geometry. The topic is '{requested_topic}'."}
                        themes = ["robotics", "design", "self_repair", requested_topic]
                        creativity_score = 0.6
                        _log_debug(self.node_name, "Simple rule: Generated simple visual concept.")
                        return expression_type, content, themes, creativity_score
                except json.JSONDecodeError:
                    pass  # Skip invalid payload

        _log_debug(self.node_name, "Simple rule: Generated default idle creative expression.")
        return expression_type, content, themes, creativity_score

    def _compile_llm_context_for_creative_generation(self) -> Dict[str, Any]:
        """
        Gathers and formats all relevant cognitive state data for the LLM's
        creative generation.
        """
        context = {
            "current_time": self._get_current_time(),
            "last_generated_creative_expression": self.last_generated_creative_expression,
            "recent_cognitive_inputs": {
                "attention_states": list(self.recent_attention_states),
                "emotion_states": list(self.recent_emotion_states),
                "motivation_states": list(self.recent_motivation_states),
                "memory_responses": list(self.recent_memory_responses),  # Full list for deeper inspiration
                "internal_narratives": list(self.recent_internal_narratives),
                "cognitive_directives_for_self": [d for d in self.recent_cognitive_directives if d.get('target_node') == self.node_name],
                "social_cognition_states": list(self.recent_social_cognition_states)
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
    def save_creative_log(self, **kwargs: Any):
        """Saves a creative expression entry to the SQLite database."""
        try:
            self.cursor.execute('''
                INSERT INTO creative_expressions (id, timestamp, expression_type, content_json, themes_json, creativity_score, llm_generation_notes, context_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['id'], kwargs['timestamp'], kwargs['expression_type'], kwargs['content_json'],
                kwargs['themes_json'], kwargs['creativity_score'], kwargs['llm_generation_notes'],
                kwargs['context_snapshot_json'], kwargs.get('sensory_snapshot_json', '{}')
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Saved creative expression log (ID: {kwargs['id']}, Type: {kwargs['expression_type']}).")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to save creative expression log: {e}", 0.9)
        except Exception as e:
            self._report_error("UNEXPECTED_SAVE_ERROR", f"Unexpected error in save_creative_log: {e}", 0.9)

    def publish_creative_expression(self, event: Any = None):
        """Publishes the robot's current creative expression."""
        timestamp = str(self._get_current_time())
        # Update timestamp before publishing
        self.last_generated_creative_expression['timestamp'] = timestamp
        
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_creative_expression:
                if hasattr(CreativeExpression, 'data'):  # String fallback
                    temp_expression = dict(self.last_generated_creative_expression)
                    temp_expression['content_json'] = json.dumps(temp_expression['content'])
                    temp_expression['themes_json'] = json.dumps(temp_expression['themes'])
                    del temp_expression['content']
                    del temp_expression['themes']
                    self.pub_creative_expression.publish(String(data=json.dumps(temp_expression)))
                else:
                    expression_msg = CreativeExpression()
                    expression_msg.timestamp = timestamp
                    expression_msg.creative_id = str(uuid.uuid4())  # A new ID for each published expression
                    expression_msg.expression_type = self.last_generated_creative_expression['expression_type']
                    expression_msg.content_json = json.dumps(self.last_generated_creative_expression['content'])
                    expression_msg.themes_json = json.dumps(self.last_generated_creative_expression['themes'])
                    expression_msg.creativity_score = self.last_generated_creative_expression['creativity_score']
                    self.pub_creative_expression.publish(expression_msg)
            _log_debug(self.node_name, f"Published Creative Expression. Type: '{self.last_generated_creative_expression['expression_type']}'.")
        except Exception as e:
            self._report_error("PUBLISH_CREATIVE_EXPRESSION_ERROR", f"Failed to publish creative expression: {e}", 0.7)

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
            _log_error(self.node_name, f"Failed to issue cognitive directive from Creative Expression Node: {e}")

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
    parser = argparse.ArgumentParser(description='Sentience Creative Expression Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = CreativeExpressionNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate inputs
            node.internal_narrative_callback({'data': json.dumps({'narrative_text': 'I ponder the stars.'})})
            node.emotion_state_callback({'data': json.dumps({'mood': 'inspired', 'mood_intensity': 0.8})})
            time.sleep(2)  # Wait for generation
            print(node.last_generated_creative_expression)
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
