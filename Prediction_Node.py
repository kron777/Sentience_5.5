```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique memory IDs and request IDs
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
    MemoryRequest = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    WorldModelState = ROSMsgFallback
    SocialCognitionState = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    MemoryRequest = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
    WorldModelState = ROSMsgFallback
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
            'memory_node': {
                'memory_processing_interval': 0.1,
                'llm_memory_threshold_salience': 0.7,
                'recent_context_window_s': 30.0,
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate memory retention (e.g., preserve empathetic memories)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 45.0
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


class MemoryNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'memory_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "long_term_memory.db")
        self.memory_processing_interval = self.params.get('memory_processing_interval', 0.1)
        self.llm_memory_threshold_salience = self.params.get('llm_memory_threshold_salience', 0.7)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 30.0)
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing memory compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 45.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Robot's memory system online, compassionately preserving compassionate experiences.")

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
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                category TEXT,
                content TEXT,
                keywords TEXT,
                source_node TEXT,
                salience REAL,
                llm_processing_notes TEXT,
                original_context_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_keywords ON memories (keywords)')
        self.conn.commit()

        # --- Internal State ---
        self.memory_request_queue: Deque[Dict[str, Any]] = deque()

        # History deques
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_internal_narratives: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_world_model_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_social_cognition_states: Deque[Dict[str, Any]] = deque(maxlen=5)

        self.cumulative_memory_salience = 0.0

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_memory_response = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_memory_response = rospy.Publisher('/memory_response', MemoryResponse, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/memory_request', MemoryRequest, self.memory_request_callback)
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            rospy.Subscriber('/internal_narrative', InternalNarrative, self.internal_narrative_callback)
            rospy.Subscriber('/world_model_state', WorldModelState, self.world_model_state_callback)
            rospy.Subscriber('/social_cognition_state', SocialCognitionState, self.social_cognition_state_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.memory_processing_interval), self._run_memory_processing_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing memory compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on memory (e.g., high salience events auto-queue for storage)
            if sensor_type == 'vision':
                self.memory_request_queue.append({
                    'request_type': 'store', 'category': 'episodic', 'content_json': json.dumps({'visual': processed}),
                    'salience': random.uniform(0.4, 0.8), 'timestamp': timestamp
                })
            elif sensor_type == 'sound':
                self.recent_social_cognition_states.append({'timestamp': timestamp, 'inferred_mood': 'reactive' if random.random() < 0.5 else 'neutral'})
            elif sensor_type == 'instructions':
                self.recent_cognitive_directives.append({'timestamp': timestamp, 'directive_type': 'store_memory', 'command_payload': json.dumps({'query_text': 'user command'})})
            # Compassionate bias: If distress in sound, boost salience for empathetic memories
            if 'distress' in str(processed):
                self.cumulative_memory_salience = min(1.0, self.cumulative_memory_salience + self.ethical_compassion_bias)
            _log_debug(self.node_name, f"{sensor_type} input updated memory context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._run_memory_processing_wrapper(None)
            time.sleep(self.memory_processing_interval)

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

    def _run_memory_processing_wrapper(self, event: Any = None):
        """Wrapper to run the async memory processing from a timer."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM memory processing task already active. Skipping new cycle.")
            return

        if self.memory_request_queue:
            request_data = self.memory_request_queue.popleft()
            self.active_llm_task = asyncio.run_coroutine_threadsafe(
                self.process_memory_request_async(request_data, event), self._async_loop
            )
        else:
            _log_debug(self.node_name, "No memory requests in queue.")

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
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.2, max_tokens: int = None) -> str:
        """
        Asynchronously calls the local LLM inference server (e.g., llama.cpp compatible API).
        Can optionally request a structured JSON response. Low temperature for factual/summarization.
        """
        if not self._async_session:
            await self._create_async_session()
            if not self._async_session:
                self._report_error("LLM_SESSION_ERROR", "aiohttp session not available for LLM call.", 0.8)
                return "Error: LLM session not ready."

        actual_max_tokens = max_tokens if max_tokens is not None else 600  # Higher max_tokens for summarization

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,  # Low temperature for factual consistency
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
        """Accumulates salience from new inputs for triggering LLM operations."""
        self.cumulative_memory_salience += score
        self.cumulative_memory_salience = min(1.0, self.cumulative_memory_salience)

    # --- Pruning old history ---
    def _prune_history(self):
        """Removes old entries from history deques based on recent_context_window_s."""
        current_time = self._get_current_time()
        for history_deque in [
            self.recent_cognitive_directives, self.recent_internal_narratives,
            self.recent_world_model_states, self.recent_social_cognition_states
        ]:
            while history_deque and (current_time - float(history_deque[0].get('timestamp', 0.0))) > self.recent_context_window_s:
                history_deque.popleft()

    # --- Callbacks (generic, ROS or direct) ---
    def memory_request_callback(self, msg: Any):
        """Handle incoming memory requests."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'request_id': (str(uuid.uuid4()), 'request_id'),
            'request_type': ('retrieve', 'request_type'),  # 'store', 'retrieve', 'update', 'delete', 'summarize'
            'category': ('general', 'category'),
            'query_text': ('', 'query_text'),
            'content_json': ('{}', 'content_json'),  # For 'store' or 'update' requests
            'keywords': ('', 'keywords'),
            'num_results': (1, 'num_results'),  # For 'retrieve' requests
            'salience': (0.5, 'salience'),  # For 'store' requests, importance of the memory
            'source_node': (self.node_name, 'source_node')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        # Parse content_json if it's a string
        if isinstance(data.get('content_json'), str):
            try:
                data['content_parsed'] = json.loads(data['content_json'])
            except json.JSONDecodeError:
                data['content_parsed'] = {}  # Fallback if not valid JSON
        else:
            data['content_parsed'] = data.get('content_json', {})  # Ensure it's a dict

        self.memory_request_queue.append(data)
        # Salience of the request directly influences LLM trigger
        self._update_cumulative_salience(data.get('salience', 0.5) * 0.7)  # Memory requests are usually important
        _log_info(self.node_name, f"Queued memory request (ID: {data['request_id']}, Type: {data['request_type']}). Queue size: {len(self.memory_request_queue)}.")

    def cognitive_directive_callback(self, msg: Any):
        """Handle incoming cognitive directives."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'directive_type': ('', 'directive_type'),
            'target_node': ('', 'target_node'), 'command_payload': ('{}', 'command_payload'),
            'urgency': (0.0, 'urgency'), 'reason': ('', 'reason')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        if data.get('target_node') == self.node_name:
            self.recent_cognitive_directives.append(data)  # Add directives for self to context
            # Directives for memory management (e.g., 'OptimizeMemory', 'ForgetMemory') are highly salient
            if data.get('directive_type') in ['OptimizeMemory', 'ForgetMemory', 'SummarizeMemory']:
                self._update_cumulative_salience(data.get('urgency', 0.0) * 0.9)
            _log_info(self.node_name, f"Received directive for self: '{data.get('directive_type', 'N/A')}' (Payload: {data.get('command_payload', 'N/A')}.)")
        else:
            self.recent_cognitive_directives.append(data)  # Add all directives for general context
        _log_debug(self.node_name, "Cognitive Directive received for context/action.")

    def internal_narrative_callback(self, msg: Any):
        """Handle incoming internal narrative data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'narrative_text': ('', 'narrative_text'),
            'main_theme': ('', 'main_theme'), 'sentiment': (0.0, 'sentiment'), 'salience_score': (0.0, 'salience_score')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_internal_narratives.append(data)
        # Internal narratives can be good candidates for narrative memory storage
        if data.get('salience_score', 0.0) > 0.6:
            # Auto-queue high salience internal narratives for storage
            self.memory_request_queue.append({
                'timestamp': data['timestamp'],
                'request_id': str(uuid.uuid4()),
                'request_type': 'store',
                'category': 'narrative',
                'query_text': '',
                'content_json': json.dumps({'text': data['narrative_text'], 'theme': data['main_theme'], 'sentiment': data['sentiment']}),
                'keywords': data['main_theme'].replace(' ', ','),
                'num_results': 0,
                'salience': data['salience_score'],
                'source_node': 'internal_narrative_node'
            })
            self._update_cumulative_salience(data.get('salience_score', 0.0) * 0.3)
        _log_debug(self.node_name, f"Received Internal Narrative (Theme: {data.get('main_theme', 'N/A')}.)")

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
        # Significant world model changes can be stored as episodic memories
        if data.get('significant_change_flag', False):
            # Auto-queue significant world model changes for storage
            self.memory_request_queue.append({
                'timestamp': data['timestamp'],
                'request_id': str(uuid.uuid4()),
                'request_type': 'store',
                'category': 'episodic',
                'query_text': '',
                'content_json': json.dumps({'description': f"Significant change in world state: {len(data['changed_entities'])} entities changed.", 'details': data}),
                'keywords': "world_change,environment",
                'num_results': 0,
                'salience': 0.7,  # High salience for important world changes
                'source_node': 'world_model_node'
            })
            self._update_cumulative_salience(0.4)
        _log_debug(self.node_name, f"Received World Model State. Significant Change: {data.get('significant_change_flag', False)}.")

    def social_cognition_state_callback(self, msg: Any):
        """Handle incoming social cognition state data."""
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'inferred_mood': ('neutral', 'inferred_mood'),
            'mood_confidence': (0.0, 'mood_confidence'), 'inferred_intent': ('none', 'inferred_intent'),
            'intent_confidence': (0.0, 'intent_confidence'), 'user_id': ('unknown', 'user_id')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        self.recent_social_cognition_states.append(data)
        # Important social interactions can be stored as social memories
        if data.get('intent_confidence', 0.0) > 0.6 or data.get('mood_confidence', 0.0) > 0.6:
            # Auto-queue salient social interactions for storage
            self.memory_request_queue.append({
                'timestamp': data['timestamp'],
                'request_id': str(uuid.uuid4()),
                'request_type': 'store',
                'category': 'social',
                'query_text': '',
                'content_json': json.dumps({'user_id': data['user_id'], 'inferred_mood': data['inferred_mood'], 'inferred_intent': data['inferred_intent']}),
                'keywords': f"social,user_{data['user_id']}",
                'num_results': 0,
                'salience': data.get('intent_confidence', 0.0) * 0.5 + data.get('mood_confidence', 0.0) * 0.5,
                'source_node': 'social_cognition_node'
            })
            self._update_cumulative_salience(0.3)
        _log_debug(self.node_name, f"Received Social Cognition State. Intent: {data.get('inferred_intent', 'N/A')}.")

    # --- Core Memory Processing Logic (Async with LLM) ---
    async def process_memory_request_async(self, request_data: Dict[str, Any], event: Any = None):
        """
        Asynchronously processes a memory request (store, retrieve, update, delete, summarize)
        using LLM for advanced operations with compassionate bias for emotional memories.
        """
        self._prune_history()  # Keep context history fresh

        request_id = request_data.get('request_id')
        request_type = request_data.get('request_type')
        category = request_data.get('category')
        query_text = request_data.get('query_text')
        content_json = request_data.get('content_json')  # Original JSON string
        content_parsed = request_data.get('content_parsed')  # Already parsed dict
        keywords = request_data.get('keywords')
        num_results = request_data.get('num_results', 1)
        salience = request_data.get('salience', 0.5)
        source_node = request_data.get('source_node', 'unknown')

        memory_response_code = 500  # Default to error
        retrieved_memories = []
        llm_processing_notes = "No LLM processing."
        original_context_snapshot = self._compile_llm_context_for_memory_op(request_data)

        # Compassionate bias: Boost salience for emotional/social memories
        if 'social' in category or 'emotional' in category:
            salience = min(1.0, salience + self.ethical_compassion_bias * 0.2)

        if request_type == 'store':
            # LLM can process content to extract better keywords or summarize
            if self.cumulative_memory_salience >= self.llm_memory_threshold_salience and salience >= 0.6:
                _log_info(self.node_name, f"Triggering LLM for memory store processing (ID: {request_id}, Category: {category}).")
                llm_processed_memory = await self._process_memory_content_llm(content_parsed, category, keywords, original_context_snapshot)
                if llm_processed_memory:
                    content_to_store = llm_processed_memory.get('processed_content', content_parsed)
                    keywords_to_store = llm_processed_memory.get('extracted_keywords', keywords)
                    llm_processing_notes = llm_processed_memory.get('processing_notes', 'LLM processed content.')
                    # Ensure content_to_store is a JSON string before passing to save
                    if isinstance(content_to_store, dict):
                        content_to_store_json = json.dumps(content_to_store)
                    else:
                        content_to_store_json = str(content_to_store)  # Fallback to string if not dict
                
                    self.store_memory(
                        id=str(uuid.uuid4()),
                        timestamp=str(self._get_current_time()),
                        category=category,
                        content=content_to_store_json,
                        keywords=keywords_to_store,
                        source_node=source_node,
                        salience=salience,
                        llm_processing_notes=llm_processing_notes,
                        original_context_json=json.dumps(original_context_snapshot),
                        sensory_snapshot_json=json.dumps(self.sensory_data)
                    )
                    memory_response_code = 200
                else:
                    _log_warn(self.node_name, f"LLM failed to process memory for storage. Storing raw content. (ID: {request_id})")
                    self.store_memory(
                        id=str(uuid.uuid4()),
                        timestamp=str(self._get_current_time()),
                        category=category,
                        content=content_json,  # Store original raw JSON string
                        keywords=keywords,
                        source_node=source_node,
                        salience=salience,
                        llm_processing_notes="LLM processing failed, raw content stored.",
                        original_context_json=json.dumps(original_context_snapshot),
                        sensory_snapshot_json=json.dumps(self.sensory_data)
                    )
                    memory_response_code = 200
            else:
                _log_debug(self.node_name, "Insufficient salience for LLM memory processing for store. Storing raw content.")
                self.store_memory(
                    id=str(uuid.uuid4()),
                    timestamp=str(self._get_current_time()),
                    category=category,
                    content=content_json,  # Store original raw JSON string
                    keywords=keywords,
                    source_node=source_node,
                    salience=salience,
                    llm_processing_notes="Low salience, raw content stored.",
                    original_context_json=json.dumps(original_context_snapshot),
                    sensory_snapshot_json=json.dumps(self.sensory_data)
                )
                memory_response_code = 200
            self.cumulative_memory_salience = 0.0  # Reset after store

        elif request_type == 'retrieve':
            # LLM can assist with semantic search and re-ranking or summarizing retrieved memories
            if self.cumulative_memory_salience >= self.llm_memory_threshold_salience:
                _log_info(self.node_name, f"Triggering LLM for semantic memory retrieval (ID: {request_id}, Query: '{query_text}').")
                # First, retrieve a broader set of potentially relevant memories from DB
                raw_retrieved = self.retrieve_memories_from_db(query_text, category, keywords, num_results * 2)  # Get more for LLM re-ranking
                
                if raw_retrieved:
                    llm_retrieved = await self._semantic_retrieve_memories_llm(query_text, raw_retrieved, num_results, original_context_snapshot)
                    if llm_retrieved:
                        retrieved_memories = llm_retrieved.get('ranked_memories', [])
                        llm_processing_notes = llm_retrieved.get('retrieval_notes', 'LLM assisted retrieval.')
                        memory_response_code = 200
                    else:
                        _log_warn(self.node_name, f"LLM semantic retrieval failed. Returning raw database results. (ID: {request_id})")
                        retrieved_memories = raw_retrieved[:num_results]
                        llm_processing_notes = "LLM retrieval failed, returning simple DB results."
                        memory_response_code = 200
                else:
                    _log_debug(self.node_name, f"No raw memories found for query '{query_text}'.")
                    memory_response_code = 404  # Not found
            else:
                _log_debug(self.node_name, "Insufficient salience for LLM semantic retrieval. Performing simple keyword retrieval.")
                retrieved_memories = self.retrieve_memories_from_db(query_text, category, keywords, num_results)
                llm_processing_notes = "Low salience, simple keyword retrieval."
                memory_response_code = 200 if retrieved_memories else 404
            self.cumulative_memory_salience = 0.0  # Reset after retrieve

        elif request_type == 'update':
            # LLM can help understand the update and apply it intelligently
            if self.cumulative_memory_salience >= self.llm_memory_threshold_salience and salience >= 0.6:
                _log_info(self.node_name, f"Triggering LLM for memory update processing (ID: {request_id}).")
                existing_memories = self.retrieve_memories_from_db(query_text, category, keywords, 1)  # Get the most relevant memory
                if existing_memories:
                    updated_content_llm = await self._update_memory_content_llm(existing_memories[0], content_parsed, original_context_snapshot)
                    if updated_content_llm:
                        updated_memory_id = existing_memories[0]['id']
                        content_to_update = updated_content_llm.get('updated_content', content_parsed)
                        llm_processing_notes = updated_content_llm.get('processing_notes', 'LLM updated content.')
                        if isinstance(content_to_update, dict):
                            content_to_update_json = json.dumps(content_to_update)
                        else:
                            content_to_update_json = str(content_to_update)
                        self.update_memory(updated_memory_id, content_to_update_json, keywords, salience, llm_processing_notes, json.dumps(original_context_snapshot), json.dumps(self.sensory_data))
                        memory_response_code = 200
                    else:
                        _log_warn(self.node_name, f"LLM memory update failed. Attempting simple update. (ID: {request_id})")
                        if existing_memories and existing_memories[0].get('id'):
                            self.update_memory(existing_memories[0]['id'], content_json, keywords, salience, "LLM update failed, simple update applied.", json.dumps(original_context_snapshot), json.dumps(self.sensory_data))
                            memory_response_code = 200
                        else:
                            memory_response_code = 404  # Not found
                else:
                    memory_response_code = 404  # Not found
            else:
                _log_debug(self.node_name, "Insufficient salience for LLM memory update. Performing simple update.")
                existing_memories = self.retrieve_memories_from_db(query_text, category, keywords, 1)
                if existing_memories and existing_memories[0].get('id'):
                    self.update_memory(existing_memories[0]['id'], content_json, keywords, salience, "Low salience, simple update applied.", json.dumps(original_context_snapshot), json.dumps(self.sensory_data))
                    memory_response_code = 200
                else:
                    memory_response_code = 404  # Not found
            self.cumulative_memory_salience = 0.0  # Reset after update

        elif request_type == 'delete':
            # Simple delete, LLM not strictly needed but could confirm
            num_deleted = self.delete_memory(query_text, category, keywords)
            memory_response_code = 200 if num_deleted > 0 else 404
            _log_info(self.node_name, f"Deleted {num_deleted} memories for query '{query_text}'.")
            self.cumulative_memory_salience = 0.0  # Reset after delete

        elif request_type == 'summarize':
            # LLM is essential for summarization
            if self.cumulative_memory_salience >= self.llm_memory_threshold_salience:
                _log_info(self.node_name, f"Triggering LLM for memory summarization (ID: {request_id}, Query: '{query_text}').")
                memories_to_summarize = self.retrieve_memories_from_db(query_text, category, keywords, 10)  # Get a batch of memories
                if memories_to_summarize:
                    summary_output = await self._summarize_memories_llm(memories_to_summarize, original_context_snapshot)
                    if summary_output:
                        retrieved_memories = [{"category": "summary", "content": summary_output.get('summary_text', "Could not generate summary."), "keywords": "summary", "id": str(uuid.uuid4()), "timestamp": str(self._get_current_time())}]
                        llm_processing_notes = summary_output.get('summarization_notes', 'LLM generated summary.')
                        memory_response_code = 200
                    else:
                        _log_warn(self.node_name, f"LLM summarization failed. (ID: {request_id})")
                        retrieved_memories = []
                        llm_processing_notes = "LLM summarization failed."
                        memory_response_code = 500  # Internal error
                else:
                    _log_warn(self.node_name, f"No memories found to summarize for query '{query_text}'.")
                    memory_response_code = 404  # Not found
            else:
                _log_warn(self.node_name, "Insufficient salience for LLM summarization. Summarization skipped.")
                memory_response_code = 400  # Bad request, or not enough salience
            self.cumulative_memory_salience = 0.0  # Reset after summarize

        else:
            _log_warn(self.node_name, f"Unknown memory request type: {request_type} for ID: {request_id}.")
            memory_response_code = 400  # Bad Request

        # Publish the response
        self.publish_memory_response(
            request_id=request_id,
            response_code=memory_response_code,
            memories_json=json.dumps(retrieved_memories)  # Send back as JSON string
        )

    async def _process_memory_content_llm(self, content_dict: Dict[str, Any], category: str, keywords_hint: str, context_snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses LLM to process new memory content for better categorization, keyword extraction, or summarization before storage.
        """
        prompt_text = f"""
        You are the Memory Node's LLM assistant. Your task is to process incoming raw memory content to enhance its quality for storage and future retrieval. Analyze the `raw_content` provided, considering the `category_hint` and `keywords_hint`.

        Raw Memory Content:
        --- Raw Content ---
        {json.dumps(content_dict, indent=2)}

        Category Hint: '{category}'
        Keywords Hint: '{keywords_hint}'

        Robot's Current Cognitive Context (for better understanding of memory's significance):
        --- Cognitive Context ---
        {json.dumps(context_snapshot, indent=2)}

        Based on this, propose:
        1.  `processed_content`: object (A refined, possibly summarized or structured version of the content, if beneficial. If the content is already good, return it as is. For text, ensure a 'text' key. For complex data, structure as a meaningful JSON object.)
        2.  `extracted_keywords`: string (A comma-separated list of highly relevant keywords for this memory, improving searchability. Incorporate `keywords_hint` but expand or refine.)
        3.  `processing_notes`: string (Brief notes on how you processed or understood this memory.)

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'processed_content': object
        3.  'extracted_keywords': string
        4.  'processing_notes': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "processed_content": {"type": "object"},  # Flexible JSON structure
                "extracted_keywords": {"type": "string"},
                "processing_notes": {"type": "string"}
            },
            "required": ["timestamp", "processed_content", "extracted_keywords", "processing_notes"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.2, max_tokens=600)

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for memory processing: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_MEMORY_PROCESS_FAILED", f"LLM call failed for memory processing: {llm_output_str}", 0.9)
            return None

    async def _semantic_retrieve_memories_llm(self, query_text: str, raw_memories: List[Dict[str, Any]], num_results: int, context_snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses LLM to perform semantic retrieval and re-ranking of memories.
        `raw_memories` are initial candidates retrieved by simple keyword matching.
        """
        memories_str_for_llm = "\n".join([f"- Memory ID: {m.get('id')}\n  Category: {m.get('category')}\n  Keywords: {m.get('keywords')}\n  Content Summary: {m.get('content', '')[:150]}..." for m in raw_memories])

        prompt_text = f"""
        You are the Memory Node's LLM assistant. Your task is to semantically retrieve and rank relevant memories based on a `user_query` from a list of `candidate_memories`. Also consider the robot's `current_cognitive_context` for a more nuanced understanding of the query's intent.

        User Query: '{query_text}'

        Candidate Memories (from initial database search):
        --- Candidate Memories ---
        {memories_str_for_llm}

        Robot's Current Cognitive Context (for query intent and relevance):
        --- Cognitive Context ---
        {json.dumps(context_snapshot, indent=2)}

        Based on the `user_query` and `current_cognitive_context`, identify the {num_results} most relevant memories from the `candidate_memories`.
        Provide your response in JSON format, containing:
        1.  `timestamp`: string (current time)
        2.  `ranked_memories`: array of objects (The selected relevant memories, each including their full original content and metadata. Ordered by relevance, highest first.)
        3.  `retrieval_notes`: string (Brief explanation for the selection and ranking.)

        Ensure each memory in `ranked_memories` includes its 'id', 'timestamp', 'category', 'content', 'keywords', 'source_node', 'salience'.
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "ranked_memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "timestamp": {"type": "string"},
                            "category": {"type": "string"},
                            "content": {"type": "string"},
                            "keywords": {"type": "string"},
                            "source_node": {"type": "string"},
                            "salience": {"type": "number"}
                        },
                        "required": ["id", "timestamp", "category", "content", "keywords", "source_node", "salience"]
                    }
                },
                "retrieval_notes": {"type": "string"}
            },
            "required": ["timestamp", "ranked_memories", "retrieval_notes"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.1, max_tokens=800)

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                # Ensure salience is float
                if 'ranked_memories' in llm_data:
                    for mem in llm_data['ranked_memories']:
                        if 'salience' in mem:
                            mem['salience'] = float(mem['salience'])
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for semantic retrieval: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_SEMANTIC_RETRIEVAL_FAILED", f"LLM call failed for semantic retrieval: {llm_output_str}", 0.9)
            return None

    async def _update_memory_content_llm(self, existing_memory: Dict[str, Any], new_content_dict: Dict[str, Any], context_snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses LLM to intelligently update a memory's content, merging new information.
        """
        prompt_text = f"""
        You are the Memory Node's LLM assistant. Your task is to intelligently update an existing memory's content with new information. You need to integrate the `new_content` into the `existing_memory`, ensuring logical consistency and preserving important details.

        Existing Memory to Update:
        --- Existing Memory ---
        {json.dumps(existing_memory, indent=2)}

        New Content to Integrate:
        --- New Content ---
        {json.dumps(new_content_dict, indent=2)}

        Robot's Current Cognitive Context (for understanding the update's purpose):
        --- Cognitive Context ---
        {json.dumps(context_snapshot, indent=2)}

        Based on this, generate:
        1.  `updated_content`: object (The merged and updated content. Ensure it's a valid JSON object. If the content is text-based, make sure it has a 'text' key. For complex data, structure as a meaningful JSON object.)
        2.  `processing_notes`: string (Brief notes on how you performed the update or any conflicts encountered.)

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'updated_content': object
        3.  'processing_notes': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "updated_content": {"type": "object"},
                "processing_notes": {"type": "string"}
            },
            "required": ["timestamp", "updated_content", "processing_notes"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.2, max_tokens=700)

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for memory update: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_MEMORY_UPDATE_FAILED", f"LLM call failed for memory update: {llm_output_str}", 0.9)
            return None

    async def _summarize_memories_llm(self, memories_list: List[Dict[str, Any]], context_snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses LLM to summarize a list of memories into a concise overview.
        """
        memories_for_llm = "\n".join([f"- ID: {m.get('id')}, Category: {m.get('category')}, Content: {m.get('content', '')[:200]}..." for m in memories_list])

        prompt_text = f"""
        You are the Memory Node's LLM assistant. Your task is to summarize a collection of `memories_list` into a concise and coherent overview. Consider the robot's `current_cognitive_context` to identify the most salient aspects for the summary.

        Memories to Summarize:
        --- Memories ---
        {memories_for_llm}

        Robot's Current Cognitive Context (for identifying key themes for summary):
        --- Cognitive Context ---
        {json.dumps(context_snapshot, indent=2)}

        Based on these memories and context, generate:
        1.  `summary_text`: string (A concise and coherent summary of the provided memories. Focus on key events, facts, or patterns.)
        2.  `summarization_notes`: string (Brief notes on the key themes or insights gained during summarization.)

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'summary_text': string
        3.  'summarization_notes': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "summary_text": {"type": "string"},
                "summarization_notes": {"type": "string"}
            },
            "required": ["timestamp", "summary_text", "summarization_notes"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.3, max_tokens=400)  # Slightly higher temp for summarization nuance

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for summarization: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_SUMMARIZATION_FAILED", f"LLM call failed for summarization: {llm_output_str}", 0.9)
            return None

    def _compile_llm_context_for_memory_op(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gathers and formats all relevant cognitive state data for the LLM's
        memory processing operations.
        """
        context = {
            "current_time": self._get_current_time(),
            "memory_request_details": request_data,
            "recent_cognitive_inputs": {
                "cognitive_directives_for_self": [d for d in self.recent_cognitive_directives if d.get('target_node') == self.node_name],
                "internal_narratives": list(self.recent_internal_narratives),
                "world_model_states": list(self.recent_world_model_states),
                "social_cognition_states": list(self.recent_social_cognition_states)
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

    # --- Database Operations (SQLite) ---
    def store_memory(self, id: str, timestamp: str, category: str, content: str, keywords: str, source_node: str, salience: float, llm_processing_notes: str, original_context_json: str, sensory_snapshot_json: str):
        """Stores a memory in the SQLite database."""
        try:
            self.cursor.execute('''
                INSERT INTO memories (id, timestamp, category, content, keywords, source_node, salience, llm_processing_notes, original_context_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (id, timestamp, category, content, keywords, source_node, salience, llm_processing_notes, original_context_json, sensory_snapshot_json))
            self.conn.commit()
            _log_info(self.node_name, f"Stored memory (ID: {id}, Category: {category}).")
        except sqlite3.Error as e:
            self._report_error("DB_STORE_ERROR", f"Failed to store memory: {e}", 0.9)
        except Exception as e:
            self._report_error("UNEXPECTED_STORE_ERROR", f"Unexpected error in store_memory: {e}", 0.9)

    def retrieve_memories_from_db(self, query_text: str, category_filter: Optional[str] = None, keywords_filter: Optional[str] = None, num_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieves memories from the SQLite database based on query and filters."""
        # Simple keyword matching for initial retrieval. LLM handles semantic.
        query = "SELECT id, timestamp, category, content, keywords, source_node, salience, llm_processing_notes FROM memories WHERE 1=1"
        params = []

        if query_text:
            query += " AND (content LIKE ? OR keywords LIKE ?)"
            params.extend([f"%{query_text}%", f"%{query_text}%"])
        if category_filter and category_filter != 'general':
            query += " AND category = ?"
            params.append(category_filter)
        if keywords_filter:  # This assumes comma-separated keywords in DB
            keywords_list = [k.strip() for k in keywords_filter.split(',') if k.strip()]
            keyword_conditions = [f"keywords LIKE ?" for _ in keywords_list]
            if keyword_conditions:
                query += " AND (" + " OR ".join(keyword_conditions) + ")"
                params.extend([f"%{k}%" for k in keywords_list])  # Use LIKE for partial matches

        query += " ORDER BY salience DESC, timestamp DESC LIMIT ?"
        params.append(num_results)

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            retrieved = []
            for row in rows:
                mem = {
                    'id': row[0],
                    'timestamp': row[1],
                    'category': row[2],
                    'content': row[3],  # Raw content string
                    'keywords': row[4],
                    'source_node': row[5],
                    'salience': row[6],
                    'llm_processing_notes': row[7]
                }
                # Attempt to parse content back to dict if it was stored as JSON string
                try:
                    mem['content_parsed'] = json.loads(mem['content'])
                except (json.JSONDecodeError, TypeError):
                    mem['content_parsed'] = mem['content']  # Keep as string if not JSON
                retrieved.append(mem)
            _log_info(self.node_name, f"Retrieved {len(retrieved)} memories for query '{query_text}'.")
            return retrieved
        except sqlite3.Error as e:
            self._report_error("DB_RETRIEVE_ERROR", f"Failed to retrieve memories: {e}", 0.9, {'query_text': query_text, 'category': category_filter})
            return []
        except Exception as e:
            self._report_error("UNEXPECTED_RETRIEVE_ERROR", f"Unexpected error in retrieve_memories_from_db: {e}", 0.9)
            return []

    def update_memory(self, memory_id: str, new_content_json: str, new_keywords: str, new_salience: float, llm_processing_notes: str, original_context_json: str, sensory_snapshot_json: str):
        """Updates an existing memory in the SQLite database."""
        try:
            self.cursor.execute('''
                UPDATE memories
                SET content = ?, keywords = ?, salience = ?, llm_processing_notes = ?, original_context_json = ?, sensory_snapshot_json = ?
                WHERE id = ?
            ''', (new_content_json, new_keywords, new_salience, llm_processing_notes, original_context_json, sensory_snapshot_json, memory_id))
            self.conn.commit()
            if self.cursor.rowcount > 0:
                _log_info(self.node_name, f"Updated memory (ID: {memory_id}).")
                return True
            else:
                _log_warn(self.node_name, f"No memory found with ID: {memory_id} for update.")
                return False
        except sqlite3.Error as e:
            self._report_error("DB_UPDATE_ERROR", f"Failed to update memory {memory_id}: {e}", 0.9)
            return False
        except Exception as e:
            self._report_error("UNEXPECTED_UPDATE_ERROR", f"Unexpected error in update_memory: {e}", 0.9)
            return False

    def delete_memory(self, query_text: str, category_filter: Optional[str] = None, keywords_filter: Optional[str] = None):
        """Deletes memories from the SQLite database."""
        query = "DELETE FROM memories WHERE 1=1"
        params = []

        # For simplicity, deletion uses keyword/text matching similar to retrieval
        if query_text:
            query += " AND (content LIKE ? OR keywords LIKE ?)"
            params.extend([f"%{query_text}%", f"%{query_text}%"])
        if category_filter and category_filter != 'general':
            query += " AND category = ?"
            params.append(category_filter)
        if keywords_filter:
            keywords_list = [k.strip() for k in keywords_filter.split(',') if k.strip()]
            keyword_conditions = [f"keywords LIKE ?" for _ in keywords_list]
            if keyword_conditions:
                query += " AND (" + " OR ".join(keyword_conditions) + ")"
                params.extend([f"%{k}%" for k in keywords_list])

        try:
            self.cursor.execute(query, params)
            self.conn.commit()
            _log_info(self.node_name, f"Deleted {self.cursor.rowcount} memories.")
            return self.cursor.rowcount
        except sqlite3.Error as e:
            self._report_error("DB_DELETE_ERROR", f"Failed to delete memories: {e}", 0.9, {'query_text': query_text, 'category': category_filter})
            return 0
        except Exception as e:
            self._report_error("UNEXPECTED_DELETE_ERROR", f"Unexpected error in delete_memory: {e}", 0.9)
            return 0

    # --- Publishing Functions ---
    def publish_memory_response(self, request_id: str, response_code: int, memories_json: str):
        """Publish the response to a memory request."""
        timestamp = str(self._get_current_time())
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_memory_response:
                if hasattr(MemoryResponse, 'data'):  # String fallback
                    response_data = {
                        'timestamp': timestamp,
                        'request_id': request_id,
                        'response_code': response_code,
                        'memories_json': memories_json  # Already JSON string
                    }
                    self.pub_memory_response.publish(String(data=json.dumps(response_data)))
                else:
                    response_msg = MemoryResponse()
                    response_msg.timestamp = timestamp
                    response_msg.request_id = request_id
                    response_msg.response_code = response_code
                    response_msg.memories_json = memories_json
                    self.pub_memory_response.publish(response_msg)
            _log_debug(self.node_name, f"Published Memory Response for request ID: {request_id}. Code: {response_code}.")
        except Exception as e:
            self._report_error("PUBLISH_MEMORY_RESPONSE_ERROR", f"Failed to publish memory response for ID '{request_id}': {e}", 0.7)

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down MemoryNode.")
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
                    self._run_memory_processing_wrapper(None)
                    time.sleep(1)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Memory Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = MemoryNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate a memory request
            request = {'request_type': 'store', 'category': 'episodic', 'content_json': json.dumps({'event': 'test event'}), 'salience': 0.8}
            node.memory_request_callback(request)
            time.sleep(2)
            print("Memory simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
    ```

```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique message IDs
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque

# --- Asyncio Imports (for potential future async operations) ---
import asyncio
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
    InteractionRequest = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    SensoryQualia = ROSMsgFallback
    InteractionRequest = ROSMsgFallback


# --- Import shared utility functions ---
# Assuming 'sentience/scripts/utils.py' exists and contains load_config
try:
    from sentience.scripts.utils import load_config
except ImportError:
    # Fallback implementation
    def load_config(node_name, config_path):
        _log_warn(node_name, f"Mocking load_config for {node_name}. Using hardcoded defaults.")
        return {
            'db_root_path': '/tmp/sentience_db',
            'default_log_level': 'INFO',
            'mock_sensors_node': {
                'publish_interval_sensory_qualia': 1.0,
                'publish_interval_interaction_request': 3.0,
                'simulated_sensory_events': [
                    {"type": "visual", "description": "a person walking by", "salience": 0.6},
                    {"type": "auditory", "description": "a knock on the door", "salience": 0.8},
                    {"type": "tactile", "description": "robot's arm brushes against a surface", "salience": 0.3},
                    {"type": "visual", "description": "a bright red object", "salience": 0.7},
                    {"type": "auditory", "description": "a human voice speaking", "salience": 0.9}
                ],
                'simulated_user_inputs': [
                    {"type": "speech_text", "text": "Hello robot, how are you?", "urgency": 0.5},
                    {"type": "speech_text", "text": "Can you fetch me the book?", "urgency": 0.8, "command_payload": {"action": "fetch", "object": "book"}},
                    {"type": "gesture", "text": "points to a direction", "urgency": 0.4, "gesture_data": {"direction": "forward"}},
                    {"type": "speech_text", "text": "That's wrong, try again!", "urgency": 0.9, "command_payload": {"feedback": "negative"}},
                    {"type": "speech_text", "text": "Thank you, that was helpful.", "urgency": 0.3, "command_payload": {"feedback": "positive"}}
                ],
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate simulations (e.g., positive user interactions)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            }
        }


def _log_info(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [INFO] {msg}", file=sys.stdout)

def _log_warn(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [WARN] {msg}", file=sys.stderr)

def _log_error(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [ERROR] {msg}", file=sys.stderr)

def _log_debug(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [DEBUG] {msg}", file=sys.stdout)


class MockSensorsNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'mock_sensors_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters from 'mock_sensors_node' section of config
        self.mock_sensor_params = self.params.get('mock_sensors_node', {})
        self.sensory_qualia_interval = self.mock_sensor_params.get('publish_interval_sensory_qualia', 1.0)
        self.interaction_request_interval = self.mock_sensor_params.get('publish_interval_interaction_request', 3.0)
        self.simulated_sensory_events = self.mock_sensor_params.get('simulated_sensory_events', [])
        self.simulated_user_inputs = self.mock_sensor_params.get('simulated_user_inputs', [])
        self.ethical_compassion_bias = self.mock_sensor_params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (for dynamic simulation)
        self.sensory_sources = self.mock_sensor_params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # Internal state
        self.execution_history = deque(maxlen=50)  # Track simulated events for logging
        self.pending_simulations: Deque[Dict[str, Any]] = deque(maxlen=10)  # Queue for dynamic simulations

        # Initialize SQLite database for mock sensor logs
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "mock_sensors_log.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS mock_sensors_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                event_type TEXT,
                data_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Mock Sensors Node online, simulating inputs with compassionate bias toward positive interactions.")

        # --- Simulated ROS Compatibility: Conditional Setup ---
        self.pub_sensory_qualia = None
        self.pub_interaction_request = None
        self.pub_error_report = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_sensory_qualia = rospy.Publisher('/sensory_qualia', SensoryQualia, queue_size=10)
            self.pub_interaction_request = rospy.Publisher('/interaction_request', InteractionRequest, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)

            # Timers for periodic publishing
            rospy.Timer(rospy.Duration(self.sensory_qualia_interval), self.publish_sensory_qualia)
            rospy.Timer(rospy.Duration(self.interaction_request_interval), self.publish_interaction_request)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing simulations compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on mock data
            if sensor_type == 'vision':
                self.pending_simulations.append({'type': 'sensory_qualia', 'data': {'type': 'visual', 'description': processed.get('description', 'scene'), 'salience': random.uniform(0.4, 0.8)}})
            elif sensor_type == 'sound':
                self.pending_simulations.append({'type': 'interaction_request', 'data': {'text': processed.get('transcription', 'sound input'), 'urgency': random.uniform(0.3, 0.6)}})
            elif sensor_type == 'instructions':
                self.pending_simulations.append({'type': 'sensory_qualia', 'data': {'type': 'tactile', 'description': processed.get('instruction', 'user command'), 'salience': random.uniform(0.5, 0.9)}})
            # Compassionate bias: If distress in sound, bias toward positive simulated interactions
            if 'distress' in str(processed):
                self.ethical_compassion_bias = min(1.0, self.ethical_compassion_bias + 0.1)
                # Adjust next simulation to be more supportive
                if self.pending_simulations:
                    self.pending_simulations[-1]['data']['compassionate_note'] = "Prioritizing empathetic response."
            _log_debug(self.node_name, f"{sensor_type} input updated simulation context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self.publish_sensory_qualia(None)
            time.sleep(self.sensory_qualia_interval)
            self.publish_interaction_request(None)
            time.sleep(self.interaction_request_interval)

    # --- Core Simulation Logic ---
    def publish_sensory_qualia(self, event: Any = None):
        """Publish a simulated SensoryQualia message with compassionate bias toward positive events."""
        if not self.simulated_sensory_events:
            _log_debug(self.node_name, "No simulated sensory events configured. Skipping SensoryQualia.")
            return

        # Pick a random sensory event from the predefined list
        simulated_event = random.choice(self.simulated_sensory_events)
        
        # Compassionate bias: Occasionally boost positive events
        if random.random() < self.ethical_compassion_bias:
            simulated_event['description'] += " (with positive emotional nuance)"
            simulated_event['salience'] = min(1.0, simulated_event.get('salience', 0.5) + 0.1)
        
        timestamp = str(self._get_current_time())
        qualia_id = str(uuid.uuid4())  # Unique ID for each qualia event
        qualia_type = simulated_event.get('type', 'generic')
        modality = 'visual' if 'visual' in qualia_type.lower() else ('auditory' if 'auditory' in qualia_type.lower() else 'tactile')
        description_summary = simulated_event.get('description', 'simulated event')
        salience_score = simulated_event.get('salience', 0.5)
        raw_data_hash = str(random.getrandbits(128))  # Simulate a hash for raw data

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
            else:
                # Dynamic mode: Log or store
                event_entry = {
                    'id': qualia_id,
                    'timestamp': timestamp,
                    'type': 'sensory_qualia',
                    'data': {
                        'timestamp': timestamp,
                        'qualia_id': qualia_id,
                        'qualia_type': qualia_type,
                        'modality': modality,
                        'description_summary': description_summary,
                        'salience_score': salience_score,
                        'raw_data_hash': raw_data_hash
                    },
                    'sensory_snapshot': self.sensory_data
                }
                self._log_mock_event(event_entry)
            _log_debug(self.node_name, f"Published Sensory Qualia: {description_summary} ({modality}).")
        except Exception as e:
            self._report_error("PUBLISH_SENSORY_QUALIA_ERROR", f"Failed to publish sensory qualia: {e}", 0.7)

    def publish_interaction_request(self, event: Any = None):
        """Publish a simulated InteractionRequest message with compassionate bias toward supportive interactions."""
        if not self.simulated_user_inputs:
            _log_debug(self.node_name, "No simulated user inputs configured. Skipping InteractionRequest.")
            return

        # Pick a random user input from the predefined list
        simulated_input = random.choice(self.simulated_user_inputs)
        
        # Compassionate bias: Occasionally bias toward positive/empathetic interactions
        if random.random() < self.ethical_compassion_bias:
            simulated_input['text'] += " (with appreciative tone)"
            simulated_input['urgency'] = min(1.0, simulated_input.get('urgency', 0.5) - 0.1)  # Lower urgency for positive

        timestamp = str(self._get_current_time())
        request_id = str(uuid.uuid4())  # Unique ID for each request
        request_type = simulated_input.get('type', 'speech_text')  # e.g., 'speech_text', 'gesture', 'command'
        user_id = simulated_input.get('user_id', 'simulated_user_1')
        command_payload = json.dumps(simulated_input.get('command_payload', {}))  # Ensure it's a JSON string
        urgency_score = simulated_input.get('urgency', 0.5)
        speech_text = simulated_input.get('text', '')
        gesture_data_json = json.dumps(simulated_input.get('gesture_data', {}))  # Ensure it's a JSON string

        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_interaction_request:
                if hasattr(InteractionRequest, 'data'):  # String fallback
                    request_data = {
                        'timestamp': timestamp,
                        'request_id': request_id,
                        'request_type': request_type,
                        'user_id': user_id,
                        'command_payload': command_payload,
                        'urgency_score': urgency_score,
                        'speech_text': speech_text,
                        'gesture_data_json': gesture_data_json
                    }
                    self.pub_interaction_request.publish(String(data=json.dumps(request_data)))
                else:
                    request_msg = InteractionRequest()
                    request_msg.timestamp = timestamp
                    request_msg.request_id = request_id
                    request_msg.request_type = request_type
                    request_msg.user_id = user_id
                    request_msg.command_payload = command_payload
                    request_msg.urgency_score = urgency_score
                    request_msg.speech_text = speech_text
                    request_msg.gesture_data_json = gesture_data_json
                    self.pub_interaction_request.publish(request_msg)
            else:
                # Dynamic mode: Log or store
                request_entry = {
                    'id': request_id,
                    'timestamp': timestamp,
                    'type': 'interaction_request',
                    'data': {
                        'timestamp': timestamp,
                        'request_id': request_id,
                        'request_type': request_type,
                        'user_id': user_id,
                        'command_payload': json.loads(command_payload) if command_payload else {},
                        'urgency_score': urgency_score,
                        'speech_text': speech_text,
                        'gesture_data_json': json.loads(gesture_data_json) if gesture_data_json else {}
                    },
                    'sensory_snapshot': self.sensory_data
                }
                self._log_mock_event(request_entry)
            _log_debug(self.node_name, f"Published Interaction Request: '{speech_text}' (Type: {request_type}).")
        except Exception as e:
            self._report_error("PUBLISH_INTERACTION_REQUEST_ERROR", f"Failed to publish interaction request: {e}", 0.7)

    def _log_mock_event(self, event_entry: Dict[str, Any]):
        """Log simulated event to DB for persistence."""
        try:
            self.cursor.execute('''
                INSERT INTO mock_sensors_log (id, timestamp, event_type, data_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                event_entry['id'], event_entry['timestamp'], event_entry['type'],
                json.dumps(event_entry['data']), json.dumps(self.sensory_data)
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log mock event: {e}")

    def _report_error(self, error_type: str, description: str, severity: float = 0.5, context: Optional[Dict] = None):
        """Report an error with compassionate note."""
        timestamp = str(self._get_current_time())
        compassionate_note = f"Compassionate reflection: Emphasize learning from this to improve future simulations. Bias: {self.ethical_compassion_bias}." if severity > 0.5 else ""
        error_msg_data = {
            'timestamp': timestamp,
            'source_node': self.node_name,
            'error_type': error_type,
            'description': description,
            'severity': severity,
            'compassionate_note': compassionate_note,
            'context': context or {}
        }
        if ROS_AVAILABLE and self.ros_enabled and self.pub_error_report:
            try:
                self.pub_error_report.publish(String(data=json.dumps(error_msg_data)))
                rospy.logerr(f"{self.node_name}: REPORTED ERROR: {error_type} - {description}")
            except Exception as e:
                _log_error(self.node_name, f"Failed to publish error report: {e}")
        else:
            _log_error(self.node_name, f"REPORTED ERROR: {error_type} - {description} (Severity: {severity})")
        # Log to DB
        self._log_mock_event({'type': 'error', 'data': error_msg_data})

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down MockSensorsNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.signal_shutdown("Node shutdown requested.")

    def run(self):
        """Run the node with simulated or actual ROS publishing."""
        if ROS_AVAILABLE and self.ros_enabled:
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                _log_info(self.node_name, "Interrupted by ROS shutdown.")
        else:
            try:
                while True:
                    time.sleep(1)  # Idle in dynamic mode; simulations run in thread
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested via KeyboardInterrupt.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Mock Sensors Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = MockSensorsNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate a few events
            time.sleep(5)
            print("Mock sensors simulation complete. Generated events logged to DB.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
