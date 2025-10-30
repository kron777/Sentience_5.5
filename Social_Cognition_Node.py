```python
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
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
    InteractionRequest = ROSMsgFallback
    InteractionResponse = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    InteractionRequest = ROSMsgFallback
    InteractionResponse = ROSMsgFallback


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
            'social_cognition_node': {
                'ethical_compassion_bias': 0.2,  # Bias toward compassionate social interpretations (e.g., empathetic responses)
                'sensory_inputs': {  # Dynamic placeholders
                    'vision': {'source': 'camera_feed', 'format': 'image_array'},
                    'sound': {'source': 'microphone', 'format': 'audio_waveform'},
                    'instructions': {'source': 'command_line', 'format': 'text'}
                }
            },
            'llm_params': {
                'model_name': "phi-2",
                'base_url': "http://localhost:8000/v1/chat/completions",
                'timeout_seconds': 10.0
            }
        }.get(node_name, {})  # Return node-specific or empty dict

# Fallback for AsyncPhi2Client if not available
class AsyncPhi2ClientFallback:
    def __init__(self, endpoint="http://localhost:8000/generate", timeout=10.0):
        self.endpoint = endpoint
        self.timeout = timeout

    async def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 128) -> str:
        # Simple fallback - in real use, implement or import
        return f"[Fallback Response] Echoing query: {prompt[:50]}..."


def _log_info(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [INFO] {msg}", file=sys.stdout)

def _log_warn(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [WARN] {msg}", file=sys.stderr)

def _log_error(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [ERROR] {msg}", file=sys.stderr)

def _log_debug(node_name: str, msg: str):
    print(f"[{datetime.now().isoformat()}] {node_name}: [DEBUG] {msg}", file=sys.stdout)


class SocialCognitionNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'social_cognition_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "social_cognition_log.db")
        self.ethical_compassion_bias = self.params.get('ethical_compassion_bias', 0.2)

        # Sensory placeholders (e.g., vision/sound influencing social cognition compassionately)
        self.sensory_sources = self.params.get('sensory_inputs', {})
        self.vision_callback = self._create_sensory_placeholder('vision')
        self.sound_callback = self._create_sensory_placeholder('sound')
        self.instructions_callback = self._create_sensory_placeholder('instructions')

        # LLM client
        self.phi2 = AsyncPhi2Client()  # Assume it's available; fallback if not

        # Internal state
        self.pending_queries: Deque[Dict[str, Any]] = deque(maxlen=5)  # Queue for queries
        self.interaction_history: Deque[Dict[str, Any]] = deque(maxlen=50)  # History for patterns

        # Initialize SQLite database for social cognition logs
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS social_cognition_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                prompt TEXT,
                response TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.conn.commit()

        # Async setup
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._async_thread.start()

        # Simulated ROS Compatibility: Conditional Setup
        self.pub_interaction_response = None
        self.sub_interaction_request = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_interaction_response = rospy.Publisher('/sentience/interaction_response', InteractionResponse, queue_size=10)
            self.sub_interaction_request = rospy.Subscriber('/sentience/interaction_request', InteractionRequest, self.on_prompt)
            rospy.Timer(rospy.Duration(1.0), self._process_pending_queries)  # Periodic processing
        else:
            # Dynamic mode: Start polling thread for simulated inputs
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        _log_info(self.node_name, "Social Cognition Node initialized with compassionate social interpretation.")

    def _create_sensory_placeholder(self, sensor_type: str):
        """Dynamic placeholder for sensory inputs influencing social cognition compassionately."""
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensory influence on social queries (e.g., add empathetic prompt)
            if sensor_type == 'vision':
                self.pending_queries.append({'type': 'query', 'data': {'prompt': f"User seen: {processed.get('description', 'person')}. Respond compassionately."}})
            elif sensor_type == 'sound':
                self.pending_queries.append({'type': 'query', 'data': {'prompt': processed.get('transcription', 'audio input')}})
            elif sensor_type == 'instructions':
                self.pending_queries.append({'type': 'query', 'data': {'prompt': processed.get('instruction', 'user command')}})
            # Compassionate bias: If distress in sound, add compassionate tone
            if 'distress' in str(processed):
                if self.pending_queries:
                    self.pending_queries[-1]['data']['prompt'] += f" (Respond with compassion, bias: {self.ethical_compassion_bias})."
            _log_debug(self.node_name, f"{sensor_type} input updated social cognition context at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._simulate_query()
            self._process_pending_queries()
            time.sleep(1.0)

    def _simulate_query(self):
        """Simulate a social query in non-ROS mode."""
        query_data = {'prompt': random.choice(["Hello, how are you?", "Can you help me?", "I'm feeling sad."])}
        self.pending_queries.append({'type': 'query', 'data': query_data})
        _log_debug(self.node_name, f"Simulated query: {query_data['prompt']}")

    # --- Core Social Cognition Logic ---
    def on_prompt(self, msg: Any):
        """Handle incoming interaction requests."""
        fields_map = {'data': ('', 'prompt_data')}
        data = parse_message_data(msg, fields_map, self.node_name)
        prompt = data.get('prompt_data', '')
        _log_info(self.node_name, f"Received prompt: {prompt}")
        try:
            response = asyncio.run_coroutine_threadsafe(self.handle_phi2(prompt), self._async_loop).result()
            self.publish_response(response)
        except Exception as e:
            error_msg = f"Exception in handle_phi2: {e}"
            _log_error(self.node_name, error_msg)
            self._log_error(self.node_name, error_msg)

    async def handle_phi2(self, prompt: str) -> str:
        """Query the LLM with compassionate bias in the prompt."""
        # Compassionate bias: Add compassionate tone to prompt
        compassionate_prompt = f"{prompt}\n\nRespond with compassion and empathy, bias: {self.ethical_compassion_bias}."
        return await self.phi2.query(compassionate_prompt)

    def _process_pending_queries(self):
        """Process pending queries in dynamic or timer mode."""
        while self.pending_queries:
            update_data = self.pending_queries.popleft()
            if update_data.get('type') == 'query':
                prompt = update_data['data']['prompt']
                response = asyncio.run_coroutine_threadsafe(self.handle_phi2(prompt), self._async_loop).result()
                self.publish_response(response)
            self.interaction_history.append(update_data)

    def publish_response(self, response: str):
        """Publish interaction response (ROS or log)."""
        try:
            if ROS_AVAILABLE and self.ros_enabled and self.pub_interaction_response:
                if hasattr(InteractionResponse, 'data'):
                    self.pub_interaction_response.publish(String(data=response))
                else:
                    response_msg = InteractionResponse(data=response)
                    self.pub_interaction_response.publish(response_msg)
            else:
                # Dynamic mode: Log
                _log_info(self.node_name, f"Published response: {response}")
        except Exception as e:
            _log_error(self.node_name, f"Failed to publish response: {e}")

    def _log_interaction(self, prompt: str, response: str, sensory_snapshot: str):
        """Log interaction to DB."""
        try:
            self.cursor.execute('''
                INSERT INTO social_cognition_log (id, timestamp, prompt, response, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), str(self._get_current_time()), prompt, response, sensory_snapshot
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            _log_error(self.node_name, f"Failed to log interaction: {e}")

    def _get_current_time(self) -> float:
        return rospy.get_time() if ROS_AVAILABLE and self.ros_enabled else time.time()

    def shutdown(self):
        """Graceful shutdown."""
        _log_info(self.node_name, "Shutting down SocialCognitionNode.")
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        if hasattr(self, '_async_loop') and self._async_thread.is_alive():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            self._async_thread.join(timeout=5.0)
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
                    self._simulate_query()
                    self._process_pending_queries()
                    time.sleep(1.0)
            except KeyboardInterrupt:
                _log_info(self.node_name, "Shutdown requested.")

        self.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentience Social Cognition Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = SocialCognitionNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        # Example dynamic usage
        if not args.ros_enabled:
            # Simulate a query
            node._simulate_query()
            time.sleep(2)
            print("Social cognition simulation complete.")
        node.run()
    except KeyboardInterrupt:
        _log_info('main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
```
