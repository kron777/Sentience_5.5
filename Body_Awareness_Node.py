```python:disable-run
#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random  # Used sparingly for unique IDs or minor variations
import uuid  # For generating unique body awareness event IDs
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque

# --- Asyncio Imports for LLM calls ---
import asyncio
import aiohttp
import threading  # To run asyncio loop in a separate thread
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
    BodyAwarenessState = ROSMsgFallback
    JointState = ROSMsgFallback
    ForceTorque = ROSMsgFallback
    TactileSensor = ROSMsgFallback
    RobotHealth = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    InternalNarrative = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    BodyAwarenessState = ROSMsgFallback
    JointState = ROSMsgFallback
    ForceTorque = ROSMsgFallback
    TactileSensor = ROSMsgFallback
    RobotHealth = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
    InternalNarrative = ROSMsgFallback


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
            'body_awareness_node': {
                'body_awareness_analysis_interval': 0.2,
                'llm_analysis_threshold_salience': 0.5,
                'recent_context_window_s': 5.0,
                'ethical_compassion_threshold': 0.3,  # Bias toward self-care and mindful movement
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


class BodyAwarenessNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'body_awareness_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "body_awareness_log.db")
        self.analysis_interval = self.params.get('body_awareness_analysis_interval', 0.2)
        self.llm_analysis_threshold_salience = self.params.get('llm_analysis_threshold_salience', 0.5)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 5.0)
        self.ethical_compassion_threshold = self.params.get('ethical_compassion_threshold', 0.3)

        # Sensory placeholders
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

        _log_info(self.node_name, "Robot's body awareness system online, nurturing mindful embodiment.")

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
            CREATE TABLE IF NOT EXISTS body_awareness_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                body_state TEXT,
                posture_description TEXT,
                anomaly_detected BOOLEAN,
                anomaly_severity REAL,
                llm_reasoning TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_awareness_timestamp ON body_awareness_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.current_body_awareness_state = {
            'timestamp': str(time.time()),
            'body_state': 'normal',
            'posture_description': 'stable',
            'anomaly_detected': False,
            'anomaly_severity': 0.0
        }

        # History deques
        self.recent_joint_states: Deque[Dict[str, Any]] = deque(maxlen=10)
        self.recent_force_torques: Deque[Dict[str, Any]] = deque(maxlen=10)
        self.recent_tactile_sensor_data: Deque[Dict[str, Any]] = deque(maxlen=10)
        self.recent_robot_health_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_memory_responses: Deque[Dict[str, Any]] = deque(maxlen=3)
        self.recent_internal_narratives: Deque[Dict[str, Any]] = deque(maxlen=5)

        self.cumulative_body_salience = 0.0

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_body_awareness_state = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_body_awareness_state = rospy.Publisher('/body_awareness_state', BodyAwarenessState, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
            rospy.Subscriber('/force_torque_sensors', ForceTorque, self.force_torque_callback)
            rospy.Subscriber('/tactile_sensors', TactileSensor, self.tactile_sensor_callback)
            rospy.Subscriber('/robot_health', RobotHealth, self.robot_health_callback)
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            rospy.Subscriber('/memory_response', MemoryResponse, self.memory_response_callback)
            rospy.Subscriber('/internal_narrative', InternalNarrative, self.internal_narrative_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.analysis_interval), self._run_body_analysis_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._shutdown_flag = threading.Event()
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

        # Initial publish
        self.publish_body_awareness_state(None)

    def _create_sensory_placeholder(self, sensor_type: str):
        def placeholder_callback(data: Any):
            timestamp = time.time()
            processed_data = data if isinstance(data, dict) else {'raw': str(data)}
            # Simulate sensor data influencing body awareness
            if sensor_type == 'vision':
                self.recent_joint_states.append({'timestamp': timestamp, 'positions': [random.uniform(-1,1) for _ in range(6)], 'salience_score': random.uniform(0.1, 0.4)})
            elif sensor_type == 'sound':
                self.recent_tactile_sensor_data.append({'timestamp': timestamp, 'pressure_sum': random.uniform(0, 10), 'salience_score': random.uniform(0.2, 0.5)})
            elif sensor_type == 'instructions':
                self.recent_cognitive_directives.append({'timestamp': timestamp, 'directive_type': 'BodyCheck', 'urgency': random.uniform(0.3, 0.7)})
            self._update_cumulative_salience(0.15)  # Sensory adds to body salience
            _log_debug(self.node_name, f"{sensor_type} input updated at {timestamp}")
        return placeholder_callback

    def _dynamic_execution_loop(self):
        """Dynamic polling loop when ROS is disabled."""
        while not self._shutdown_flag.is_set():
            self._run_body_analysis_wrapper(None)
            time.sleep(self.analysis_interval)

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

    def _run_body_analysis_wrapper(self, event: Any = None):
        """Wrapper to run the async body awareness analysis from a ROS timer."""
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM body analysis task already active. Skipping new cycle.")
            return
        
        # Schedule the async task
        self.active_llm_task = asyncio.run_coroutine_threadsafe(
            self.analyze_body_state_async(event), self._async_loop
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
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.3, max_tokens: int = 300) -> str:
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
            "temperature": temperature,  # Low temperature for factual/reasoning tasks
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
        self.cumulative_body_salience += score
        self.cumulative_body_salience = min(1.0, self.cumulative_body_salience)

    # --- Pruning old history ---
    def _prune_history(self):
        """Removes old entries from history deques based on recent_context_window_s."""
        current_time = self._get_current_time()
        for history_deque in [
            self.recent_joint_states, self.recent_force_torques, self.recent_tactile_sensor_data,
            self.recent_robot_health_states, self.recent_cognitive_directives,
            self.recent_memory_responses, self.recent_internal_narratives
        ]:
            while history_deque and (current_time - float(history_deque[0].get('timestamp', 0.0))) > self.recent_context_window_s:
                history_deque.popleft()

    # --- Callbacks (generic, ROS or direct) ---
    def joint_state_callback(self, msg: Any):
        # NOTE: Standard sensor_msgs/JointState doesn't have a 'timestamp' attribute directly on the message,
        # but in ROS it has a header with stamp. Assuming message converted to dictionary or has direct attributes.
        # If using actual sensor_msgs.msg.JointState, adjust 'timestamp' access (e.g., msg.header.stamp.to_sec())
        fields_map = {
            'header.stamp': (str(self._get_current_time()), 'timestamp'),  # Accessing header.stamp, assuming parsed
            'name': ([], 'joint_names'),  # List of joint names
            'position': ([], 'positions'),  # List of joint positions
            'velocity': ([], 'velocities'),  # List of joint velocities
            'effort': ([], 'efforts')  # List of joint efforts
        }
        # Special handling for JointState if it's not a String message.
        # If it's a real JointState message, its structure needs direct attribute access.
        # For simplicity, if it's String fallback, it's JSON.
        if hasattr(msg, 'data') and isinstance(msg.data, str):
            data = parse_message_data(msg, fields_map, self.node_name)
        else:  # Assume actual JointState message type
            # Directly access attributes of JointState message
            data = {
                'timestamp': str(msg.header.stamp.to_sec()) if hasattr(msg, 'header') else str(self._get_current_time()),
                'joint_names': msg.name if hasattr(msg, 'name') else [],
                'positions': msg.position if hasattr(msg, 'position') else [],
                'velocities': msg.velocity if hasattr(msg, 'velocity') else [],
                'efforts': msg.effort if hasattr(msg, 'effort') else []
            }
        
        # Calculate a simple 'salience' for joint state changes (e.g., high velocity/effort indicates activity)
        # This is a heuristic; real salience might need more sophisticated models.
        salience = 0.0
        if 'velocities' in data:
            salience += sum([abs(v) for v in data['velocities']]) * 0.05
        if 'efforts' in data:
            salience += sum([abs(e) for e in data['efforts']]) * 0.02
        data['salience_score'] = min(1.0, salience)  # Clamp salience

        self.recent_joint_states.append(data)
        self._update_cumulative_salience(data['salience_score'] * 0.3)
        _log_debug(self.node_name, f"Received Joint State (Activity Salience: {data['salience_score']:.2f}).")

    def force_torque_callback(self, msg: Any):
        # Assuming ForceTorque message has 'header.stamp', 'wrench.force.x', 'wrench.torque.x' etc.
        fields_map = {
            'header.stamp': (str(self._get_current_time()), 'timestamp'),
            'wrench.force.x': (0.0, 'force_x'), 'wrench.force.y': (0.0, 'force_y'), 'wrench.force.z': (0.0, 'force_z'),
            'wrench.torque.x': (0.0, 'torque_x'), 'wrench.torque.y': (0.0, 'torque_y'), 'wrench.torque.z': (0.0, 'torque_z')
        }
        if hasattr(msg, 'data') and isinstance(msg.data, str):
            data = parse_message_data(msg, fields_map, self.node_name)
        else:  # Assume actual ForceTorque message type (e.g., geometry_msgs/WrenchStamped)
            data = {
                'timestamp': str(msg.header.stamp.to_sec()) if hasattr(msg, 'header') else str(self._get_current_time()),
                'force_x': msg.wrench.force.x if hasattr(msg, 'wrench') else 0.0,
                'force_y': msg.wrench.force.y if hasattr(msg, 'wrench') else 0.0,
                'force_z': msg.wrench.force.z if hasattr(msg, 'wrench') else 0.0,
                'torque_x': msg.wrench.torque.x if hasattr(msg, 'wrench') else 0.0,
                'torque_y': msg.wrench.torque.y if hasattr(msg, 'wrench') else 0.0,
                'torque_z': msg.wrench.torque.z if hasattr(msg, 'wrench') else 0.0
            }

        salience = 0.0
        # High force/torque indicates interaction or impact
        if 'force_x' in data:
            salience += abs(data['force_x']) * 0.1
        if 'force_y' in data:
            salience += abs(data['force_y']) * 0.1
        if 'force_z' in data:
            salience += abs(data['force_z']) * 0.1
        data['salience_score'] = min(1.0, salience * 0.5)  # Clamp and scale

        self.recent_force_torques.append(data)
        self._update_cumulative_salience(data['salience_score'] * 0.5)  # Force/torque implies significant interaction
        _log_debug(self.node_name, f"Received Force/Torque data (Salience: {data['salience_score']:.2f}).")

    def tactile_sensor_callback(self, msg: Any):
        # Assuming TactileSensor message has 'timestamp', 'contact_points_json', 'pressure_sum', 'num_contacts'
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'contact_points_json': ('[]', 'contact_points_json'),
            'pressure_sum': (0.0, 'pressure_sum'),
            'num_contacts': (0, 'num_contacts')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        if isinstance(data.get('contact_points_json'), str):
            try:
                data['contact_points'] = json.loads(data['contact_points_json'])
            except json.JSONDecodeError:
                data['contact_points'] = []

        salience = data.get('pressure_sum', 0.0) * 0.3 + data.get('num_contacts', 0) * 0.1
        data['salience_score'] = min(1.0, salience)  # Clamp salience

        self.recent_tactile_sensor_data.append(data)
        self._update_cumulative_salience(data['salience_score'] * 0.4)  # Tactile input implies physical interaction
        _log_debug(self.node_name, f"Received Tactile Sensor data (Contacts: {data['num_contacts']}, Pressure: {data['pressure_sum']:.2f}).")

    def robot_health_callback(self, msg: Any):
        # Assuming RobotHealth message has 'timestamp', 'overall_status', 'battery_level', 'motor_temps_json', 'error_flags_json'
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'),
            'overall_status': ('normal', 'overall_status'),
            'battery_level': (100.0, 'battery_level'),
            'motor_temps_json': ('{}', 'motor_temps_json'),
            'error_flags_json': ('{}', 'error_flags_json')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        if isinstance(data.get('motor_temps_json'), str):
            try:
                data['motor_temps'] = json.loads(data['motor_temps_json'])
            except json.JSONDecodeError:
                data['motor_temps'] = {}
        if isinstance(data.get('error_flags_json'), str):
            try:
                data['error_flags'] = json.loads(data['error_flags_json'])
            except json.JSONDecodeError:
                data['error_flags'] = {}

        salience = 0.0
        if data.get('overall_status') == 'critical':
            salience = 1.0
        elif data.get('overall_status') == 'warning':
            salience = 0.7
        if data.get('battery_level', 100.0) < 20.0:
            salience = max(salience, 0.6)
        if data.get('error_flags'):  # If any error flags are true
            if any(data['error_flags'].values()):
                salience = max(salience, 0.8)
        data['salience_score'] = min(1.0, salience)

        self.recent_robot_health_states.append(data)
        self._update_cumulative_salience(data['salience_score'] * 0.9)  # Health issues are high priority
        _log_debug(self.node_name, f"Received Robot Health State (Status: {data['overall_status']}, Salience: {data['salience_score']:.2f}).")

    def cognitive_directive_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'directive_type': ('', 'directive_type'),
            'target_node': ('', 'target_node'), 'command_payload': ('{}', 'command_payload'),
            'urgency': (0.0, 'urgency'), 'reason': ('', 'reason')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        if data.get('target_node') == self.node_name:
            self.recent_cognitive_directives.append(data)  # Add directives for self to context
            # Directives for body integrity checks or specific posture adjustments are highly salient
            if data.get('directive_type') in ['CheckBodyIntegrity', 'AdjustPosture']:
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
                data['memories']
```
