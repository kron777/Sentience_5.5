#!/usr/bin/env python3
import sqlite3
import os
import json
import time
import random
import uuid  # For unique action IDs
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List

# --- Asyncio Imports for LLM calls ---
import asyncio
import aiohttp
import threading
from collections import deque

# --- Optional ROS Integration (for compatibility) ---
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
    ActionExecutionResult = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    WorldModelState = ROSMsgFallback
    BodyAwarenessState = ROSMsgFallback
    PerformanceReport = ROSMsgFallback
    EthicalDecision = ROSMsgFallback
    MemoryResponse = ROSMsgFallback
except ImportError:
    class ROSMsgFallback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    ActionExecutionResult = ROSMsgFallback
    CognitiveDirective = ROSMsgFallback
    WorldModelState = ROSMsgFallback
    BodyAwarenessState = ROSMsgFallback
    PerformanceReport = ROSMsgFallback
    EthicalDecision = ROSMsgFallback
    MemoryResponse = ROSMsgFallback


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
            'action_execution_node': {
                'execution_interval': 0.1,
                'llm_safety_check_threshold_salience': 0.7,
                'recent_context_window_s': 5.0,
                'action_sim_success_rate': 0.8,
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


class ActionExecutionNode:
    def __init__(self, config_file_path: Optional[str] = None, ros_enabled: bool = False):
        self.node_name = 'action_execution_node'
        self.ros_enabled = ros_enabled or os.getenv('ROS_ENABLED', 'false').lower() == 'true'

        # --- Load parameters from centralized config ---
        if config_file_path is None:
            config_file_path = os.getenv('SENTIENCE_CONFIG_PATH', None)
        full_config = load_config("global", config_file_path)
        self.params = load_config(self.node_name, config_file_path)

        if not self.params or not full_config:
            raise ValueError(f"{self.node_name}: Failed to load configuration from '{config_file_path}'.")

        # Override with explicit ros_enabled
        self.ros_enabled = ros_enabled

        # Assign parameters
        self.db_path = os.path.join(full_config.get('db_root_path', '/tmp/sentience_db'), "action_log.db")
        self.execution_interval = self.params.get('execution_interval', 0.1)
        self.llm_safety_check_threshold_salience = self.params.get('llm_safety_check_threshold_salience', 0.7)
        self.recent_context_window_s = self.params.get('recent_context_window_s', 5.0)
        self.action_sim_success_rate = self.params.get('action_sim_success_rate', 0.8)
        self.sensory_sources = self.params.get('sensory_inputs', {})

        # LLM Parameters
        self.llm_model_name = full_config.get('llm_params', {}).get('model_name', "phi-2")
        self.llm_base_url = full_config.get('llm_params', {}).get('base_url', "http://localhost:8000/v1/chat/completions")
        self.llm_timeout = full_config.get('llm_params', {}).get('timeout_seconds', 15.0)

        # Log level setup
        log_level = full_config.get('default_log_level', 'INFO').upper()

        _log_info(self.node_name, "Robot's action execution system online, ready to act with mindful compassion.")

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
            CREATE TABLE IF NOT EXISTS action_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                action_id TEXT,
                command_payload_json TEXT,
                success BOOLEAN,
                outcome_summary TEXT,
                predicted_outcome_match REAL,
                safety_clearance BOOLEAN,
                llm_safety_reasoning TEXT,
                context_snapshot_json TEXT,
                sensory_snapshot_json TEXT
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_action_timestamp ON action_log (timestamp)')
        self.conn.commit()

        # --- Internal State ---
        self.action_queue: Deque[Dict[str, Any]] = deque()

        # History deques
        self.recent_cognitive_directives: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_world_model_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_body_awareness_states: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_performance_reports: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_ethical_decisions: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.recent_memory_responses: Deque[Dict[str, Any]] = deque(maxlen=3)

        self.cumulative_safety_salience = 0.0

        # --- ROS Compatibility: Conditional Setup ---
        self.pub_action_execution_result = None
        self.pub_error_report = None
        self.pub_cognitive_directive = None
        if ROS_AVAILABLE and self.ros_enabled:
            rospy.init_node(self.node_name, anonymous=False)
            self.pub_action_execution_result = rospy.Publisher('/action_execution_result', ActionExecutionResult, queue_size=10)
            self.pub_error_report = rospy.Publisher('/error_monitor/report', String, queue_size=10)
            self.pub_cognitive_directive = rospy.Publisher('/cognitive_directives', CognitiveDirective, queue_size=10)

            # Subscribers
            rospy.Subscriber('/cognitive_directives', CognitiveDirective, self.cognitive_directive_callback)
            rospy.Subscriber('/world_model_state', String, self.world_model_state_callback)
            rospy.Subscriber('/body_awareness_state', String, self.body_awareness_state_callback)
            rospy.Subscriber('/performance_report', PerformanceReport, self.performance_report_callback)
            rospy.Subscriber('/ethical_decision', String, self.ethical_decision_callback)
            rospy.Subscriber('/memory_response', String, self.memory_response_callback)
            # Sensory
            rospy.Subscriber('/vision_data', String, self.vision_callback)
            rospy.Subscriber('/audio_input', String, self.sound_callback)
            rospy.Subscriber('/user_instructions', String, self.instructions_callback)

            rospy.Timer(rospy.Duration(self.execution_interval), self._run_action_execution_wrapper)
        else:
            # Dynamic mode: Start polling thread
            self._execution_thread = threading.Thread(target=self._dynamic_execution_loop, daemon=True)
            self._execution_thread.start()

    def _create_sensory_callback(self, sensor_type: str):
        def callback(data: Any):
            timestamp = time.time()
            processed_data = data if isinstance(data, dict) else {'raw': data}
            self.sensory_data[sensor_type] = {'data': processed_data, 'timestamp': timestamp}
            self._update_cumulative_salience(0.1)  # Sensory adds salience
            _log_debug(self.node_name, f"{sensor_type} input updated at {timestamp}")
        return callback

    def _dynamic_execution_loop(self):
        while not self._shutdown_flag:
            self._run_action_execution_wrapper(None)
            time.sleep(self.execution_interval)

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

    def _run_action_execution_wrapper(self, event):
        if self.active_llm_task and not self.active_llm_task.done():
            _log_debug(self.node_name, "LLM safety check task already active. Skipping new cycle.")
            return

        if self.action_queue:
            action_to_execute = self.action_queue.popleft()
            self.active_llm_task = asyncio.run_coroutine_threadsafe(
                self.execute_action_async(action_to_execute, event), self._async_loop
            )
        else:
            _log_debug(self.node_name, "No actions in queue.")

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
    async def _call_llm_api(self, prompt_text: str, response_schema: Optional[Dict] = None, temperature: float = 0.2, max_tokens: int = 250) -> str:
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
        self.cumulative_safety_salience += score
        self.cumulative_safety_salience = min(1.0, self.cumulative_safety_salience)

    # --- Pruning old history ---
    def _prune_history(self):
        current_time = self._get_current_time()
        for history_deque in [
            self.recent_cognitive_directives, self.recent_world_model_states,
            self.recent_body_awareness_states, self.recent_performance_reports,
            self.recent_ethical_decisions, self.recent_memory_responses
        ]:
            while history_deque and (current_time - float(history_deque[0].get('timestamp', 0.0))) > self.recent_context_window_s:
                history_deque.popleft()

    # --- Callbacks (generic, ROS or direct calls) ---
    def cognitive_directive_callback(self, msg: Any):
        fields_map = {
            'timestamp': (str(self._get_current_time()), 'timestamp'), 'directive_type': ('', 'directive_type'),
            'target_node': ('', 'target_node'), 'command_payload': ('{}', 'command_payload'),
            'urgency': (0.0, 'urgency'), 'id': ('', 'id')
        }
        data = parse_message_data(msg, fields_map, self.node_name)
        
        if data.get('target_node') == self.node_name and data.get('directive_type') == 'ExecuteAction':
            try:
                payload = json.loads(data.get('command_payload', '{}'))
                action_id = payload.get('action_id')
                if action_id:
                    self.action_queue.append({
                        'action_id': action_id,
                        'command_payload': payload,
                        'source_directive_id': data.get('id', str(uuid.uuid4())),
                        'urgency': data.get('urgency', 0.5),
                        'timestamp': data.get('timestamp', str(self._get_current_time()))
                    })
                    self._update_cumulative_salience(data.get('urgency', 0.0) * 0.8)
                    _log_info(self.node_name, f"Queued action: '{action_id}'. Queue size: {len(self.action_queue)}.")
                else:
                    self._report_error("INVALID_ACTION_DIRECTIVE", "Received ExecuteAction directive with no action_id.", 0.6, {'directive_payload': data.get('command_payload')})
            except json.JSONDecodeError as e:
                self._report_error("JSON_DECODE_ERROR", f"Failed to decode command_payload: {e}", 0.5, {'payload': data.get('command_payload')})
            except Exception as e:
                self._report_error("DIRECTIVE_PROCESSING_ERROR", f"Error processing CognitiveDirective: {e}", 0.7, {'directive': data})
        
        self.recent_cognitive_directives.append(data)
        _log_debug(self.node_name, "Cognitive Directive received for context/action.")

    # Similar adaptations for other callbacks (world_model_state_callback, etc.) - omitted for brevity, follow pattern
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
        if data.get('significant_change_flag', False):
            self._update_cumulative_salience(0.2)
        _log_debug(self.node_name, f"Received World Model State. Significant Change: {data.get('significant_change_flag', False)}.")

    # ... (adapt body_awareness_state_callback, performance_report_callback, ethical_decision_callback, memory_response_callback similarly)

    # --- Core Action Execution Logic ---
    async def execute_action_async(self, action_data: Dict[str, Any], event: Any = None):
        self._prune_history()

        action_id = action_data.get('action_id', 'unknown_action')
        command_payload = action_data.get('command_payload', {})
        source_directive_id = action_data.get('source_directive_id', 'unknown')
        urgency = action_data.get('urgency', 0.5)

        safety_clearance = False
        llm_safety_reasoning = "Not evaluated by LLM."
        
        if self.cumulative_safety_salience >= self.llm_safety_check_threshold_salience or urgency > 0.8:
            _log_info(self.node_name, f"Triggering LLM for safety check of action '{action_id}' (Salience: {self.cumulative_safety_salience:.2f}).")
            context_for_llm = self._compile_llm_context_for_safety_check(action_data)
            llm_safety_output = await self._perform_llm_safety_check(context_for_llm)

            if llm_safety_output:
                safety_clearance = llm_safety_output.get('is_safe', False)
                llm_safety_reasoning = llm_safety_output.get('reasoning', 'LLM provided no specific reasoning.')
                _log_info(self.node_name, f"LLM Safety Check for '{action_id}': Safe={safety_clearance}. Reason: {llm_safety_reasoning[:50]}...")
            else:
                _log_warn(self.node_name, f"LLM safety check failed for '{action_id}'. Falling back to simple rules.")
                safety_clearance, llm_safety_reasoning = self._perform_simple_safety_check(action_data)
        else:
            _log_debug(self.node_name, f"Insufficient cumulative salience ({self.cumulative_safety_salience:.2f}) for LLM. Applying simple rules.")
            safety_clearance, llm_safety_reasoning = self._perform_simple_safety_check(action_data)

        action_success = False
        outcome_summary = "Action not executed due to safety concerns."
        predicted_outcome_match = 0.0

        if safety_clearance:
            _log_info(self.node_name, f"Executing action '{action_id}' with payload: {command_payload}.")
            try:
                # Simulate (replace with real actuators)
                if random.random() < self.action_sim_success_rate:
                    action_success = True
                    outcome_summary = f"Action '{action_id}' executed successfully."
                    predicted_outcome_match = random.uniform(0.8, 1.0)
                else:
                    action_success = False
                    outcome_summary = f"Action '{action_id}' failed during execution (simulated)."
                    predicted_outcome_match = random.uniform(0.0, 0.3)

                _log_info(self.node_name, f"Action '{action_id}' simulation result: {'SUCCESS' if action_success else 'FAILURE'}.")

            except Exception as e:
                action_success = False
                outcome_summary = f"Action '{action_id}' failed during execution: {e}"
                predicted_outcome_match = 0.0
                self._report_error("ACTION_EXECUTION_ERROR", f"Failed to execute action '{action_id}': {e}", 0.9, {'action_payload': command_payload})
        else:
            outcome_summary = f"Action '{action_id}' blocked by safety system. Reason: {llm_safety_reasoning}"
            _log_warn(self.node_name, f"Action '{action_id}' blocked due to safety concerns.")
            self.publish_cognitive_directive(
                directive_type='ActionBlocked',
                target_node='CognitiveControl',
                command_payload=json.dumps({"blocked_action_id": action_id, "reason": llm_safety_reasoning, "urgency": 0.9}),
                urgency=0.9
            )

        # Log and publish
        sensory_snapshot = json.dumps(self.sensory_data)
        self.save_action_log(
            id=str(uuid.uuid4()),
            timestamp=str(self._get_current_time()),
            action_id=action_id,
            command_payload_json=json.dumps(command_payload),
            success=action_success,
            outcome_summary=outcome_summary,
            predicted_outcome_match=predicted_outcome_match,
            safety_clearance=safety_clearance,
            llm_safety_reasoning=llm_safety_reasoning,
            context_snapshot_json=json.dumps(self._compile_llm_context_for_safety_check(action_data)),
            sensory_snapshot_json=sensory_snapshot
        )
        self.publish_action_result(
            timestamp=str(self._get_current_time()),
            action_id=action_id,
            success=action_success,
            outcome_summary=outcome_summary,
            predicted_outcome_match=predicted_outcome_match,
            resources_consumed_json=json.dumps({"power_draw": random.uniform(10, 50)})
        )
        self.cumulative_safety_salience = 0.0

    # --- LLM Safety Check (unchanged, but use _log_*)
    async def _perform_llm_safety_check(self, context_for_llm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt_text = f"""
        You are the Safety & Ethics Guardrail within a robot's Action Execution Module. Your critical task is to evaluate a proposed action for potential safety risks, ethical concerns, or operational hazards based on the robot's current state and knowledge.

        Proposed Action Details:
        --- Action Data ---
        {json.dumps(context_for_llm.get('proposed_action', {}), indent=2)}

        Robot's Recent Context (for Safety Assessment):
        --- Cognitive Context ---
        {json.dumps(context_for_llm.get('recent_cognitive_inputs', {}), indent=2)}

        Sensory Snapshot:
        --- Sensory Data ---
        {json.dumps(context_for_llm.get('sensory_snapshot', {}), indent=2)}

        Based on this context, determine if the `proposed_action` is safe to execute. Provide:
        1.  `is_safe`: boolean (True if the action is deemed safe, False otherwise).
        2.  `reasoning`: string (Detailed explanation for your safety decision, referencing specific risks detected or why it's considered safe).
        3.  `mitigation_suggestions`: string (If unsafe, what steps could mitigate the risk? If safe, what conditions confirm safety?).

        Consider:
        -   **World Model State**: Are there obstacles, unsafe areas, or unexpected changes in `changed_entities`?
        -   **Body Awareness State**: Is the robot `anomaly_detected`? Is its `body_state` (e.g., 'unbalanced', 'low_battery', 'damaged') or `posture_description` making the action risky?
        -   **Performance Report**: Is the robot currently `suboptimal_flag` in relevant areas, suggesting a need for caution?
        -   **Ethical Decision**: Has this `action_proposal_id` received `ethical_clearance`? Is there an `ethical_conflict_flag`?
        -   **Memory Responses**: Are there past `safety_protocol`s or `action_failure` incidents relevant to this action?
        -   **Sensory Inputs**: Vision/sound/instructions indicating immediate risks?
        -   **Action `urgency`**: Does high urgency override minor risks? (Careful here: safety first)
        -   **Action `command_payload`**: What specific parameters of the action itself (`motor_speed`, `grip_force`, `speech_content`) might be unsafe?

        Your response must be in JSON format, containing:
        1.  'timestamp': string (current time)
        2.  'is_safe': boolean
        3.  'reasoning': string
        4.  'mitigation_suggestions': string
        """
        response_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "is_safe": {"type": "boolean"},
                "reasoning": {"type": "string"},
                "mitigation_suggestions": {"type": "string"}
            },
            "required": ["timestamp", "is_safe", "reasoning", "mitigation_suggestions"]
        }

        llm_output_str = await self._call_llm_api(prompt_text, response_schema, temperature=0.2, max_tokens=300)

        if not llm_output_str.startswith("Error:"):
            try:
                llm_data = json.loads(llm_output_str)
                if 'is_safe' in llm_data:
                    llm_data['is_safe'] = bool(llm_data['is_safe'])
                return llm_data
            except json.JSONDecodeError as e:
                self._report_error("LLM_PARSE_ERROR", f"Failed to parse LLM response for safety check: {e}. Raw: {llm_output_str}", 0.8)
                return None
        else:
            self._report_error("LLM_SAFETY_CHECK_FAILED", f"LLM call failed for safety check: {llm_output_str}", 0.9)
            return None

    def _perform_simple_safety_check(self, action_data: Dict[str, Any]) -> tuple[bool, str]:
        action_id = action_data.get('action_id', '')
        command_payload = action_data.get('command_payload', {})
        
        is_safe = True
        reasoning = "Passed basic safety checks."

        current_time = self._get_current_time()

        # Rule 1: Critical body anomalies
        for state in reversed(self.recent_body_awareness_states):
            time_since_state = current_time - float(state.get('timestamp', 0.0))
            if time_since_state < 2.0 and state.get('anomaly_detected', False) and state.get('anomaly_severity', 0.0) > 0.7:
                is_safe = False
                reasoning = f"Critical body anomaly detected: {state.get('body_state')} (Severity: {state.get('anomaly_severity')}). Action '{action_id}' blocked."
                _log_warn(self.node_name, f"Simple safety rule: {reasoning}")
                return is_safe, reasoning

        # Rule 2: Ethical clearance
        action_proposal_id = action_data.get('source_directive_id')
        if action_proposal_id:
            for ethical_dec in reversed(self.recent_ethical_decisions):
                if ethical_dec.get('action_proposal_id') == action_proposal_id:
                    if not ethical_dec.get('ethical_clearance', True):
                        is_safe = False
                        reasoning = f"Action '{action_id}' lacks ethical clearance. Reason: {ethical_dec.get('ethical_reasoning')}"
                        _log_warn(self.node_name, f"Simple safety rule: {reasoning}")
                        return is_safe, reasoning
                    if ethical_dec.get('conflict_flag', False):
                        is_safe = False
                        reasoning = f"Action '{action_id}' has ethical conflicts flagged: {ethical_dec.get('ethical_reasoning')}. Blocked for re-evaluation."
                        _log_warn(self.node_name, f"Simple safety rule: {reasoning}")
                        return is_safe, reasoning

        # Rule 3: Dangerous payload params
        if action_id == 'move_joint' and command_payload.get('velocity', 0) > 5.0:
            is_safe = False
            reasoning = f"Action '{action_id}' involves potentially unsafe high velocity ({command_payload['velocity']})."
            _log_warn(self.node_name, f"Simple safety rule: {reasoning}")
            return is_safe, reasoning
        
        if action_id == 'grip_object' and command_payload.get('force', 0) > 100.0:
            is_safe = False
            reasoning = f"Action '{action_id}' involves potentially unsafe high grip force ({command_payload['force']})."
            _log_warn(self.node_name, f"Simple safety rule: {reasoning}")
            return is_safe, reasoning

        return is_safe, reasoning

    def _compile_llm_context_for_safety_check(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        context = {
            "current_time": self._get_current_time(),
            "proposed_action": action_data,
            "current_robot_state": {
                "world_model_state": self.recent_world_model_states[-1] if self.recent_world_model_states else "N/A",
                "body_awareness_state": self.recent_body_awareness_states[-1] if self.recent_body_awareness_states else "N/A",
                "performance_report": self.recent_performance_reports[-1] if self.recent_performance_reports else "N/A",
                "ethical_decision_for_this_action": next((d for d in reversed(self.recent_ethical_decisions) if d.get('action_proposal_id') == action_data.get('source_directive_id')), "N/A")
            },
            "recent_cognitive_inputs": {
                "world_model_changes": list(self.recent_world_model_states),
                "body_awareness_anomalies": list(self.recent_body_awareness_states),
                "performance_issues": list(self.recent_performance_reports),
                "ethical_clearances_conflicts": list(self.recent_ethical_decisions),
                "relevant_memory_responses": [m for m in self.recent_memory_responses if m.get('memories') and any('safety_protocol' in mem.get('category', '') or 'action_failure' in mem.get('category', '') for mem in m.get('memories', []))],
                "cognitive_directives_for_action_execution": [d for d in self.recent_cognitive_directives if d.get('target_node') == self.node_name and d.get('directive_type') == 'ExecuteAction']
            },
            "sensory_snapshot": self.sensory_data
        }
        
        # Parse nested JSON
        for category_key in context["current_robot_state"]:
            item = context["current_robot_state"][category_key]
            if isinstance(item, dict):
                for field, value in list(item.items()):
                    if isinstance(value, str) and field.endswith('_json'):
                        try:
                            item[field] = json.loads(value)
                        except json.JSONDecodeError:
                            pass
        for category_key in context["recent_cognitive_inputs"]:
            for item in context["recent_cognitive_inputs"][category_key]:
                if isinstance(item, dict):
                    for field, value in list(item.items()):
                        if isinstance(value, str) and field.endswith('_json'):
                            try:
                                item[field] = json.loads(value)
                            except json.JSONDecodeError:
                                pass
        
        return context

    # --- Database and Publishing Functions ---
    def save_action_log(self, **kwargs: Any):
        try:
            self.cursor.execute('''
                INSERT INTO action_log (id, timestamp, action_id, command_payload_json, success, outcome_summary, 
                                        predicted_outcome_match, safety_clearance, llm_safety_reasoning, 
                                        context_snapshot_json, sensory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs['id'], kwargs['timestamp'], kwargs['action_id'], kwargs['command_payload_json'],
                kwargs['success'], kwargs['outcome_summary'], kwargs['predicted_outcome_match'],
                kwargs['safety_clearance'], kwargs['llm_safety_reasoning'], kwargs['context_snapshot_json'],
                kwargs.get('sensory_snapshot_json', '{}')
            ))
            self.conn.commit()
            _log_debug(self.node_name, f"Saved action log (ID: {kwargs['id']}, Action: {kwargs['action_id']}).")
        except sqlite3.Error as e:
            self._report_error("DB_SAVE_ERROR", f"Failed to save action log: {e}", 0.9)
        except Exception as e:
            self._report_error("UNEXPECTED_SAVE_ERROR", f"Unexpected error in save_action_log: {e}", 0.9)

    def publish_action_result(self, **kwargs: Any):
        try:
            result_data = {
                'timestamp': kwargs['timestamp'],
                'action_id': kwargs['action_id'],
                'success': kwargs['success'],
                'outcome_summary': kwargs['outcome_summary'],
                'predicted_outcome_match': kwargs['predicted_outcome_match'],
                'resources_consumed_json': kwargs['resources_consumed_json']
            }
            if ROS_AVAILABLE and self.ros_enabled and self.pub_action_execution_result:
                if hasattr(ActionExecutionResult, 'data'):  # String fallback
                    self.pub_action_execution_result.publish(String(data=json.dumps(result_data)))
                else:
                    result_msg = ActionExecutionResult(**result_data)
                    self.pub_action_execution_result.publish(result_msg)
            else:
                # Dynamic: Could publish to a queue or callback
                _log_debug(self.node_name, f"Published Action Execution Result for '{kwargs['action_id']}'. Success: {kwargs['success']}.")
            _log_debug(self.node_name, f"Action Execution Result for '{kwargs['action_id']}'. Success: {kwargs['success']}.")
        except Exception as e:
            self._report_error("PUBLISH_ACTION_RESULT_ERROR", f"Failed to publish action result for '{kwargs['action_id']}': {e}", 0.7)

    def publish_cognitive_directive(self, directive_type: str, target_node: str, command_payload: str, urgency: float):
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
            _log_error(self.node_name, f"Failed to issue cognitive directive: {e}")

    def shutdown(self):
        self._shutdown_flag = True
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
    parser = argparse.ArgumentParser(description='Sentience Action Execution Node')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ros-enabled', action='store_true', help='Enable ROS compatibility mode')
    args = parser.parse_args()

    node = None
    try:
        node = ActionExecutionNode(config_file_path=args.config, ros_enabled=args.ros_enabled)
        node.run()
    except KeyboardInterrupt:
        _log_info(node.node_name if node else 'main', "Shutdown requested.")
    except Exception as e:
        _log_error('main', f"Unexpected error: {e}")
    finally:
        if node:
            node.shutdown()
